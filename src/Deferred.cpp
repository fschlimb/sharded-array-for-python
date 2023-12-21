// SPDX-License-Identifier: BSD-3-Clause

/*
  Creation/destruction of Deferreds.
  Implementation of worker loop processing deferred objects.
  This worker loop is executed in a separate thread until the system
  gets shut down.
*/

#include "include/sharpy/Deferred.hpp"
#include "include/sharpy/Mediator.hpp"
#include "include/sharpy/Registry.hpp"
#include "include/sharpy/Service.hpp"
#include "include/sharpy/Transceiver.hpp"
#include "include/sharpy/itac.hpp"
#include "include/sharpy/jit/mlir.hpp"

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <oneapi/tbb/concurrent_queue.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

namespace SHARPY {

// thread-safe FIFO queue holding deferred objects
extern tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

// if needed, object/promise is broadcasted to worker processes
// (for controller/worker mode)
void _dist(const Runable *p) {
  if (getTransceiver() && getTransceiver()->is_cw() &&
      getTransceiver()->rank() == 0)
    getMediator()->to_workers(p);
}

// create a enriched future
Deferred::future_type Deferred::get_future() {
  return {std::move(promise_type::get_future().share()),
          _guid,
          _dtype,
          _shape,
          _device,
          _team};
}

// defer a array-producing computation by adding it to the queue.
// return a future for the resulting array.
// set is_global to false if result is a local temporary which does not need a
// guid
Deferred::future_type defer_array(Runable::ptr_type &&_d, bool is_global) {
  Deferred *d = dynamic_cast<Deferred *>(_d.get());
  if (!d)
    throw std::runtime_error("Expected Deferred Array promise");
  if (is_global) {
    _dist(d);
    if (d->guid() == Registry::NOGUID) {
      d->set_guid(Registry::get_guid());
    }
  }
  auto f = d->get_future();
  Registry::put(f);
  push_runable(std::move(_d));
  return f;
}

// defer a global array producer
void Deferred::defer(Runable::ptr_type &&p) { defer_array(std::move(p), true); }

void Runable::defer(Runable::ptr_type &&p) { push_runable(std::move(p)); }

void Runable::fini() { _deferred.clear(); }

// process promises as they arrive through calls to defer
// This is run in a separate thread until shutdown is requested.
// Shutdown is indicated by a Deferred object which evaluates to false.
// The loop repeatedly creates MLIR functions for jit-compilation by letting
// Deferred objects add their MLIR code until an object can not produce MLIR
// but wants immediate execution (indicated by generate_mlir returning true).
// When execution is needed, the function signature (input args, return
// statement) is finalized, the function gets compiled and executed. The loop
// completes by calling run() on the requesting object.
void process_promises() {
  int vtProcessSym, vtSHARPYClass, vtPopSym;
  VT(VT_classdef, "sharpy", &vtSHARPYClass);
  VT(VT_funcdef, "process", vtSHARPYClass, &vtProcessSym);
  VT(VT_funcdef, "pop", vtSHARPYClass, &vtPopSym);
  VT(VT_begin, vtProcessSym);

  bool done = false;
  jit::JIT jit;
  std::vector<Runable::ptr_type> deleters;

  do {
    ::mlir::OpBuilder builder(&jit.context());
    auto loc = builder.getUnknownLoc();

    // Create a MLIR module
    auto module = builder.create<::mlir::ModuleOp>(loc);
    // Create the jit func
    // create dummy type, we'll replace it with the actual type later
    auto dummyFuncType = builder.getFunctionType({}, {});
    if (false) {
      ::mlir::OpBuilder::InsertionGuard guard(builder);
      // Insert before module terminator.
      builder.setInsertionPoint(module.getBody(),
                                std::prev(module.getBody()->end()));
      auto func = builder.create<::mlir::func::FuncOp>(loc, "_debugFunc",
                                                       dummyFuncType);
      func.setPrivate();
    }
    std::string fname("sharpy_jit");
    auto function =
        builder.create<::mlir::func::FuncOp>(loc, fname, dummyFuncType);
    // create function entry block
    auto &entryBlock = *function.addEntryBlock();
    // Set the insertion point in the builder to the beginning of the function
    // body
    builder.setInsertionPointToStart(&entryBlock);
    // we need to keep runables/deferred/futures alive until we set their values
    // below
    std::vector<Runable::ptr_type> runables;

    jit::DepManager dm(function);
    Runable::ptr_type d;

    if (!deleters.empty()) {
      for (auto &dl : deleters) {
        if (dl->generate_mlir(builder, loc, dm)) {
          assert(!"deleters must generate MLIR");
        }
        runables.emplace_back(std::move(dl));
      }
      deleters.clear();
    } else {
      while (true) {
        VT(VT_begin, vtPopSym);
        _deferred.pop(d);
        VT(VT_end, vtPopSym);
        if (d) {
          if (d->isDeleter()) {
            deleters.emplace_back(std::move(d));
          } else {
            if (d->generate_mlir(builder, loc, dm)) {
              break;
            };
            // keep alive for later set_value
            runables.emplace_back(std::move(d));
          }
        } else {
          // signals system shutdown
          done = true;
          break;
        }
      }
    }

    if (!runables.empty()) {
      // get input buffers (before results!)
      auto input = std::move(dm.finalize_inputs());
      // create return statement and adjust function type
      uint64_t osz = dm.handleResult(builder);
      // also request generation of c-wrapper function
      function->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                        builder.getUnitAttr());
      if (jit.verbose())
        function.getFunctionType().dump();
      // add the function to the module
      module.push_back(function);

      if (osz > 0 || !input.empty()) {
        // compile and run the module
        auto output = jit.run(module, fname, input, osz);
        if (output.size() != osz)
          throw std::runtime_error("failed running jit");

        // push results to deliver promises
        dm.deliver(output, osz);
      } else {
        if (jit.verbose())
          std::cerr << "\tskipping\n";
      }
    } // no else needed

    // now we execute the deferred action which could not be compiled
    if (d) {
      py::gil_scoped_acquire acquire;
      d->run();
      d.reset();
    }
  } while (!done);
}
} // namespace SHARPY
