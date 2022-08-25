// SPDX-License-Identifier: BSD-3-Clause

#include "include/ddptensor/Deferred.hpp"
#include "include/ddptensor/Transceiver.hpp"
#include "include/ddptensor/Mediator.hpp"
#include "include/ddptensor/Registry.hpp"

#include <oneapi/tbb/concurrent_queue.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <iostream>
#include <unordered_set>

static tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

void push_runable(Runable::ptr_type && r)
{
    _deferred.push(std::move(r));
}

void _dist(const Runable * p)
{
    if(is_cw() && theTransceiver->rank() == 0)
        theMediator->to_workers(p);
}

Deferred::future_type Deferred::get_future()
{
    return {std::move(promise_type::get_future().share()), _guid, _dtype, _rank};
}

Deferred::future_type defer_tensor(Runable::ptr_type && _d, bool is_global)
{
    Deferred * d = dynamic_cast<Deferred*>(_d.get());
    if(!d) throw std::runtime_error("Expected Deferred Tensor promise");
    if(is_global) {
        _dist(d);
        d->set_guid(Registry::get_guid());
    }
    auto f = d->get_future();
    Registry::put(f);
    push_runable(std::move(_d));
    return f;
}

void Deferred::defer(Runable::ptr_type && p)
{
    defer_tensor(std::move(p), true);
}

void Runable::defer(Runable::ptr_type && p)
{
    push_runable(std::move(p));
}

void Runable::fini()
{
    _deferred.clear();
}

void process_promises()
{
    bool done = false;
    jit::JIT jit;

    do {
        ::mlir::OpBuilder builder(&jit._context);
        auto loc = builder.getUnknownLoc();
    
        // Create a MLIR module
        auto module = builder.create<::mlir::ModuleOp>(loc);
        // Create a func
        auto dtype = builder.getI64Type();
        // create dummy type, we'll replace it with the actual type later
        auto dummyFuncType = builder.getFunctionType({}, dtype);
        std::string fname("ddpt_jit");
        auto function = builder.create<::mlir::func::FuncOp>(loc, fname, dummyFuncType);
        // create function entry block
        auto &entryBlock = *function.addEntryBlock();
        // Set the insertion point in the builder to the beginning of the function body
        builder.setInsertionPointToStart(&entryBlock);
        // we need to keep runables/deferred/futures alive until we set their values below
        std::vector<Runable::ptr_type> runables;

        jit::DepManager dm(function);

        while(true) {
            Runable::ptr_type d;
            _deferred.pop(d);
            if(d) {
                if(d->generate_mlir(builder, loc, dm)) {
                    d.reset();
                    break;
                };
                // keep alive for later set_value
                runables.push_back(std::move(d));
            } else {
                // signals system shutdown
                done = true;
                break;
            }
        }

        if(runables.empty()) continue;

        // create return statement and adjust function type
        uint64_t sz = dm.handleResult(builder);
        // also request generation of c-wrapper function
        function->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(), ::mlir::UnitAttr::get(&jit._context));
        // add the function to the module
        module.push_back(function);
        module.dump();

        // compile and run the module
        assert(sizeof(intptr_t) == sizeof(void*));
        intptr_t * output = new intptr_t[sz];
        if(jit.run(module, fname, output)) throw std::runtime_error("failed running jit");

        // push results to deliver promises
        dm.deliver(output, sz);
    } while(!done);
}

void sync()
{
    // FIXME this does not wait for the last deferred to complete
    while(!_deferred.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

