// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <sharpy/Deferred.hpp>
#include <sharpy/Factory.hpp>
#include <sharpy/MeshSharding.hpp>
#include <sharpy/Transceiver.hpp>

#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/IR/SymbolTable.h>

namespace SHARPY {

// Runable to generate mesh in MLIR
class DeferredMesh : public Runable {
private:
  std::string _name;
  std::vector<int64_t> _shape;

public:
  DeferredMesh() = default;
  // Constructor by perfect forwarding
  template <typename NameType, typename ShapeType>
  DeferredMesh(NameType &&name, ShapeType &&shape)
      : _name(std::forward<NameType>(name)),
        _shape(std::forward<ShapeType>(shape)) {
    if (_name.empty()) {
      throw std::runtime_error("Mesh name cannot be empty");
    }
    if (_shape.empty()) {
      _shape.push_back(getTransceiver()->nranks());
    }
    if (VPROD(_shape) != static_cast<int64_t>(getTransceiver()->nranks())) {
      throw std::runtime_error("Mesh shape does not match number of ranks");
    }
  }

  bool generate_mlir(::mlir::OpBuilder &b, const ::mlir::Location &loc,
                     jit::DepManager &d) override {
    auto n = b.getStringAttr(_name);
    auto s = b.getDenseI64ArrayAttr(_shape);
    mlir::OpBuilder::InsertionGuard guard(b);
    auto m =
        ::mlir::SymbolTable::getNearestSymbolTable(b.getBlock()->getParentOp());
    auto body = ::mlir::cast<::mlir::ModuleOp>(m).getBody();
    b.setInsertionPoint(body, body->begin());
    b.create<mlir::mesh::MeshOp>(loc, n, s);
    return false;
  }

  FactoryId factory() const override { return F_MESH; }
  template <typename S> void serialize(S &ser) {
    throw std::runtime_error("Not implemented");
  }
};

// enqueue a runable to generate mesh in MLIR and return name
std::string Mesh::init_mesh(std::string name, std::vector<int64_t> shape) {
  auto mesh = std::make_unique<DeferredMesh>(name, shape);
  push_runable(std::move(mesh));
  return name;
}

// Runable to generate meshsharding in MLIR
class DeferredMeshSharding : public Runable {
private:
  std::shared_ptr<MeshSharding> _meshSharding;

public:
  DeferredMeshSharding() = default;
  // Constructor by perfect forwarding
  template <typename MeshShardingType>
  DeferredMeshSharding(MeshShardingType &&meshSharding)
      : _meshSharding(std::forward<MeshShardingType>(meshSharding)) {}

  bool generate_mlir(::mlir::OpBuilder &b, const ::mlir::Location &l,
                     jit::DepManager &d) override {
    std::cerr << "Generating MLIR for meshsharding on mesh "
              << _meshSharding->mesh() << " with splitAxes[\n";
    for (auto j : _meshSharding->splitAxes()) {
      std::cerr << "\t[";
      for (auto i : j)
        std::cout << i << ", ";
      std::cerr << "]\n";
    }
    std::cerr << "]\n";
    return false;
  }
  FactoryId factory() const override { return F_MESHSHARDING; }
  template <typename S> void serialize(S &ser) {
    throw std::runtime_error("Not implemented");
  }
};

std::shared_ptr<MeshSharding>
Mesh::init_mesh_sharding(const std::string &mesh,
                         const std::vector<std::vector<int64_t>> &splitAxes) {
  // enqueue a meshsharding operation
  // returns the name as the unique identifier
  static std::string defaultMesh;
  if (mesh.empty()) {
    if (splitAxes.empty()) {
      return {};
    }
    if (defaultMesh.empty()) {
      defaultMesh = init_mesh("defaultmesh", {});
    }
  }
  auto meshSharding = std::make_shared<MeshSharding>(
      mesh.empty() ? defaultMesh : mesh, splitAxes);
  auto deferredMeshSharding =
      std::make_unique<DeferredMeshSharding>(meshSharding);
  push_runable(std::move(deferredMeshSharding));
  return meshSharding;
}

// Runable to generate sharding annotation in MLIR
class DeferredShard : public Deferred {
private:
  id_type _a;
  std::shared_ptr<MeshSharding> _meshSharding;

public:
  DeferredShard() = default;
  // Constructor by perfect forwarding
  template <typename MeshShardingType>
  DeferredShard(const array_i::future_type &a, MeshShardingType &&meshSharding)
      : _a(a.guid()),
        _meshSharding(std::forward<MeshShardingType>(meshSharding)) {}

  bool generate_mlir(::mlir::OpBuilder &b, const ::mlir::Location &l,
                     jit::DepManager &d) override {
    return false;
  }
  FactoryId factory() const override { return F_SHARD; }
  template <typename S> void serialize(S &ser) {
    throw std::runtime_error("Not implemented");
  }
};

// enqueue a Deferred to generate sharding annotation in MLIR and
// return new array
FutureArray *Mesh::shard(const FutureArray &a,
                         const std::shared_ptr<MeshSharding> &meshSharding) {
  return new FutureArray(defer<DeferredShard>(a.get(), meshSharding));
}

FACTORY_INIT(DeferredMesh, F_MESH);
FACTORY_INIT(DeferredMeshSharding, F_MESHSHARDING);
FACTORY_INIT(DeferredShard, F_SHARD);

} // namespace SHARPY
