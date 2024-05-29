// SPDX-License-Identifier: BSD-3-Clause

#include "FutureArray.hpp"
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <string>
#include <vector>

namespace SHARPY {

/// @brief MeshSharding class
/// Defines how a potential array is mapped onto the given mesh
/// See MLIR's Mesh dialect for sharding semantics
class MeshSharding {
  std::string _mesh;
  std::vector<::llvm::SmallVector<int16_t>> _splitAxes;

public:
  // Constructor by perfect forwarding
  template <typename MeshType, typename SplitAxesType>
  MeshSharding(MeshType &&mesh, SplitAxesType &&splitAxes)
      : _mesh(std::forward<MeshType>(mesh)),
        _splitAxes(std::forward<SplitAxesType>(splitAxes)) {}
  template <typename MeshType>
  MeshSharding(MeshType &&mesh,
               const std::vector<std::vector<int16_t>> &splitAxes)
      : _mesh(std::forward<MeshType>(mesh)), _splitAxes(splitAxes.size()) {
    auto sa = _splitAxes.begin();
    for (auto a : splitAxes) {
      sa->append(a.begin(), a.end());
      ++sa;
    }
  }
  // Accessors
  const std::string &mesh() const { return _mesh; }
  const std::vector<::llvm::SmallVector<int16_t>> &splitAxes() const {
    return _splitAxes;
  }
};

struct Mesh {
  /// @brief Create a (device) mesh with the given name and shape
  /// @param name unique identifier for the mesh
  /// @param shape shape of the mesh
  /// @return unique name of the mesh
  /// @note The shape must be a product of the number of ranks
  /// @note If shape is empty, assumes default shape [nranks]
  static std::string init_mesh(std::string name, std::vector<int64_t> shape);

  /// @brief Create a mesh sharding for given mesh and split axes
  /// @param mesh name of mesh
  /// @param splitAxes Vector of Vector of split dimensions, one vector for each
  /// dimension of the mesh
  /// @return MeshSharding object
  /// @note if mesh and splitAxis are empty returns nullptr, which represents no
  /// sharding
  /// @note if mesh is empty and splitAxis is non-empty uses the default mesh
  /// @note if mesh is non-empty and splitAxis is empty, returns a sharding
  /// along the first dimension of the mesh and target
  static std::shared_ptr<MeshSharding>
  init_mesh_sharding(const std::string &mesh,
                     const std::vector<std::vector<int16_t>> &splitAxes);

  /// @brief Shard the given FutureArray based on the given mesh sharding
  static FutureArray *shard(const FutureArray &a,
                            const std::shared_ptr<MeshSharding> &meshSharding);
};

} // namespace SHARPY
