// SPDX-License-Identifier: BSD-3-Clause

#include "FutureArray.hpp"
#include <memory>
#include <string>
#include <vector>

namespace SHARPY {

class MeshSharding {
  std::string _mesh;
  std::vector<std::vector<int64_t>> _splitAxes;

public:
  // Constructor by perfect forwarding
  template <typename MeshType, typename SplitAxesType>
  MeshSharding(MeshType &&mesh, SplitAxesType &&splitAxes)
      : _mesh(std::forward<MeshType>(mesh)),
        _splitAxes(std::forward<SplitAxesType>(splitAxes)) {}
  // Accessors
  const std::string &mesh() const { return _mesh; }
  const std::vector<std::vector<int64_t>> &splitAxes() const {
    return _splitAxes;
  }
};

struct Mesh {
  static std::string init_mesh(std::string name, std::vector<int64_t> shape);
  static std::shared_ptr<MeshSharding>
  init_mesh_sharding(const std::string &mesh,
                     const std::vector<std::vector<int64_t>> &splitAxes);
  static FutureArray *shard(const FutureArray &a,
                            const std::shared_ptr<MeshSharding> &meshSharding);
};

} // namespace SHARPY
