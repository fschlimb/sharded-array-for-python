#include "ddptensor/Factory.hpp"

std::vector<Factory::ptr_type> s_factories(FACTORY_LAST);

const Factory *Factory::get(FactoryId id) { return s_factories[id].get(); }

void Factory::put(Factory::ptr_type &&factory) {
  auto id = factory->id();
  s_factories[id] = std::move(factory);
}
