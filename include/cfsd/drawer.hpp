#ifndef DRAWER_HPP
#define DRAWER_HPP

#include "cfsd/common.hpp"
#include ""

namespace cfsd {

class Drawer {
  public:
    using Ptr = std::shared_ptr<Drawer>;
    Drawer() {}
    ~Drawer() {}

    static Drawer::Ptr create();
  private:


};

} // namespace cfsd

#endif // DRAWER_HPP