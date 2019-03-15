#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "cfsd/common.hpp"

#include <ceres/ceres.h>

namespace cfsd {

class Optimizer {
  public:
    Optimizer(const bool verbose);

  private:
    bool _verbose;
};

} // namespace cfsd

#endif // OPTIMIZER_HPP