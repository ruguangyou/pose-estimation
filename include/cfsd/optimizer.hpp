#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "cfsd/common.hpp"
#include "cfsd/imu-preintegrator.hpp"

#include <ceres/ceres.h>

namespace cfsd {

class ImuPreintegrator;

class Optimizer {
  public:
    Optimizer(const bool verbose);

    void localOptimize(const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator);

  private:
    bool _verbose;
};

} // namespace cfsd

#endif // OPTIMIZER_HPP