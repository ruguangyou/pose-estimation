#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "cfsd/common.hpp"
#include "cfsd/imu-preintegrator.hpp"

#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"
#endif

#include <ceres/ceres.h>

namespace cfsd {

class ImuPreintegrator;

class Optimizer {
  public:
    Optimizer(const bool verbose);

    #ifdef USE_VIEWER
    void setViewer(const cfsd::Ptr<Viewer>& pViewer) { _pViewer = pViewer; }
    #endif

    void localOptimize(const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator);

  private:
    bool _verbose;

    #ifdef USE_VIEWER
    cfsd::Ptr<Viewer> _pViewer;
    #endif
};

} // namespace cfsd

#endif // OPTIMIZER_HPP