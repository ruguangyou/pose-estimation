#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-model.hpp"
#include "cfsd/map.hpp"
#include "cfsd/feature-tracker.hpp"
#include "cfsd/imu-preintegrator.hpp"

#include <ceres/ceres.h>

namespace cfsd {

class Optimizer {
  public:
    Optimizer(const cfsd::Ptr<Map>& _pMap, const cfsd::Ptr<FeatureTracker>& pFeatureTracker, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose);

    // ~Optimizer();

    /* Only optimize motion (i.e. vehicle states), keep landmarks fixed.
       map points: (x x   x  x  x x)  <- fixed
                   /| |\ /|\ | /| |\
           frames: #  #  #  #  #  #  #  $    (# is keyframe, $ is latest frame)
                   | <=fixed=> |  | <=> | <- local-window to be optimizer
    */
    void motionOnlyBA();

    // void localOptimize();

  private:
    bool _verbose;

    cfsd::Ptr<Map> _pMap;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    cfsd::Ptr<CameraModel> _pCameraModel;

    double _pose[WINDOWSIZE][6];  // pose (rotation vector, translation vector / position)
    double _v_bga[WINDOWSIZE][9]; // velocity, bias of gyroscope, bias of accelerometer
};

} // namespace cfsd

#endif // OPTIMIZER_HPP