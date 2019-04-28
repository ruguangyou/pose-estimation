#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "cfsd/cost-functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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
    void motionOnlyBA(const cv::Mat& img);

    // Estimate initial IMU bias, align initial IMU acceleration to gravity.
    void initialGravityVelocity();
    
    void initialAlignment();

    void initialGyrBias();

    void initialAccBias();

    // void localOptimize();

  private:
    bool _verbose;

    cfsd::Ptr<Map> _pMap;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    cfsd::Ptr<CameraModel> _pCameraModel;

    double _pose[WINDOWSIZE][6];  // pose (rotation vector, translation vector / position)
    double _v_bga[WINDOWSIZE][9]; // velocity, bias of gyroscope, bias of accelerometer

  // public:
  //   std::vector<Eigen::Vector3d> _accs;
  //   std::vector<Eigen::Vector3d> _gyrs;
};

} // namespace cfsd

#endif // OPTIMIZER_HPP