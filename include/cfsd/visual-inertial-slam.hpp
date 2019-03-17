#ifndef VISUAL_INERTIAL_SLAM_HPP
#define VISUAL_INERTIAL_SLAM_HPP

#include "cfsd/camera-model.hpp"
#include "cfsd/feature-tracker.hpp"
#include "cfsd/imu-preintegrator.hpp"
#include "cfsd/optimizer.hpp"
// #include "cfsd/map.hpp"

#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"
#endif

namespace cfsd {

enum SensorType {
  ACCELEROMETER,
  GYROSCOPE
};

class VisualInertialSLAM {
  public:
    // visual inertial odometry state
    enum VIOstate {
        // INITIALIZING,
        OK,
        LOST
    };

  public:
    VisualInertialSLAM(const bool verbose);

    #ifdef USE_VIEWER
    void setViewer(const cfsd::Ptr<Viewer>& pViewer) { _pOptimizer->setViewer(pViewer); }
    #endif

    void processImage(const long& timestamp, const cv::Mat& img);

    void processImu();

    void collectImuData(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z);

  private:
    bool _verbose;
    VIOstate _state;

    cfsd::Ptr<CameraModel> _pCameraModel;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    cfsd::Ptr<Optimizer> _pOptimizer;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    // cfsd::Ptr<Map> _pMap;
    
};

} // namespace cfsd

#endif // VISUAL_INERTIAL_SLAM_HPP