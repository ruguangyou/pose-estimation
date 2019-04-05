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
    void setViewer(const cfsd::Ptr<Viewer>& pViewer) { _pMap->_pViewer = pViewer; }
    #endif

    void processImage(const cv::Mat& grayL, const cv::Mat& grayR);

    void processImu(const long& timestamp, const double& gyrX, const double& gyrY, const double& gyrZ, const double& accX, const double& accY, const double& accZ);

    // void optimize();

    void setImgTimestamp(const long& timestamp);

    void collectImuData(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z);

  private:
    bool _verbose;
    VIOstate _state;

    cfsd::Ptr<CameraModel> _pCameraModel;

    cfsd::Ptr<Map> _pMap;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    cfsd::Ptr<Optimizer> _pOptimizer;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    double _gyrX, _gyrY, _gyrZ;
    double _accX, _accY, _accZ;
    bool _gyrGot{false}, _accGot{false};
    
};

} // namespace cfsd

#endif // VISUAL_INERTIAL_SLAM_HPP