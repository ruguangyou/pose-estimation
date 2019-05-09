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

#include <fstream>

namespace cfsd {

enum SensorType {
  ACCELEROMETER,
  GYROSCOPE
};

class VisualInertialSLAM {
  public:
    // visual inertial odometry state
    enum VIOstate {
        SYNCHRONIZING, // matching imu and camera timestamp
        SFM, // struct from motion
        INITIALIZING, // initialize imu
        OK, // process
        LOST // caused by lose track or corrupted bias, need re-initialization
    };

  public:
    VisualInertialSLAM(const bool verbose);

    #ifdef USE_VIEWER
    void setViewer(const cfsd::Ptr<Viewer>& pViewer) { _pMap->_pViewer = pViewer; }
    #endif

    bool process(const cv::Mat& grayL, const cv::Mat& grayR, const long& imgTimestamp);

    void collectImuData(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z);

    void saveResults();

  private:
    const bool _verbose;
    
    VIOstate _state;

    cfsd::Ptr<CameraModel> _pCameraModel;

    cfsd::Ptr<Map> _pMap;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    cfsd::Ptr<Optimizer> _pOptimizer;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    Eigen::Vector3d _gyr, _acc;
    bool _gyrGot{false}, _accGot{false};

    // bool _readyToAlign{false};

    // bool _gyrInitialized{false};
    // bool _poseInitialized{false};
    // bool _accInitialized{false};

    int _sfmCount{0};
    
};

} // namespace cfsd

#endif // VISUAL_INERTIAL_SLAM_HPP