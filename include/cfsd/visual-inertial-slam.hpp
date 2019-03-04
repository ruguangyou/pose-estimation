#ifndef VISUAL_INERTIAL_SLAM_HPP
#define VISUAL_INERTIAL_SLAM_HPP

#include "cfsd/camera-model.hpp"
#include "cfsd/feature-tracker.hpp"
// #include "cfsd/map.hpp"

namespace cfsd {

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

    void process(const long& timestamp, const cv::Mat& img);

  private:
    bool _verbose;
    VIOstate _state;

    cfsd::Ptr<CameraModel> _pCameraModel;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    // Map::Ptr _map;
    
};

} // namespace cfsd

#endif // VISUAL_INERTIAL_SLAM_HPP