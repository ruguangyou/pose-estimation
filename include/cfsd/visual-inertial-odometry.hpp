#ifndef VISUAL_INERTIAL_ODOMETRY_HPP
#define VISUAL_INERTIAL_ODOMETRY_HPP

#include "cfsd/common.hpp"
#include "cfsd/feature-tracker.hpp"
// #include "cfsd/map.hpp"

namespace cfsd {

class VisualInertialOdometry {
  public:
    enum VIOstate { INITIALIZING, RUNNING, LOST };
    using Ptr = std::shared_ptr<VisualInertialOdometry>; // typedef std::shared_ptr<VisualInertialOdometry> Ptr;

  public:
    VisualInertialOdometry(bool verbose, bool debug);

    static VisualInertialOdometry::Ptr create(bool verbose, bool debug);
    
    // several feature tracking methods are available to choose
    // todo: compare the performance of these methods
    // void featureTracking(Detector d);

    // void bundleAdjustment();

    // void imuPreintegration();

    void processFrame(long timestamp, cv::Mat& img);

  private:
    bool _verbose, _debug;
    VIOstate _state;

    FeatureTracker::Ptr _featureTracker;
    std::vector<KeyFrame::Ptr> _keyFrames;
    std::vector<std::vector<cv::DMatch>> _frameMatches;
    std::vector<SophusSE3Type> _camPoses;
    // Map::Ptr _map;

  public:
    inline const SophusSE3Type& getLatestCamPose() const { return _camPoses.back(); }
};

} // namespace cfsd

#endif // VISUAL_INERTIAL_ODOMETRY_HPP