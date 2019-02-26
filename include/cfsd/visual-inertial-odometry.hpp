#ifndef VISUAL_INERTIAL_ODOMETRY_HPP
#define VISUAL_INERTIAL_ODOMETRY_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-frame.hpp"
#include "cfsd/imu-frame.hpp"
#include "cfsd/key-frame.hpp"
#include "cfsd/feature-tracker.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/feature2d/feature2d.hpp>
#include <opencv2/imgproc/imgproc.cpp>
#include <opencv2/highgui/highgui.cpp>

namespace cfsd {

class VisualInertialOdometry {
  public:
    enum VIOstate { INITIALIZING, OK, LOST };
    using Ptr = std::shared_ptr<VisualInertialOdometry>; // typedef std::shared_ptr<VisualInertialOdometry> Ptr;

  public:
    VisualInertialOdometry();
    ~VisualInertialOdometry();

    static VisualInertialOdometry::Ptr create();
    
    // several feature tracking methods are available to choose
    // todo: compare the performance of these methods
    // void featureTracking(Detector d);

    // void bundleAdjustment();

    // void imuPreintegration();

    void processFrame()
    void addKeyFrame();


  private:
    VIOstate _state;

    FeatureTracker::Ptr _featureTracker;
    std::vector<KeyFrame::Ptr> _keyFrames;
};

} // namespace cfsd

#endif // VISUAL_INERTIAL_ODOMETRY_HPP