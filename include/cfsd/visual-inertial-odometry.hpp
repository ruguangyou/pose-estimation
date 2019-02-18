#ifndef VISUAL_INERTIAL_ODOMETRY_HPP
#define VISUAL_INERTIAL_ODOMETRY_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-frame.hpp"
#include "cfsd/imu-frame.hpp"
#include "cfsd/key-frame.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/feature2d/feature2d.hpp>
#include <opencv2/imgproc/imgproc.cpp>
#include <opencv2/highgui/highgui.cpp>

namespace cfsd {

class VisualInertialOdometry {
  public:
    enum VIOstate { INITIALIZING = -1, OK = 0, LOST };
    using Ptr = std::shared_ptr<VisualInertialOdometry>; // typedef std::shared_ptr<VisualInertialOdometry> Ptr;

  public:
    VisualInertialOdometry();
    ~VisualInertialOdometry();

    static VisualInertialOdometry::Ptr create();
    
    // several feature tracking methods are available to choose
    // todo: compare the performance of these methods
    // void featureTracking(Detector d);

    void bundleAdjustment();

    // void imuPreintegration();

    void addKeyFrame();


  protected:
    // detect and match keypoints
    // void extractKeypoints();
    // void matchKeypoints();
    // void triangulate();

    // select key frame
    // bool isKeyFrame();

    // preintegration
    // void preintegrate();


  private:
    FeatureTracker::Ptr _featureTracker;

    std::vector<KeyFrame::Ptr> _keyFrames;
    

    cv::Ptr<cv::ORB> _orb;  // orb detector

};

} // namespace cfsd

#endif // VISUAL_INERTIAL_ODOMETRY_HPP