#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-frame.hpp"

#include <opencv2/feature2d/feature2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

class FeatureTracker {
  public:
    enum DetectorType { ORB, BRISK };
    using Ptr = std::shared_ptr<FeatureTracker>;
    static DetectorType _detectorType;
  
  public:
    FeatureTracker();
    ~FeatureTracker();
    static FeatureTracker::Ptr create();

    // extract and match keypoints
    void extractKeypoints();
    void matchKeypoints();
    bool curIsKeyFrame();
    
    // compute camera pose:
    // shoule use RANSAC scheme for outlier rejection,
    // and solve 3D-2D PnP problem (in particular, P3P problem)
    Sophus::SE3d computePose();

  private: 
    unsigned long _id;
    double _timestamp;
    bool _verbose;

    KeyFrame::Ptr _keyRef; // reference frame is a key frame
    CameraFrame::Ptr _camCur;

    // keypoints and descriptors of current camera frame's left image
    std::vector<cv::KeyPoint> _keypoints;
    cv::Mat _descriptors;

    // detector
    // cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BRISK> _brisk;
    
    // feature matches
    std::vector<cv::DMatch> _matches;
    float _matchRatio; // ratio for selecting good matches
    float _minMatchDist; // min match distance, based on experience, e.g. 30.0f

    // key frame selection
    float _matchPercent;  // percentage of good matches in total matches, for key frame selection
    float _matchThresLow; // lower bound of matching percentage
    float _matchThresUp;  // upper bound of matching percentage

};

} // namespace cfsd

#endif // FEATURE_TRACKER_HPP