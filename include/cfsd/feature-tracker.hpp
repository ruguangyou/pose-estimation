#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/camera-frame.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/feature2d/feature2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

class FeatureTracker {
  public:
    enum DetectorType { ORB = 0, BRISK = 1 };
    using Ptr = std::shared_ptr<FeatureTracker>;
  
    // default constructor and deconstructor
    FeatureTracker();
    ~FeatureTracker();
    FeatureTracker(DetectorType type, bool verbose);

    // factory function
    static FeatureTracker::Ptr create(DetectorType type, bool verbose);

    // extract and match keypoints
    void extractKeypoints(); // extract current camera keypoints
    void matchKeypoints();   // match current frame and key frame
    bool curIsKeyFrame();    // determine if current frame would be a key frame
    
    // compute camera pose:
    // shoule use RANSAC scheme for outlier rejection,
    // and solve 3D-2D PnP problem (in particular, P3P problem)
    void computePose(Sophus::SE3d& pose);

  private: 
    // unsigned long _id;
    // double _timestamp;
    bool _verbose, _debug;
    DetectorType _detectorType;
    
    KeyFrame::Ptr _keyRef; // reference frame is a key frame
    CameraFrame::Ptr _camCur;

    // keypoints and descriptors of current camera frame's left image
    std::vector<cv::KeyPoint> _keypoints;
    cv::Mat _descriptors;

    // detector
    // cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BRISK> _brisk;
    
    // feature matching parameters
    std::vector<cv::DMatch> _matches;
    float _matchRatio;    // ratio for selecting good matches
    float _minMatchDist;  // min match distance, based on experience, e.g. 30.0f

    // key frame selection parameters
    float _matchThresLow; // lower bound of matching percentage
    float _matchThresUp;  // upper bound of matching percentage

    // percentage of good matches in total matches, for key frame selection
    float _matchPercent;

  public:
    inline const KeyFrame::Ptr& getKeyFrame() const { return _keyRef; }
    inline const CameraFrame::Ptr& getCamFrame() const {return _camCur; }

    void setKeyFrame(KeyFrame::Ptr keyFrame);
    void setCameraFrame(CameraFrame::Ptr camFrame);

};

} // namespace cfsd

#endif // FEATURE_TRACKER_HPP