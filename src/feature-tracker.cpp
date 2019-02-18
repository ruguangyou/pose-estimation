#include "cfsd/feature-tracker.hpp"

namespace cfsd {

FeatureTracker::FeatureTracker() {}

FeatureTracker::~FeatureTracker() {}

FeatureTracker::Ptr FeatureTracker::create() {}

void FeatureTracker::extrackKeypoints() {
    // extract keypoints from the left image of current camera frame
    // todo: provide different detectors; temperarily use ORB detector
    // todo: detector initialization in constructor, e.g. _orb = cv::ORB::create(...)
    switch(_detectorType) {
        case ORB:
            _orb->detect(_camCur->getImgLeft(), _keypoints);
            _orb->compute(_camCur->getImgLeft(), _keypoints, _descriptors);
            break;
        case BRISK:

            break;
        default:
            
    }
}

void FeatureTracker::matchKeypoints() {
    // match descriptors of current frame with reference frame
    std::vector<cv::DMatch> matches;
    // maybe could try other matchers?
    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force match
    matcher.match(_keyRef->getDescriptors(), _descriptors, matches);
    
    // select best matches
    float min_dist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance })->distance;
        // syntax: 
        // 1) function template performs argument deduction, so we can use a simple form instead of min_element<T>(...)
        // 2) min_element returns an iterator, in this case, cv::DMatch iterator
        // 3) when using iterator to access object's data member, dereference is needed, similar usage as pointer

    _matches.clear();
    for (cv::DMatch& m : matches) {
        if (m.distance < std::max(min_dist * _matchRatio, _minMatchDist)) {
            _matches.push_back(m);
        }
    }

    _matchPercent = _matches.size() / matches.size();
    if (_verbose) {
        std::cout << "[REF-CUR] number of total matches: " << matches.size() << std::endl;
        std::cout << "[REF-CUR] number of good matches: " << _matches.size() << std::endl;
    }
}

bool FeatureTracker::curIsKeyFrame() {
    return (_matchPercent >= _matchThresLow && _matchPercent <= _matchThresUp);
}

Sophus::SE3d FeatureTracker::computePose() {
    // estimate camera pose by solving 3D-2D PnP problem using RANSAC scheme
    std::vector<cv::Point3f> points3D;  // 3D points triangulated from reference frame
    std::vector<cv::Point2f> points2D;  // 2D points in current frame

    const std::vector<cv::Point3f>& points3DRef = _keyRef.getPoints3D();
    for (cv::DMatch& m : _matches) {
        // matches is computed from two sets of descriptors: matcher.match(_descriptorsRef, _descriptorsCur, matches)
        // the queryIdx corresponds to _descriptorsRef
        // the trainIdx corresponds to _descriptorsCur
        // the index of descriptors corresponds to the index of keypoints
        points3D.push_back(_points3DRef[m.queryIdx]);
        points2D.push_back(_keypoints[n.trainIdx].pt);
    }

    cv::Mat camMatrix, distCoeffs, rvec, tvec, inliers;
    cv::eigen2cv(_camCur.getCamLeft(), camMatrix);
    cv::eigen2cv(_camCur.getDistLeft(), distCoeffs);
    // if the 3D points is world coordinates, the computed transformation (rvec 
    //    and tvec) is from world coordinate system to camera coordinate system
    // if the 3D points is left camera coordinates of keyframe, the transformation
    //    is from keyframe's left camera to current left camera
    // here, assume world coordinates are used, then rvec and tvec describe left camera pose relative to world
    cv::solvePnPRansac(points3D, points2D, camMatrix, distCoeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers, SOLVEPNP_ITERATIVE);

    cv::Mat cvR;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    cv::Rodrigues(rvec, cvR);
    cv::cv2eigen(cvR, R);
    cv::cv2eigen(tvec, t);
    return Sophus::SE3d(R, t);
}


} // namespace cfsd