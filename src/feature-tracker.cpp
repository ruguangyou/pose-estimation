#include "cfsd/feature-tracker.hpp"

namespace cfsd {

FeatureTracker::FeatureTracker() {}

FeatureTracker::~FeatureTracker() {}

FeatureTracker::FeatureTracker(bool verbose, bool debug) : _verbose(verbose), _debug(debug) {
    std::string type = Config::get<std::string>("detectorType");
    if (type == "ORB") {
        _detectorType = ORB;
    }
    else if (type == "BRISK") {
        _detectorType = BRISK;
    }
    else {
        std::cout << "Unexpected type, will use ORB as default detector" << std::endl;
        _detectorType = ORB; 
    }

    _matchRatio = Config::get<float>("matchRatio");
    _minMatchDist = Config::get<float>("minMatchDist");
    _matchThresLow = Config::get<float>("matchThresLow");
    _matchThresUp = Config::get<float>("matchThresUp");
    
    switch(_detectorType) {
        case ORB:
        { // give a scope, s.t. local variable can be declared
            if (_verbose) { std::cout << "Setup ORB detector" << std::endl; }
            int numberOfFeatures = Config::get<int>("numberOfFeatures");
            float scaleFactor = Config::get<float>("scaleFactor");
            int levelPyramid = Config::get<int>("levelPyramid");
            _orb = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid);
            break;
        }
        case BRISK:
        {
            if (_verbose) { std::cout << "Setup BRISK detector" << std::endl; }
            break;
        }
    }
}

FeatureTracker::Ptr FeatureTracker::create(bool verbose = false, bool debug = false) {
    return FeatureTracker::Ptr(new FeatureTracker(verbose, debug));
}

void FeatureTracker::extractKeypoints() {
    // extract keypoints from the left image of current camera frame
    // todo: provide different detectors; temperarily use ORB detector
    // todo: detector initialization in constructor, e.g. _orb = cv::ORB::create(...)
    switch(_detectorType) {
        case ORB:
            _orb->detectAndCompute(_camCur->getImgLeft(), cv::noArray(), _keypoints, _descriptors);
            break;
        case BRISK:
            _brisk->detectAndCompute(_camCur->getImgLeft(), cv::noArray(), _keypoints, _descriptors);
            break;
    }
}

void FeatureTracker::matchKeypoints() {
    // match descriptors of current frame with reference frame
    std::vector<cv::DMatch> matches;
    // maybe could try other matchers?
    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force match
    matcher.match(_keyRef->getDescriptors(), _descriptors, matches);
    
    // select best matches
    float min_dist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
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

    _matchPercent = (float) _matches.size() / _keyRef->getNumOfPoints();
    if (_debug) {
        std::cout << "[REF-CUR] number of good matches: " << _matches.size() << std::endl;
        std::cout << "[REF-CUR] number of total matches: " << _keyRef->getNumOfPoints() << std::endl;
        std::cout << "[REF-CUR] matching percentage: " << _matchPercent << std::endl;
    }
}

bool FeatureTracker::curIsKeyFrame() {
    return (_matchPercent >= _matchThresLow && _matchPercent <= _matchThresUp);
}

void FeatureTracker::computeCamPose(SophusSE3Type& pose) {
    // estimate camera pose by solving 3D-2D PnP problem using RANSAC scheme
    std::vector<cvPoint3Type> points3D;  // 3D points triangulated from reference frame
    std::vector<cvPoint2Type> points2D;  // 2D points in current frame

    const std::vector<cvPoint3Type>& points3DRef = _keyRef->getPoints3D();
    for (cv::DMatch& m : _matches) {
        // matches is computed from two sets of descriptors: matcher.match(_descriptorsRef, _descriptorsCur, matches)
        // the queryIdx corresponds to _descriptorsRef
        // the trainIdx corresponds to _descriptorsCur
        // the index of descriptors corresponds to the index of keypoints
        points3D.push_back(points3DRef[m.queryIdx]);
        points2D.push_back(_keypoints[m.trainIdx].pt);
        if (_debug) {
            std::cout << "3D points in world coordinate: " << points3DRef[m.queryIdx] << ", and the pixel in image: " << _keypoints[m.trainIdx].pt << std::endl;
        }
    }

    cv::Mat camMatrix, distCoeffs, rvec, tvec, inliers;
    cv::eigen2cv(_camCur->getCamLeft(), camMatrix);
    cv::eigen2cv(_camCur->getDistLeft(), distCoeffs);
    // if the 3D points is world coordinates, the computed transformation (rvec 
    //    and tvec) is from world coordinate system to camera coordinate system
    // if the 3D points is left camera coordinates of keyframe, the transformation
    //    is from keyframe's left camera to current left camera
    // here, assume world coordinates are used, then rvec and tvec describe left camera pose relative to world
    cv::solvePnPRansac(points3D, points2D, camMatrix, cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE); // since didn't undistort before
    // cv::solvePnPRansac(points3D, points2D, camMatrix, distCoeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    // cv::solvePnP(points3D, points2D, camMatrix, distCoeffs, rvec, tvec);

    if (_debug) {
        std::cout << "[RANSAC PnP] number of inliers: " << inliers.rows << std::endl;
    }

    cv::Mat cvR;
    EigenMatrix3Type R;
    EigenVector3Type t;
    cv::Rodrigues(rvec, cvR);
    cv::cv2eigen(cvR, R);
    cv::cv2eigen(tvec, t);
    pose = SophusSE3Type(R, t);
}

} // namespace cfsd