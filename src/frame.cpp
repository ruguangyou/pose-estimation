#include "cfsd/key-frame.hpp"

namespace cfsd {

KeyFrame::KeyFrame() {}

KeyFrame::~KeyFrame() {}

KeyFrame::KeyFrame(long id, long timestamp, CameraFrame::Ptr camFrame, bool verbose, bool debug)
    : _id(id), _timestamp(timestamp), _camFrame(camFrame), _verbose(verbose), _debug(debug) {
        _matchRatio = Config::get<float>("matchRatio");
        _minMatchDist = Config::get<float>("minMatchDist");
        int numberOfFeatures = Config::get<int>("numberOfFeatures");
        float scaleFactor = Config::get<float>("scaleFactor");
        int levelPyramid = Config::get<int>("levelPyramid");
        _orb = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid);
    }

KeyFrame::Ptr KeyFrame::create(long timestamp, CameraFrame::Ptr camFrame, bool verbose = false, bool debug = false) {
    static long frameCounter = 0;
    return KeyFrame::Ptr(new KeyFrame(frameCounter++, timestamp, camFrame, verbose, debug));
}

void KeyFrame::matchAndTriangulate() {
    // use ORB detection
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    _orb->detectAndCompute(_camFrame->getImgLeft(),  cv::noArray(), keypointsL, descriptorsL);
    _orb->detectAndCompute(_camFrame->getImgRight(), cv::noArray(), keypointsR, descriptorsR);
    
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsL, descriptorsR, matches);
    float min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2){ return m1.distance < m2.distance; })->distance;

    // drop bad matches whose distance is too large
    std::vector<cv::Point2d> goodPointsL, goodPointsR;
    for (cv::DMatch& m : matches) {
        if (m.distance < std::max(_matchRatio * min_dist, _minMatchDist)) {
            _camKeypoints.push_back(keypointsL[m.queryIdx]);
            if (_camDescriptors.rows == 0) {
                _camDescriptors = descriptorsL.row(m.queryIdx);
            }
            else {
                cv::vconcat(_camDescriptors, descriptorsL.row(m.queryIdx), _camDescriptors);
            }
            goodPointsL.push_back(keypointsL[m.queryIdx].pt);
            goodPointsR.push_back(keypointsR[m.trainIdx].pt);
        }
    }

    if (_debug) {
        std::cout << "[LEFT-RIGHT] number of total matches: " << matches.size() << std::endl;
        std::cout << "[LEFT-RIGHT] number of good matches: " << goodPointsL.size() << std::endl;
    }

    cv::Mat projL, projR;
    Eigen::Matrix<double,3,4> proj;
    proj = _camFrame->getCamLeft() * _SE3CamLeft.matrix3x4();
    cv::eigen2cv(proj, projL);
    proj = _camFrame->getCamRight() * _SE3CamRight.matrix3x4();
    cv::eigen2cv(proj, projR);

    cv::Mat homogeneous4D;  // the output of triangulatePoints() is 4xN array
    cv::triangulatePoints(projL, projR, goodPointsL, goodPointsR, homogeneous4D);

    for (int i = 0; i < homogeneous4D.cols; ++i) {
        float sign = (homogeneous4D.at<float>(2,i) / homogeneous4D.at<float>(3,i)) > 0 ? 1.0 : -1.0;
        _points3D.push_back(
            sign * cv::Point3d(homogeneous4D.at<double>(0,i) / homogeneous4D.at<double>(3,i),
                                homogeneous4D.at<double>(1,i) / homogeneous4D.at<double>(3,i),
                                homogeneous4D.at<double>(2,i) / homogeneous4D.at<double>(3,i)));
    }
}

void KeyFrame::setCamPose(Sophus::SE3d camPose) { 
    _SE3CamLeft = camPose;
    _SE3CamRight = _camFrame->getLeftToRight() * _SE3CamLeft;
}

} // namespace