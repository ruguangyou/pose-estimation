#include "cfsd/key-frame.hpp"

namespace cfsd {

KeyFrame::KeyFrame() {}

KeyFrame::~KeyFrame() {}

KeyFrame::KeyFrame(long id, double timestamp, CameraFrame::Ptr camFrame)
    : _id(id), _timestamp(timestamp), _camFrame(camFrame) {}

KeyFrame::Ptr KeyFrame::create(double timestamp, CameraFrame::Ptr camFrame) {
    static long factory_id = 0;
    return KeyFrame::Ptr(new KeyFrame(factory_id++, timestamp, camFrame));
}

void KeyFrame::matchAndTriangulate() {
    // use ORB detection
    cv::Ptr<cv::ORB> orb = cv::ORB::create(); // use default parameters temperarily
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorR;
    orb->detectAndCompute(_camFrame.getImgLeft(),  cv::noArray(), keypointsL, descriptorsL);
    orb->detectAndCompute(_camFrame.getImgRight(), cv::noArray(), keypointsR, descriptorsR);
    
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsL, descriptorR, matches);
    float min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2){ return m1.distance < m2.distance; })->distance;

    // drop bad matches whose distance is too large
    std::vector<cv::Point2f> goodPointsL, goodPointsR;
    for (cv::DMatch& m : matches) {
        if (m.distance < std::max(min_dist*2, 30.0f)) {  // temperarily use 2 and 30.0f
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

    if (_verbose) {
        std::cout << "[LEFT-RIGHT] number of total matches: " << matches.size() << std::endl;
        std::cout << "[LEFT-RIGHT] number of good matches: " << goodPointsL.size() << std::endl;
    }

    cv::Mat projL, projR;
    cv::eigen2cv(_camFrame->getCamLeft() * _SE3CamLeft.matrix3x4(), projL);
    cv::eigen2cv(_camFrame->getCamRight() * (_camFrame->getLeftToRight() * _SE3CamLeft).matrix3x4(), projR);

    cv::Mat homogeneous4D;  // the output of triangulatePoints() is 4xN array
    cv::triangulatePoints(projL, projR, goodPointsL, goodPointsR, homogeneous4D);

    if (_debug) {
        std::cout << "homogeneous4D shape and channels: " 
                  << homogeneous4D.shape() << ", " 
                  << homogeneous4D.channels() << std::endl;
    }

    for (int i = 0; i < homogeneous4D.rows; ++i) {
        _points3D.push_back(
            cv::Point3f(homogeneous4D.at<float>(0,i) / homogeneous4D.at<float>(3,i),
                        homogeneous4D.at<float>(1,i) / homogeneous4D.at<float>(3,i),
                        homogeneous4D.at<float>(2,i) / homogeneous4D.at<float>(3,i)));
    }
}

void KeyFrame::setCamPose(Sophus::SE3d camPose) {
    _SE3CamLeft = camPose;
}

} // namespace