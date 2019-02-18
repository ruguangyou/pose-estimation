#include "cfsd/key-frame.hpp"

namespace cfsd {

KeyFrame::KeyFrame() {}
KeyFrame::~KeyFrame() {}

KeyFrame::Ptr KeyFrame::create() {}

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
    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Point2f> goodPointsL, goodPointsR;
    for (cv::DMatch& m : matches) {
        if (m.distance < std::max(min_dist*2, 30.0f)) {  // temperarily use 2 and 30.0f
            goodMatches.push_back(m);
            goodPointsL.push_back(keypointsL[m.queryIdx].pt);
            goodPointsR.push_back(keypointsR[m.trainIdx].pt);
        }
    }

    if (_verbose) {
        std::cout << "[LEFT-RIGHT] number of total matches: " << matches.size() << std::endl;
        std::cout << "[LEFT-RIGHT] number of good matches: " << goodMatches.size() << std::endl;
    }

    cv::Mat projL, projR;
    cv::eigen2cv()

}

} // namespace