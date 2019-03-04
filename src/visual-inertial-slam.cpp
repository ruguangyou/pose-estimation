#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : _state(OK), _verbose(verbose) {
    _pCameraModel = std::make_shared<CameraModel>();
    _pFeatureTracker = std::make_shared<FeatureTracker>(verbose);
    // _map = Map::create(_verbose, _debug);
}

void VisualInertialSLAM::process(const long& timestamp, const cv::Mat& gray) {
    switch (_state) {
        case OK:
        {
            cv::Mat grayL = gray(cv::Rect(0, 0, gray.cols/2, gray.rows));
            cv::Mat grayR = gray(cv::Rect(gray.cols/2, 0, gray.cols/2, gray.rows));
            
            // #ifdef DEBUG
            // cv::imshow("grayL", grayL);
            // cv::imshow("grayR", grayR);
            // cv::waitKey(0);
            // #endif
            
            _pFeatureTracker->processFrame(grayL, grayR);

            break;
        }
        case LOST:
        {
            // if lost the track of features ...
            // need relocalization
            break;
        }
    }
}

} // namespace cfsd
