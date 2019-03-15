#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : _state(OK), _verbose(verbose) {
    
    _pCameraModel = std::make_shared<CameraModel>();
    
    _pFeatureTracker = std::make_shared<FeatureTracker>(_pCameraModel, verbose);
    
    _pOptimizer = std::make_shared<Optimizer>(verbose);

    _pImuPreintegrator = std::make_shared<ImuPreintegrator>(_pOptimizer, verbose);
    
    // _map = Map::create(_verbose, _debug);
}

void VisualInertialSLAM::processImage(const long& timestamp, const cv::Mat& gray) {
    switch (_state) {
        case OK:
        {
            // Split image into left and right.
            cv::Mat grayL = gray(cv::Rect(0, 0, gray.cols/2, gray.rows));
            cv::Mat grayR = gray(cv::Rect(gray.cols/2, 0, gray.cols/2, gray.rows));
            
            #ifdef DEBUG_IMG
            cv::imshow("grayL", grayL);
            cv::imshow("grayR", grayR);
            cv::waitKey(0);
            #endif
            
            _pFeatureTracker->process(grayL, grayR);

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


void VisualInertialSLAM::processImu(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z) {
    if (st == ACCELEROMETER)
        _pImuPreintegrator->collectAccData(timestamp, x, y, z);
    else
        _pImuPreintegrator->collectGyrData(timestamp, x, y, z);
    
    _pImuPreintegrator->process();
}


} // namespace cfsd
