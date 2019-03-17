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
            
            auto start = std::chrono::steady_clock::now();
            _pFeatureTracker->process(grayL, grayR);
            auto end = std::chrono::steady_clock::now();
            std::cout << "Feature-tracking elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

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

void VisualInertialSLAM::collectImuData(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z) {
    if (st == ACCELEROMETER)
        _pImuPreintegrator->collectAccData(timestamp, x, y, z);
    else
        _pImuPreintegrator->collectGyrData(timestamp, x, y, z);
}

void VisualInertialSLAM::processImu() {
    while (!_pImuPreintegrator->isProcessable()) {
        // if (_verbose) std::cout << "Measurements not ready!" << std::endl;
        // Do nothing but wait until a specific number of measurements have been collected.
    }

    auto start = std::chrono::steady_clock::now();
    _pImuPreintegrator->process();
    auto end = std::chrono::steady_clock::now();
    std::cout << "Preintegration elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
}


} // namespace cfsd
