#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : _state(OK), _verbose(verbose) {
    
    _pCameraModel = std::make_shared<CameraModel>();

    _pMap = std::make_shared<Map>(verbose);
    
    _pImuPreintegrator = std::make_shared<ImuPreintegrator>(_pMap, verbose);

    _pFeatureTracker = std::make_shared<FeatureTracker>(_pMap, _pCameraModel, verbose);

    _pOptimizer = std::make_shared<Optimizer>(_pMap, _pFeatureTracker, _pImuPreintegrator, _pCameraModel, verbose);

}

void VisualInertialSLAM::process(const cv::Mat& grayL, const cv::Mat& grayR, const long& imgTimestamp) {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    
    switch (_state) {
        case OK:
        {
            // Do imu preintegration.
            start = std::chrono::steady_clock::now();
            _pImuPreintegrator->processImu(imgTimestamp);
            end = std::chrono::steady_clock::now();
            std::cout << "Imu-preintegration elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

            // Do feature tracking.
            start = std::chrono::steady_clock::now();
            bool emptyMatch = _pFeatureTracker->processImage(grayL, grayR);
            end = std::chrono::steady_clock::now();
            std::cout << "Feature-tracking elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

            //TODO................
            // what if the processing time exceeds 100ms?

            // Perform motion-only BA.
            if (!emptyMatch) {
                start = std::chrono::steady_clock::now();
                _pOptimizer->motionOnlyBA();
                end = std::chrono::steady_clock::now();
                std::cout << "motion-only BA elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            }

            // This step should be after the motion-only BA, s.t. we can know if current frame is keyframe and also the current camera pose.
            start = std::chrono::steady_clock::now();
            _pFeatureTracker->featurePoolUpdate();
            end = std::chrono::steady_clock::now();
            std::cout << "feature pool update elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

            break;
        }
        case INITIALIZING:
        {
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
    switch (st) {
        case ACCELEROMETER:
            _acc << (double)x, (double)y, (double)z;
            _accGot = true;
            break;
        case GYROSCOPE:
            _gyr << (double)x, (double)y, (double)z;
            if (_gyr.norm() > 1000) std::cout << x << ", " << y << ", " << z << std::endl;
            _gyrGot = true;
    }
    if (_accGot && _gyrGot) {
        _pImuPreintegrator->pushImuData(timestamp, _gyr, _acc);
        _gyrGot = false;
        _accGot = false;
    }
}

} // namespace cfsd
