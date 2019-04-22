#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : _state(SYNCHRONIZING), _verbose(verbose) {
    
    _pCameraModel = std::make_shared<CameraModel>();

    _pMap = std::make_shared<Map>(verbose);
    
    _pImuPreintegrator = std::make_shared<ImuPreintegrator>(_pMap, verbose);

    _pFeatureTracker = std::make_shared<FeatureTracker>(_pMap, _pCameraModel, verbose);

    _pOptimizer = std::make_shared<Optimizer>(_pMap, _pFeatureTracker, _pImuPreintegrator, _pCameraModel, verbose);

}

bool VisualInertialSLAM::process(const cv::Mat& grayL, const cv::Mat& grayR, const long& imgTimestamp) {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    
    switch (_state) {
        case OK:
        {
            // Do imu preintegration.
            start = std::chrono::steady_clock::now();
            if(!_pImuPreintegrator->processImu(imgTimestamp)) {
                std::cerr << "Error occurs in imu-preintegration!" << std::endl;
                return false;
            }
            _pMap->pushImuConstraint(_pImuPreintegrator->_ic);
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
                _pOptimizer->motionOnlyBA(grayL);
                end = std::chrono::steady_clock::now();
                std::cout << "motion-only BA elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
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
            if (!_gyrInitialized) {
                std::cout << "Initializing gyroscope bias..." << std::endl;
                _pImuPreintegrator->processImu(imgTimestamp);
                _pOptimizer->initialGyrBias();
                _pImuPreintegrator->reset();
                std::cout << "Gyr bias initialization Done!" << std::endl << std::endl;
                _gyrInitialized = true;
            }
            else if (!_poseInitialized) {
                std::cout << "Initializing body pose..." << std::endl;
                _pImuPreintegrator->processImu(imgTimestamp);
                _pOptimizer->initialAlignment(true);
                _pImuPreintegrator->reset();
                std::cout << "Body pose initialization Done!" << std::endl << std::endl;
                _poseInitialized = true;
            }
            else if (!_accInitialized) {
                std::cout << "Initializing accelerometer bias..." << std::endl;
                _pImuPreintegrator->processImu(imgTimestamp);
                _pOptimizer->initialAlignment(false);
                _pImuPreintegrator->reset();
                std::cout << "Acc bias initialization Done!" << std::endl << std::endl;
                _accInitialized = true;
            }
            else {    
                // Initial stereo pair matching.
                std::cout << "Initializing stereo pair matching..." << std::endl;
                _pFeatureTracker->processImage(grayL, grayR);
                std::cout << "Initial matching Done!" << std::endl << std::endl;
                
                // Add the initial frame as keyframe.
                _pFeatureTracker->featurePoolUpdate();
                
                _state = OK;
            }
            break;
        }
        case SYNCHRONIZING:
        {
            if (_pImuPreintegrator->processImu(imgTimestamp))
                _state = INITIALIZING;
            break;
        }
        case LOST:
        {
            // if lost the track of features ...
            // need relocalization
            break;
        }
    }
    return true;
}

void VisualInertialSLAM::collectImuData(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z) {
    switch (st) {
        case ACCELEROMETER:
            _acc << (double)x, (double)y, (double)z;
            _accGot = true;
            std::cout << "rotated acc:\n" << _pMap->_R[0] * _acc + _pMap->_gravity << std::endl;
            break;
        case GYROSCOPE:
            _gyr << (double)x, (double)y, (double)z;
            _gyrGot = true;
    }
    if (_accGot && _gyrGot) {
        _pImuPreintegrator->pushImuData(timestamp, _gyr, _acc);
        _gyrGot = false;
        _accGot = false;
    }
}

} // namespace cfsd
