#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : _state(SYNCHRONIZING), _verbose(verbose) {
    
    _pCameraModel = std::make_shared<CameraModel>();

    _pMap = std::make_shared<Map>(_pCameraModel, verbose);
    
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
            // ........... integration time between two keyframes should not be too large because of the drifting nature of imu.

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
                std::cout << "Motion-only BA elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
                if (_pMap->_needReinitialize) {
                    std::cout << "Bias corrupted, need reintialization." << std::endl << std::endl;
                    // _state = INITIALIZING;
                    // break;
                }
            }

            // This step should be after the motion-only BA, s.t. we can know if current frame is keyframe and also the current camera pose.
            start = std::chrono::steady_clock::now();
            _pFeatureTracker->featurePoolUpdate();
            end = std::chrono::steady_clock::now();
            std::cout << "Feature pool update elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

            break;
        }
        case INITIALIZING:
        {
            std::cout << "Initializing gyroscope bias..." << std::endl;
            start = std::chrono::steady_clock::now();
            _pOptimizer->initialGyrBias();
            end = std::chrono::steady_clock::now();
            std::cout << "elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            std::cout << "Gyr bias initialization Done!" << std::endl << std::endl;

            std::cout << "Initializing gravity and velocity..." << std::endl;
            start = std::chrono::steady_clock::now();
            _pOptimizer->initialGravityVelocity(); // assume zero acc bias
            end = std::chrono::steady_clock::now();
            std::cout << "elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            std::cout << "Gravity and velocity initialization Done!" << std::endl << std::endl;

            std::cout << "Refining gravity..." << std::endl;
            start = std::chrono::steady_clock::now();
            _pOptimizer->initialAlignment();
            end = std::chrono::steady_clock::now();
            std::cout << "elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            std::cout << "Gravity refinement Done!" << std::endl << std::endl;
            
            std::cout << "Initializing accelerometer bias..." << std::endl;
            start = std::chrono::steady_clock::now();
            _pOptimizer->initialAccBias();
            end = std::chrono::steady_clock::now();
            std::cout << "elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            std::cout << "Acc bias initialization Done!" << std::endl << std::endl;

            _pImuPreintegrator->reset();
            _pMap->reset(0);
            
            // Initial stereo pair matching.
            std::cout << "Initializing stereo pair matching..." << std::endl;
            _pFeatureTracker->processImage(grayL, grayR);
            std::cout << "Initial matching Done!" << std::endl << std::endl;
            
            // Add the initial frame as keyframe.
            _pFeatureTracker->featurePoolUpdate();
            
            _state = OK;
            break;
        }
        case SFM:
        {
            if (_sfmCount < WINDOWSIZE-1) {
                // Relative transformation from current frame to preveious frame.
                Eigen::Vector3d r, p;

                _pImuPreintegrator->processImu(imgTimestamp);
                
                // If sfm return true, it means this frame has significant pose change; otherwise, ignore this frame.
                start = std::chrono::steady_clock::now();
                if (_pFeatureTracker->structFromMotion(grayL, grayR, r, p)) {
                    _pMap->pushSfm(r, p, _pImuPreintegrator->_ic);
                    _pImuPreintegrator->reset();
                    _sfmCount++;
                }
                end = std::chrono::steady_clock::now();
                std::cout << "SfM elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
            }
            else {
                _sfmCount = 0;
                _state = INITIALIZING;
            }
            break;
        }
        case SYNCHRONIZING:
        {
            if (_pImuPreintegrator->processImu(imgTimestamp)) {
                Eigen::Vector3d r, p;
                _pFeatureTracker->structFromMotion(grayL, grayR, r, p, true);
                _state = SFM;
            }
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
