#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : _state(OK), _verbose(verbose) {
    
    _pCameraModel = std::make_shared<CameraModel>();

    _pMap = std::make_shared<Map>(verbose);
    
    _pImuPreintegrator = std::make_shared<ImuPreintegrator>(_pMap, verbose);

    _pOptimizer = std::make_shared<Optimizer>(_pMap, _pImuPreintegrator, _pCameraModel, verbose);

    _pFeatureTracker = std::make_shared<FeatureTracker>(_pMap, _pOptimizer, _pImuPreintegrator, _pCameraModel, verbose);

}

void VisualInertialSLAM::processImage(const cv::Mat& grayL, const cv::Mat& grayR) {
    switch (_state) {
        case OK:
        { 
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

void VisualInertialSLAM::processImu(const long& timestamp,
                                    const double& gyrX, const double& gyrY, const double& gyrZ,
                                    const double& accX, const double& accY, const double& accZ) {
    /*  IMU coordinate system => camera coordinate system
              / x (roll)                  / z
             /                           /
            ------ y (pitch)            ------ x
            |                           |
            | z (yaw)                   | y
        
        last year's proxy-ellipse2n (this is what will be received if using replay data collected in 2018-12-05)
                 z |  / x
                   | /
           y _ _ _ |/
    */
    Eigen::Vector3d gyr, acc;
    gyr << gyrY, gyrZ, gyrX;
    acc << accY, accZ, accX;

    // auto start = std::chrono::steady_clock::now();
    _pImuPreintegrator->process(timestamp, gyr, acc);
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "One preintegration elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
}

void VisualInertialSLAM::collectImuData(const cfsd::SensorType& st, const long& timestamp, const float& x, const float& y, const float& z) {
    switch (st) {
        case ACCELEROMETER:
            _accX = x; _accY = y; _accZ = z;
            _accGot = true;
            break;
        case GYROSCOPE:
            _gyrX = x; _gyrY = y; _gyrZ = z;
            _gyrGot = true;
    }
    if (_accGot && _gyrGot) {
        processImu(timestamp, _gyrX, _gyrY, _gyrZ, _accX, _accY, _accZ);
        _gyrGot = false;
        _accGot = false;
    }
}

// void VisualInertialSLAM::optimize() {
//     if (_pImuPreintegrator->isLocalOptimizable()) {
//         // Local optimization.
//         auto start = std::chrono::steady_clock::now();
//         _pOptimizer->localOptimize();
//         auto end = std::chrono::steady_clock::now();
//         std::cout << "Local optimization elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
//     }
// }

void VisualInertialSLAM::setImgTimestamp(const long& timestamp) {
    _pImuPreintegrator->setImgTimestamp(timestamp);
}


} // namespace cfsd
