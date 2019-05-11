#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : 
    _state(SYNCHRONIZING), _verbose(verbose), _gyr(), _acc(), _pCameraModel(), _pMap(), _pLoopClosure(), _pFeatureTracker(), _pOptimizer(), _pImuPreintegrator() {
    
    _pCameraModel = std::make_shared<CameraModel>();

    _pMap = std::make_shared<Map>(_pCameraModel, verbose);

    _pImuPreintegrator = std::make_shared<ImuPreintegrator>(_pMap, verbose);
    
    _pLoopClosure = std::make_shared<LoopClosure>(verbose);

    _pFeatureTracker = std::make_shared<FeatureTracker>(_pMap, _pCameraModel, _pLoopClosure, verbose);

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

            cv::Mat descriptorsMat;
            // Do feature tracking.
            start = std::chrono::steady_clock::now();
            bool emptyMatch = _pFeatureTracker->processImage(grayL, grayR, descriptorsMat);
            end = std::chrono::steady_clock::now();
            std::cout << "Feature-tracking elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

            start = std::chrono::steady_clock::now();
            _pLoopClosure->detectLoop(descriptorsMat);
            end = std::chrono::steady_clock::now();
            std::cout << "Query database elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

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
            if (_pMap->_isKeyframe) {
                start = std::chrono::steady_clock::now();
                _pFeatureTracker->featurePoolUpdate(imgTimestamp);
                end = std::chrono::steady_clock::now();
                std::cout << "Feature pool update elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

                // Add left images (in the form of descriptors) to the bag-of-words database.
                start = std::chrono::steady_clock::now();
                _pLoopClosure->addImage(descriptorsMat);
                end = std::chrono::steady_clock::now();
                std::cout << "Add image to database elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
            }

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
            
            cv::Mat descriptorsMat;
            // Initial stereo pair matching.
            std::cout << "Initializing stereo pair matching..." << std::endl;
            _pFeatureTracker->processImage(grayL, grayR, descriptorsMat);
            std::cout << "Initial matching Done!" << std::endl << std::endl;

            // Add the initial frame as keyframe.
            _pFeatureTracker->featurePoolUpdate(imgTimestamp);

            _pLoopClosure->addImage(descriptorsMat);
            
            _state = OK;
            break;
        }
        case SFM:
        {
            if (_sfmCount < WINDOWSIZE-1) {
                // Relative transformation from current frame to preveious frame.
                Eigen::Vector3d r, p;

                if(!_pImuPreintegrator->processImu(imgTimestamp)) {
                    std::cerr << "Error occurs in imu-preintegration!" << std::endl;
                    return false;
                }
                
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

void VisualInertialSLAM::saveResults() {
    std::cout << "Saving results..." << std::endl;

    // Write estimated states to file.
    std::ofstream ofs("states.csv", std::ofstream::out);
    ofs << "timestamp,qw,qx,qy,qz,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n";
    Eigen::Quaterniond q;
    Eigen::Vector3d p, v, bg, ba;
    for (int i = 1; i < _pMap->_pKeyframes.size(); i++) {
        const cfsd::Ptr<Keyframe>& frame = _pMap->_pKeyframes[i];

        ofs << frame->timestamp << ",";

        q = frame->R.unit_quaternion();
        ofs << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ",";
        
        p = frame->p;
        ofs << p(0)  << ","  << p(1)  << ","  << p(2)  << ",";

        v = frame->v;
        ofs << v(0) << "," << v(1) << "," << v(2) << ",";

        bg = frame->pImuConstraint->bg_i + frame->dbg;
        ofs << bg(0) << "," << bg(1) << "," << bg(2) << ",";

        ba = frame->pImuConstraint->ba_i + frame->dba;
        ofs << ba(0) << "," << ba(1) << "," << ba(2) << "\n";
    }
    ofs.close();

    std::cout << "Saved" << std::endl << std::endl;
}

} // namespace cfsd
