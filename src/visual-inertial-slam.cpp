#include "cfsd/visual-inertial-slam.hpp"

namespace cfsd {

VisualInertialSLAM::VisualInertialSLAM(const bool verbose) : 
    _state(SYNCHRONIZING), _verbose(verbose), _gyr(), _acc(), _pCameraModel(), _pMap(), _pLoopClosure(), _pFeatureTracker(), _pOptimizer(), _pImuPreintegrator() {
    
    _pCameraModel = std::make_shared<CameraModel>();

    _pMap = std::make_shared<Map>(_pCameraModel, verbose);

    _pImuPreintegrator = std::make_shared<ImuPreintegrator>(_pMap, verbose);
    
    _pFeatureTracker = std::make_shared<FeatureTracker>(_pMap, _pCameraModel, verbose);

    _pOptimizer = std::make_shared<Optimizer>(_pMap, _pFeatureTracker, _pImuPreintegrator, _pCameraModel, verbose);

    _pLoopClosure = std::make_shared<LoopClosure>(_pMap, _pOptimizer, _pCameraModel, verbose);

    if (Config::get<bool>("loopClosure"))
        _loopThread = std::thread(&LoopClosure::run, _pLoopClosure);

    #ifdef USE_VIEWER
    // A thread for visulizing.
    _pViewer = std::make_shared<Viewer>();
    _pMap->_pViewer = _pViewer;
    _viewerThread = std::thread(&Viewer::run, _pViewer); // no need to detach, since there is a while loop in Viewer::run()
    #endif
}

bool VisualInertialSLAM::process(const cv::Mat& grayL, const cv::Mat& grayR, const long& imgTimestamp) {
    // Remap to undistorted and rectified image (detection mask needs to be updated)
    // cv::remap (about 3~5 ms) takes less time than cv::undistort (about 20~30 ms) method
    auto start = std::chrono::steady_clock::now();
    cv::Mat imgL, imgR;
    cv::remap(grayL, imgL, _pCameraModel->_rmap[0][0], _pCameraModel->_rmap[0][1], cv::INTER_LINEAR);
    cv::remap(grayR, imgR, _pCameraModel->_rmap[1][0], _pCameraModel->_rmap[1][1], cv::INTER_LINEAR);
    auto end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "remap elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    
    switch (_state) {
        case OK:
        {
            auto atStart = std::chrono::steady_clock::now();
            // Do imu preintegration.
            start = std::chrono::steady_clock::now();
            if(!_pImuPreintegrator->processImu(imgTimestamp)) {
                std::cerr << "Error occurs in imu-preintegration!" << std::endl;
                return false;
            }
            end = std::chrono::steady_clock::now();
            if(_verbose) std::cout << "Imu-preintegration elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
            _pMap->pushImuConstraint(_pImuPreintegrator->_ic);

            // Do feature tracking.
            cv::Mat descriptorsMat;
            start = std::chrono::steady_clock::now();
            bool emptyMatch = _pFeatureTracker->processImage(imgL, imgR, descriptorsMat);
            end = std::chrono::steady_clock::now();
            if(_verbose) std::cout << "Feature-tracking elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

            if (emptyMatch) {
                if(_verbose) std::cout << "Current image frame has no match with history frames!" << std::endl << std::endl;
                // if (++_noMatch > 3)
                //     _state = LOST;
                break;
            }

            // Perform motion-only BA.
            start = std::chrono::steady_clock::now();
            _pOptimizer->motionOnlyBA();
            end = std::chrono::steady_clock::now();
            if(_verbose) std::cout << "Motion-only BA elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
  
            // TODO................................
            if (_pMap->_needReinitialize) {
                if(_verbose) std::cout << "Bias corrupted, need reintialization." << std::endl << std::endl;
                // _state = INITIALIZING;
                // break;
            }

            // This step should be after the motion-only BA, s.t. we can know if current frame is keyframe and also the current camera pose.
            _pMap->checkKeyframe();

            auto atEnd = std::chrono::steady_clock::now();
            #ifdef SHOW_IMG
            showImage(imgL, std::chrono::duration<double, std::milli>(atEnd-atStart).count());
            #endif

            _pMap->manageMapPoints();

            if (_pMap->_isKeyframe) {
                _pImuPreintegrator->updateBias();
                _pImuPreintegrator->reset();

                start = std::chrono::steady_clock::now();
                _pFeatureTracker->featurePoolUpdate(imgTimestamp);
                end = std::chrono::steady_clock::now();
                if(_verbose) std::cout << "Feature pool update elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

                // Add left image (in the form of descriptors) to the bag-of-words database.
                start = std::chrono::steady_clock::now();
                _pLoopClosure->addImage(descriptorsMat);
                end = std::chrono::steady_clock::now();
                if(_verbose) std::cout << "Add image to database elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

                // Detect loop for the newly inserted keyframe.
                start = std::chrono::steady_clock::now();
                int curFrameID = _pMap->_pKeyframes.size()-1;
                int minLoopFrameID = _pLoopClosure->detectLoop(descriptorsMat, curFrameID);
                end = std::chrono::steady_clock::now();
                if(_verbose) std::cout << "Query database elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
                
                if (minLoopFrameID >= 0) {
                    // A possible loop candidate, set the flag for further processing.
                    _pLoopClosure->setToCloseLoop(minLoopFrameID, curFrameID);
                }
                else {
                    // No loop candidate, set _toCloseLoop to false via overloading function.
                    _pLoopClosure->setToCloseLoop();
                }
            }
            break;
        }
        case SYNCHRONIZING:
        {
            if (_pImuPreintegrator->processImu(imgTimestamp)) {
                if (Config::get<bool>("initializeWithSfm")) {
                    Eigen::Vector3d r, p;
                    _pFeatureTracker->structFromMotion(imgL, imgR, r, p, true);
                }
                _sfmCount++;
                _state = SFM;
            }
            break;
        }
        case SFM:
        {
            // Relative transformation from current frame to preveious frame.
            Eigen::Vector3d r, p;

            if(!_pImuPreintegrator->processImu(imgTimestamp)) {
                std::cerr << "Error occurs in imu-preintegration!" << std::endl;
                return false;
            }
            
            if (Config::get<bool>("initializeWithSfm")) {
                // If sfm return true, it means this frame has significant pose change; otherwise, ignore this frame.
                start = std::chrono::steady_clock::now();
                if (_pFeatureTracker->structFromMotion(imgL, imgR, r, p)) {
                    _pMap->pushSfm(r, p, _pImuPreintegrator->_ic);
                    _pImuPreintegrator->reset();
                    _sfmCount++;
                }
                end = std::chrono::steady_clock::now();
                if (_verbose) std::cout << "SfM elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
            }
            else {
                // Assume the vehicle doesn't move during initialization.
                r << 0.0, 0.0, 0.0;
                p << 0.0, 0.0, 0.0;
                _pMap->pushSfm(r, p, _pImuPreintegrator->_ic);
                _pImuPreintegrator->reset();
                _sfmCount++;
            }

            if (_sfmCount == INITWINDOWSIZE) {
                _sfmCount = 0;
                _state = INITIALIZING;
            }
            else 
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

            // if(!_pImuPreintegrator->processImu(imgTimestamp)) {
            //     std::cerr << "Error occurs in imu-preintegration!" << std::endl;
            //     return false;
            // }

            _pImuPreintegrator->reset();
            _pMap->reset(0);
            
            cv::Mat descriptorsMat;
            // Initial stereo pair matching.
            std::cout << "Initializing stereo pair matching..." << std::endl;
            _pFeatureTracker->processImage(imgL, imgR, descriptorsMat);
            std::cout << "Initial matching Done!" << std::endl << std::endl;

            // Add the initial frame as keyframe.
            _pFeatureTracker->featurePoolUpdate(imgTimestamp);

            _pLoopClosure->addImage(descriptorsMat);
            
            _state = OK;
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
    
    if (Config::get<bool>("globalOptimize")) {
        std::cout << std::endl << "Perfrom global optimization..." << std::endl;
        auto start = std::chrono::steady_clock::now();
        _pOptimizer->fullBA();
        auto end = std::chrono::steady_clock::now();
        std::cout << "Global optimization done! elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
        
        // Write estimated states to file.
        std::ofstream ofs1("fullBA.csv", std::ofstream::out);
        ofs1 << "timestamp,qw,qx,qy,qz,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n";
        for (int i = 1; i < _pMap->_pKeyframes.size(); i++) {
            const cfsd::Ptr<Keyframe>& frame = _pMap->_pKeyframes[i];

            ofs1 << frame->timestamp << ",";

            q = frame->R.unit_quaternion();
            ofs1 << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ",";
            
            p = frame->p;
            ofs1 << p(0)  << ","  << p(1)  << ","  << p(2)  << ",";

            v = frame->v;
            ofs1 << v(0) << "," << v(1) << "," << v(2) << ",";

            bg = frame->pImuConstraint->bg_i + frame->dbg;
            ofs1 << bg(0) << "," << bg(1) << "," << bg(2) << ",";

            ba = frame->pImuConstraint->ba_i + frame->dba;
            ofs1 << ba(0) << "," << ba(1) << "," << ba(2) << "\n";
        }
        ofs1.close();
    }

    std::cout << "Saved" << std::endl << std::endl;

    #ifdef USE_VIEWER
    if (_viewerThread.joinable())
        _viewerThread.join();
    #endif
}

#ifdef SHOW_IMG
void VisualInertialSLAM::showImage(cv::Mat& imgL, const double& dt) {
    // Show pixels and reprojected pixels after optimization.
    int yOffset = _pFeatureTracker->_cropOffset;
    const cfsd::Ptr<Keyframe>& latestFrame = _pMap->_pKeyframes.back();
    for (auto mapPointID : latestFrame->mapPointIDs) {
        cfsd::Ptr<MapPoint> mp = _pMap->_pMapPoints[mapPointID];
        Eigen::Vector3d pixel_homo = _pCameraModel->_P_L.block<3,3>(0,0) * (_pCameraModel->_T_CB * (latestFrame->R.inverse() * (mp->position - latestFrame->p)));
        cv::Point2d& pixel = mp->pixels[_pMap->_pKeyframes.size()-1];
        #ifdef CFSD
        cv::rectangle(imgL, cv::Point(pixel.x-4, pixel.y-4 + yOffset), cv::Point(pixel.x+4, pixel.y+4 + yOffset), cv::Scalar(0));
        cv::circle(imgL, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2) + yOffset), 3, cv::Scalar(255));
        #else
        cv::rectangle(imgL, cv::Point(pixel.x-4, pixel.y-4), cv::Point(pixel.x+4, pixel.y+4), cv::Scalar(0));
        cv::circle(imgL, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(255));
        #endif
    }
    cv::Mat out;
    cv::vconcat(imgL, cv::Mat::zeros(25, imgL.cols, CV_8U), out);
    std::stringstream text;
    text << " fps: " << std::round(1000.0f/dt) << ", #keyframes: " << _pMap->_pKeyframes.size()-1 << ", #map points: " << _pMap->_pMapPoints.size() << ", #points in current frame: " << latestFrame->mapPointIDs.size();
    cv::putText(out, text.str(), cv::Point(4, out.rows-8), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255), 1);
    cv::imshow("pixel (black square) vs. reprojected pixel (white circle)", out);
    cv::waitKey(Config::get<int>("delay"));
}
#endif

} // namespace cfsd
