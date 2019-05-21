#include "cfsd/map.hpp"

namespace cfsd {

Map::Map(const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) : _pCameraModel(pCameraModel), _verbose(verbose) {
    
    _pKeyframes.push_back(std::make_shared<Keyframe>());

    _minRotation = Config::get<double>("keyframeRotation");
    _minTranslation = Config::get<double>("keyframeTranslation");
    
    _maxImuTime = Config::get<double>("maxImuTime");

    _maxGyrBias = Config::get<double>("maxGyrBias");
    _maxAccBias = Config::get<double>("maxAccBias");

    double g = Config::get<double>("gravity");
    /*  cfsd imu coordinate system
              / x
             /
            ------ y
            |
            | z

        euroc imu coordinate system
              x |  / z
                | /
                ------ y
    
        kitti imu coordinate system
              z |  / x
                | /
          y -----
    */
    #ifdef CFSD
    _gravity << 0, 0, g; // for cfsd
    #endif

    #ifdef EUROC
    _gravity << -g, 0, 0; // for euroc
    #endif
    
    #ifdef KITTI
    _gravity << 0, 0, -g; // for kitti
    #endif
}

void Map::pushSfm(const Eigen::Vector3d& r, const Eigen::Vector3d& p, const cfsd::Ptr<ImuConstraint>& ic) {
    // (r, p): current camera frame C2 to previeous camera frame C1, i.e. C1<-C2
    // T_BC: camera frame to body/imu frame, i.e. B1<-C1, B2<-C2 
    // (_R, _p): body frame to world frame, i.e. W<-B
    // W<-B2 = W<-B1 * B1<-B2 = W<-B1 * B1<-C1 * C1<-C2 * C2<-B2 = W<-B1 * B1<-C1 * C1<-C2 * (B2<-C2).inverse()
    // _R.back() and _p.back() represents W<-B1
    // now we want to push_back W<-B2
    Sophus::SE3d T_C1C2(Sophus::SO3d::exp(r), p);
    Sophus::SE3d T_WB1(_pKeyframes.back()->R, _pKeyframes.back()->p);
    Sophus::SE3d T_WB2 = T_WB1 * _pCameraModel->_T_BC * T_C1C2 * _pCameraModel->_T_CB;
    
    cfsd::Ptr<Keyframe> sfmFrame = std::make_shared<Keyframe>();
    sfmFrame->R = T_WB2.so3();
    sfmFrame->v = T_WB2.translation();
    sfmFrame->pImuConstraint = ic;

    _pKeyframes.push_back(sfmFrame);
}

void Map::repropagate(const int& start, const Eigen::Vector3d& delta_dbg, const Eigen::Vector3d& delta_dba) {
    for (int i = 1; i < INITWINDOWSIZE; i++) {
        cfsd::Ptr<ImuConstraint>& ic = _pKeyframes[start+i]->pImuConstraint;
        ic->bg_i = ic->bg_i + delta_dbg;
        ic->ba_i = ic->ba_i + delta_dba;
        ic->delta_R_ij = ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * delta_dbg);
        ic->delta_v_ij = ic->delta_v_ij + ic->d_v_bg_ij * delta_dbg + ic->d_v_ba_ij * delta_dba;
        ic->delta_p_ij = ic->delta_p_ij + ic->d_p_bg_ij * delta_dbg + ic->d_p_ba_ij * delta_dba;
    }
}

void Map::setInitialGravity(const Eigen::Vector3d& g) {
    // Gravity direction in the initial body frame.
    _init_gravity = g / g.norm();

    if (_verbose) {
        std::cout << "Gravity w.r.t the initial body frame:" << std::endl
                  << "estimated direction (unit vector):\n" << _init_gravity << std::endl
                  << "estimated magnititude: " << g.norm() << std::endl;
    }
}

void Map::updateInitialVelocity(const int& start, double delta_v[INITWINDOWSIZE][3]) {
    for (int i = 0; i < INITWINDOWSIZE; i++)
        _pKeyframes[start+i]->v = _pKeyframes[start+i]->v + Eigen::Vector3d(delta_v[i][0], delta_v[i][1], delta_v[i][2]);
        // _v[start+i] = _v[start+i] + Eigen::Vector3d(delta_v[i][0], delta_v[i][1], delta_v[i][2]);
}

void Map::updateInitialRotation(const int& start, const Eigen::Vector3d& delta_r) {
    // dR is the rotation from initial body frame to world frame.
    Sophus::SO3d dR = Sophus::SO3d::exp(delta_r);
    
    if (_verbose) {
        std::cout << "Initial quaternion:\n" << dR.unit_quaternion().w() << std::endl << dR.unit_quaternion().vec() << std::endl;
        std::cout << "Rotate initial gravity (unit vector):\n" << dR * _init_gravity << std::endl;
    }

    for (int i = 0; i < INITWINDOWSIZE; i++) {
        // (world <- body) = (world <- initial body) * (initial body <- body)
        cfsd::Ptr<Keyframe>& sfmFrame = _pKeyframes[start+i];
        sfmFrame->R = dR * sfmFrame->R;
        sfmFrame->v = dR.matrix() * sfmFrame->v;
        sfmFrame->p = dR.matrix() * sfmFrame->p;
    }
}

void Map::reset(const int& start) {
    // Keep the last sfm frame.
    _pKeyframes[start] = _pKeyframes[start+INITWINDOWSIZE-1]; // initial keyframe
    int n = _pKeyframes.size()-INITWINDOWSIZE+1;
    _pKeyframes.resize(n);
}

void Map::pushImuConstraint(const cfsd::Ptr<ImuConstraint>& ic) {
    if (_isKeyframe || _atBeginning) {
        // This is provided as initial value for ceres optimization.
        cfsd::Ptr<Keyframe>& latestKeyframe = _pKeyframes.back();
        Sophus::SO3d R_j = latestKeyframe->R * ic->delta_R_ij;
        Eigen::Vector3d v_j = latestKeyframe->v + _gravity * ic->dt + latestKeyframe->R * ic->delta_v_ij;
        Eigen::Vector3d p_j = latestKeyframe->p + latestKeyframe->v * ic->dt + _gravity * ic->dt2 / 2 + latestKeyframe->R * ic->delta_p_ij;
        
        cfsd::Ptr<Keyframe> newFrame = std::make_shared<Keyframe>();
        newFrame->R = R_j;
        newFrame->v = v_j;
        newFrame->p = p_j;
        newFrame->pImuConstraint = ic;
        _pKeyframes.push_back(newFrame);

        _atBeginning = false;
    }
    else {
        cfsd::Ptr<Keyframe>& latestFrame = _pKeyframes.back();
        cfsd::Ptr<Keyframe>& latestKeyframe = _pKeyframes[_pKeyframes.size()-2];

        latestFrame->R = latestKeyframe->R * (ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * latestKeyframe->dbg));
        latestFrame->v = latestKeyframe->v + _gravity * ic->dt + latestKeyframe->R * (ic->delta_v_ij + ic->d_v_bg_ij * latestKeyframe->dbg + ic->d_v_ba_ij * latestKeyframe->dba);
        latestFrame->p = latestKeyframe->p + latestKeyframe->v * ic->dt + _gravity * ic->dt2 / 2 + latestKeyframe->R * (ic->delta_p_ij + ic->d_p_bg_ij * latestKeyframe->dbg + ic->d_p_ba_ij * latestKeyframe->dba);
        latestFrame->dbg.setZero();
        latestFrame->dba.setZero();
        latestFrame->pImuConstraint = ic;
    }

    _sumImuTime += ic->dt;
}

void Map::checkKeyframe() {
    int n = _pKeyframes.size();

    Sophus::SE3d T_WB_i = Sophus::SE3d(_pKeyframes[n-2]->R, _pKeyframes[n-2]->p);
    Sophus::SE3d T_WB_j = Sophus::SE3d(_pKeyframes[n-1]->R, _pKeyframes[n-1]->p);
    Sophus::SE3d T_ji = T_WB_j * T_WB_i.inverse();

    Eigen::Vector3d dr = T_ji.so3().log();
    Eigen::Vector3d dp = T_ji.translation();

    // Integration time between two keyframes should not be too large because of the drifting nature of imu.
    _isKeyframe = (dr.norm() > _minRotation || dp.norm() > _minTranslation || _sumImuTime > _maxImuTime);

    if (_isKeyframe) {
        _sumImuTime = 0;
        if (_verbose) std::cout << "=> this frame IS a keyframe" << std::endl << std::endl;
    }
    else
        if (_verbose) std::cout << "=> this frame NOT a keyframe" << std::endl << std::endl;
}

void Map::manageMapPoints() {
    // If there are too many map points, erase those only seen by few frames.
    int minFrames = 0;
    if (_pMapPoints.size() > 10000) minFrames = 3;
    else if (_pMapPoints.size() > 8000) minFrames = 2;
    else if (_pMapPoints.size() > 4000) minFrames = 1;

    if (minFrames > 0) {
        auto iter = _pMapPoints.begin();
        // Keep the latest map points untouched.    
        for (int i = 0; i < _pMapPoints.size() - 1000; i++) {
            if (iter->second->pixels.size() <= minFrames)
                iter = _pMapPoints.erase(iter);
            else
                iter++;
        }
    }

    if (!_isKeyframe) {
        // The last frame is not a keyframe, so the relative recording in map points should be removed.
        int frameID = _pKeyframes.size()-1;
        for (auto mapPointID : _pKeyframes.back()->mapPointIDs) {
            // Some unnecessary map points might have been erased.
            if (_pMapPoints.find(mapPointID) == _pMapPoints.end())
                continue;
            _pMapPoints[mapPointID]->pixels.erase(frameID);
        }
        _pKeyframes.back()->mapPointIDs.clear();
        _pKeyframes.back()->descriptors = cv::Mat();
    }
}

void Map::updateStates(double delta_pose[WINDOWSIZE][6], double delta_v_dbga[WINDOWSIZE][9]) {
    int actualSize = (_pKeyframes.size() > WINDOWSIZE) ? WINDOWSIZE : _pKeyframes.size();
    int n = _pKeyframes.size() - actualSize;

    for (int i = 0 ; i < actualSize; i++) {
        cfsd::Ptr<Keyframe>& windowFrame = _pKeyframes[n+i];
        Sophus::SE3d T_WB1(windowFrame->R, windowFrame->p);

        #ifdef USE_VIEWER
        _pViewer->pushRawPosition(windowFrame->p, i);
        #endif

        windowFrame->dba = windowFrame->dba + Eigen::Vector3d(delta_v_dbga[i][6], delta_v_dbga[i][7], delta_v_dbga[i][8]);
        windowFrame->dbg = windowFrame->dbg + Eigen::Vector3d(delta_v_dbga[i][3], delta_v_dbga[i][4], delta_v_dbga[i][5]);
        windowFrame->v = windowFrame->v + Eigen::Vector3d(delta_v_dbga[i][0], delta_v_dbga[i][1], delta_v_dbga[i][2]);
        windowFrame->p = windowFrame->p + windowFrame->R * Eigen::Vector3d(delta_pose[i][3], delta_pose[i][4], delta_pose[i][5]);
        windowFrame->R = windowFrame->R * Sophus::SO3d::exp(Eigen::Vector3d(delta_pose[i][0], delta_pose[i][1], delta_pose[i][2]));

        // if (Eigen::Vector3d(delta_pose[i][3], delta_pose[i][4], delta_pose[i][5]).norm() > 0.1 || Eigen::Vector3d(delta_pose[i][0], delta_pose[i][1], delta_pose[i][2]).norm() > 0.1) {
        //     Sophus::SE3d T_WB2(windowFrame->R, windowFrame->p);
        //     // Update landmarks' 3D position.
        //     std::vector<Eigen::Vector3d>& points = _frameAndPoints[n+i];
        //     for (int j = 0; j < points.size(); j++)
        //         points[j] = T_WB2 * T_WB1.inverse() * points[j];
        //     #ifdef USE_VIEWER
        //     _pViewer->pushLandmark(points, i);
        //     #endif
        // }

        #ifdef USE_VIEWER
        _pViewer->pushPosition(windowFrame->p, i);
        #endif
    }

    #ifdef USE_VIEWER
    _pViewer->pushPose(_pKeyframes.back()->R.matrix());
    #endif

    Eigen::Vector3d updated_bg = _pKeyframes.back()->pImuConstraint->bg_i + _pKeyframes.back()->dbg;
    Eigen::Vector3d updated_ba = _pKeyframes.back()->pImuConstraint->ba_i + _pKeyframes.back()->dba;
    _needReinitialize = updated_bg.norm() > _maxGyrBias || updated_ba.norm() > _maxAccBias;

    if (_verbose) {
        std::cout << "estimated pose:\n" << Sophus::SE3d(_pKeyframes.back()->R, _pKeyframes.back()->p).matrix3x4() << std::endl;
        std::cout << "estimated velocity:\n" << _pKeyframes.back()->v << std::endl;
        std::cout << "estimated gyr bias:\n" << updated_bg << std::endl;
        std::cout << "estimated acc bias:\n" << updated_ba << std::endl;
    }
}

void Map::updateImuBias(Eigen::Vector3d& bg_i, Eigen::Vector3d& ba_i) {
    bg_i = _pKeyframes.back()->pImuConstraint->bg_i + _pKeyframes.back()->dbg;
    ba_i = _pKeyframes.back()->pImuConstraint->ba_i + _pKeyframes.back()->dba;
    if (_verbose) {
        std::cout << "updated gyr bias:\n" << bg_i << std::endl;
        std::cout << "updated acc bias:\n" << ba_i << std::endl;
    }
}

void Map::updateAllStates(double** delta_pose, double** delta_v_dbga) {
    for (int i = 0; i < _pKeyframes.size()-1; i++) {
        cfsd::Ptr<Keyframe>& keyframe = _pKeyframes[1+i];

        keyframe->dba = keyframe->dba + Eigen::Vector3d(delta_v_dbga[i][6], delta_v_dbga[i][7], delta_v_dbga[i][8]);

        keyframe->dbg = keyframe->dbg + Eigen::Vector3d(delta_v_dbga[i][3], delta_v_dbga[i][4], delta_v_dbga[i][5]);
        
        keyframe->v = keyframe->v + Eigen::Vector3d(delta_v_dbga[i][0], delta_v_dbga[i][1], delta_v_dbga[i][2]);

        keyframe->p = keyframe->p + keyframe->R * Eigen::Vector3d(delta_pose[i][3], delta_pose[i][4], delta_pose[i][5]);

        keyframe->R = keyframe->R * Sophus::SO3d::exp(Eigen::Vector3d(delta_pose[i][0], delta_pose[i][1], delta_pose[i][2]));

        #ifdef USE_VIEWER
        _pViewer->resetIdx();
        _pViewer->pushPosition(keyframe->p, i);
        #endif
    }

    // #ifdef USE_VIEWER
    // _pViewer->pushPose(_pKeyframes.back()->R.matrix());
    // #endif
}

void Map::pushLoopInfo(const int& curFrameID, const int& loopFrameID, const Eigen::Vector3d& rLoopToCur, const Eigen::Vector3d& pLoopToCur) {
    Sophus::SO3d R = Sophus::SO3d::exp(rLoopToCur);

    std::lock_guard<std::mutex> loopLock(_loopMutex);
    _pLoopInfos[curFrameID] = std::make_shared<LoopInfo>(loopFrameID, R, pLoopToCur);
}

bool Map::getLoopInfo(const int& curFrameID, int& loopFrameID, Sophus::SO3d& R, Eigen::Vector3d& p) {
    std::lock_guard<std::mutex> loopLock(_loopMutex);
    
    if (_pLoopInfos.find(curFrameID) == _pLoopInfos.end())
        return false;
    
    loopFrameID = _pLoopInfos[curFrameID]->loopFrameID;
    R = _pLoopInfos[curFrameID]->R;
    p = _pLoopInfos[curFrameID]->p;
    return true;
}

Sophus::SE3d Map::getBodyPose() {
    // Return the latest frame's pose, i.e. T_WB.
    return Sophus::SE3d(_pKeyframes.back()->R, _pKeyframes.back()->p);
}

} // namespace cfsd