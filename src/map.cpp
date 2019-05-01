#include "cfsd/map.hpp"

namespace cfsd {

Map::Map(const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) : _pCameraModel(pCameraModel), _verbose(verbose), _gravity() {
    _R.push_back(Sophus::SO3d());
    _v.push_back(Eigen::Vector3d::Zero());
    _p.push_back(Eigen::Vector3d::Zero());
    _dbg.push_back(Eigen::Vector3d::Zero());
    _dba.push_back(Eigen::Vector3d::Zero());

    _frames.resize(1);

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
    
        kitti imu coordinate system
              z |  / x
                | /
          y -----

        euroc imu coordinate system
              x |  / z
                | /
                ------ y
    */
    // _gravity << 0, 0, g; // for cfsd
    // _gravity << 0, 0, -g; // for kitti
    _gravity << -g, 0, 0; // for euroc
}

void Map::pushSfm(const Eigen::Vector3d& r, const Eigen::Vector3d& p, const cfsd::Ptr<ImuConstraint>& ic) {
    // (r, p): current camera frame C2 to previeous camera frame C1, i.e. C1<-C2
    // T_BC: camera frame to body/imu frame, i.e. B1<-C1, B2<-C2 
    // (_R, _p): body frame to world frame, i.e. W<-B
    // W<-B2 = W<-B1 * B1<-B2 = W<-B1 * B1<-C1 * C1<-C2 * C2<-B2 = W<-B1 * B1<-C1 * C1<-C2 * (B2<-C2).inverse()
    // _R.back() and _p.back() represents W<-B1
    // now we want to push_back W<-B2
    Sophus::SE3d T_C1C2(Sophus::SO3d::exp(r), p);
    Sophus::SE3d T_WB1(_R.back(), _p.back());
    Sophus::SE3d T_WB2 = T_WB1 * _pCameraModel->_T_BC * T_C1C2 * _pCameraModel->_T_CB;
    _R.push_back(T_WB2.so3());
    _p.push_back(T_WB2.translation());
    // _R.push_back(_R.back() * Sophus::SO3d::exp(r));
    // _p.push_back(_R.back() * p + _p.back());
    _v.push_back(Eigen::Vector3d::Zero());

    _imuConstraint.push_back(ic);
}

void Map::repropagate(const int& start, const Eigen::Vector3d& delta_dbg, const Eigen::Vector3d& delta_dba) {
    for (int i = 0; i < WINDOWSIZE-1; i++) {
        cfsd::Ptr<ImuConstraint> ic = _imuConstraint[start+i];
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

    std::cout << "Gravity w.r.t the initial body frame:" << std::endl
              << "estimated direction (unit vector):\n" << _init_gravity << std::endl
              << "estimated magnititude:" << g.norm() << std::endl;
}

void Map::updateInitialVelocity(const int& start, double delta_v[WINDOWSIZE][3]) {
    for (int i = 0; i < WINDOWSIZE; i++)
        _v[start+i] = _v[start+i] + Eigen::Vector3d(delta_v[i][0], delta_v[i][1], delta_v[i][2]);
}

void Map::updateInitialRotation(const int& start, const Eigen::Vector3d& delta_r) {
    // dR is the rotation from initial body frame to world frame.
    Sophus::SO3d dR = Sophus::SO3d::exp(delta_r);
    std::cout << "Initial quaternion:\n" << dR.unit_quaternion().w() << std::endl << dR.unit_quaternion().vec() << std::endl;
    std::cout << "Rotate initial gravity (unit vector):\n" << dR * _init_gravity << std::endl;

    for (int i = 0; i < WINDOWSIZE; i++) {
        // (world <- body) = (world <- initial body) * (initial body <- body)
        _R[start+i] = dR * _R[start+i]; 
        _v[start+i] = dR.matrix() * _v[start+i];
        _p[start+i] = dR.matrix() * _p[start+i];
    }
}

void Map::reset(const int& start) {
    _R[start] = _R[start+WINDOWSIZE-1];
    _v[start] = _v[start+WINDOWSIZE-1];
    _p[start] = _p[start+WINDOWSIZE-1];
    _imuConstraint[start] = _imuConstraint[start+WINDOWSIZE-2];

    int n = _R.size()-WINDOWSIZE+1;
    _R.resize(n);
    _v.resize(n);
    _p.resize(n);
    _imuConstraint.resize(n-1);
}

void Map::pushImuConstraint(const cfsd::Ptr<ImuConstraint>& ic) {
    if (_notPushed || _isKeyframe) {
        // This is provided as initial value for ceres optimization.
        Sophus::SO3d R_j = _R.back() * ic->delta_R_ij;
        Eigen::Vector3d v_j = _v.back() + _gravity * ic->dt + _R.back() * ic->delta_v_ij;
        Eigen::Vector3d p_j = _p.back() + _v.back() * ic->dt + _gravity * ic->dt2 / 2 + _R.back() * ic->delta_p_ij;
        
        _R.push_back(R_j);
        _v.push_back(v_j);
        _p.push_back(p_j);
        _dbg.push_back(Eigen::Vector3d::Zero());
        _dba.push_back(Eigen::Vector3d::Zero());
        _imuConstraint.push_back(ic);
        _notPushed = false;
    }
    else {
        int idx = _R.size() - 2;
        _R.back() = _R[idx] * (ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * _dbg[idx]));
        _v.back() = _v[idx] + _gravity * ic->dt + _R[idx] * (ic->delta_v_ij + ic->d_v_bg_ij * _dbg[idx] + ic->d_v_ba_ij * _dba[idx]);
        _p.back() = _p[idx] + _v[idx] * ic->dt + _gravity * ic->dt2 / 2 + _R[idx] * (ic->delta_p_ij + ic->d_p_bg_ij * _dbg[idx] + ic->d_p_ba_ij * _dba[idx]);
        _dbg.back().setZero();
        _dba.back().setZero();
        _imuConstraint.back() = ic;
    }

    _sumImuTime += ic->dt;
}

void Map::checkKeyframe() {
    int n = _R.size();

    Sophus::SE3d T_WB_i = Sophus::SE3d(_R[n-2], _p[n-2]);
    Sophus::SE3d T_WB_j = Sophus::SE3d(_R[n-1], _p[n-1]);
    Sophus::SE3d T_ji = T_WB_j * T_WB_i.inverse();

    Eigen::Vector3d dr = T_ji.so3().log();
    Eigen::Vector3d dp = T_ji.translation();

    _isKeyframe = (dr.norm() > _minRotation || dp.norm() > _minTranslation || _sumImuTime > _maxImuTime);

    if (_isKeyframe) {
        _sumImuTime = 0;
        std::cout << "=> this frame IS a keyframe" << std::endl;
    }
    else
        std::cout << "=> this frame NOT a keyframe" << std::endl;
}

void Map::updateStates(double delta_pose[WINDOWSIZE][6], double delta_v_dbga[WINDOWSIZE][9]) {
    int n = _R.size() - WINDOWSIZE;
    if (n < 0) n = 0;
    int i;
    for (i = 0 ; n < _R.size(); i++, n++) {
        #ifdef USE_VIEWER
        _pViewer->pushRawPosition(_p[n], i);
        #endif

        _dba[n] = _dba[n] + Eigen::Vector3d(delta_v_dbga[i][6], delta_v_dbga[i][7], delta_v_dbga[i][8]);

        _dbg[n] = _dbg[n] + Eigen::Vector3d(delta_v_dbga[i][3], delta_v_dbga[i][4], delta_v_dbga[i][5]);
        
        _v[n] = _v[n] + Eigen::Vector3d(delta_v_dbga[i][0], delta_v_dbga[i][1], delta_v_dbga[i][2]);

        _p[n] = _p[n] + _R[n] * Eigen::Vector3d(delta_pose[i][3], delta_pose[i][4], delta_pose[i][5]);

        // Update landmark position.
        // Sophus::SO3d updated_R = _R[n] * Sophus::SO3d::exp(Eigen::Vector3d(delta_pose[i][0], delta_pose[i][1], delta_pose[i][2]));
        // for (int j = 0; j < _frames[n].size(); j++)
        //     _frames[n][j].second = updated_R * (_R[n].inverse() * _frames[n][j].second);
        //     // update in Viewer as well?
        // _R[n] = updated_R;

        _R[n] = _R[n] * Sophus::SO3d::exp(Eigen::Vector3d(delta_pose[i][0], delta_pose[i][1], delta_pose[i][2]));

        #ifdef USE_VIEWER
        _pViewer->pushOptimizedPosition(_p[n], i);
        #endif
    }

    Eigen::Vector3d updated_bg = _imuConstraint[n-2]->bg_i + _dbg[n-1];
    Eigen::Vector3d updated_ba = _imuConstraint[n-2]->ba_i + _dba[n-1];
    _needReinitialize = updated_bg.norm() > _maxGyrBias || updated_ba.norm() > _maxAccBias;

    std::cout << "estimated pose:\n" << Sophus::SE3d(_R[n-1], _p[n-1]).matrix3x4() << std::endl;
    std::cout << "estimated velocity:\n" << _v[n-1] << std::endl;
    std::cout << "estimated gyr bias:\n" << updated_bg << std::endl;
    std::cout << "estimated acc bias:\n" << updated_ba << std::endl;
}

void Map::updateImuBias(Eigen::Vector3d& bg_i, Eigen::Vector3d& ba_i) {
    if (_isKeyframe) {
        bg_i = _imuConstraint.back()->bg_i + _dbg.back();
        ba_i = _imuConstraint.back()->ba_i + _dba.back();
        std::cout << "updated gyr bias:\n" << bg_i << std::endl;
        std::cout << "updated acc bias:\n" << ba_i << std::endl;
    }
}

Sophus::SE3d Map::getBodyPose() {
    return Sophus::SE3d(_R.back(), _p.back());
}

} // namespace cfsd