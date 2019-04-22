#include "cfsd/map.hpp"

namespace cfsd {

Map::Map(const bool verbose) : _verbose(verbose), _gravity() {
    _R.push_back(Sophus::SO3d());
    _v.push_back(Eigen::Vector3d::Zero());
    _p.push_back(Eigen::Vector3d::Zero());
    _dbg.push_back(Eigen::Vector3d::Zero());
    _dba.push_back(Eigen::Vector3d::Zero());

    _minRotation = Config::get<double>("keyframe_rotation");
    _minTranslation = Config::get<double>("keyframe_translation");

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

void Map::checkKeyframe() {
    int n = _R.size();

    Sophus::SE3d T_WB_i = Sophus::SE3d(_R[n-2], _p[n-2]);
    Sophus::SE3d T_WB_j = Sophus::SE3d(_R[n-1], _p[n-1]);
    Sophus::SE3d T_ji = T_WB_j * T_WB_i.inverse();

    Eigen::Vector3d dr = T_ji.so3().log();
    Eigen::Vector3d dp = T_ji.translation();

    _isKeyframe = (dr.norm() > _minRotation || dp.norm() > _minTranslation);

    if (_isKeyframe)
        std::cout << "=> this frame IS a keyframe" << std::endl;
    else
        std::cout << "=> this frame NOT a keyframe" << std::endl;
}

void Map::pushImuConstraint(cfsd::Ptr<ImuConstraint>& ic) {
    // Roughly estimate the state at time j without bias_i updated.
    // This is provided as initial value for ceres optimization.
    Sophus::SO3d R_j = _R.back() * ic->delta_R_ij;
    Eigen::Vector3d v_j = _v.back() + _gravity * ic->dt + _R.back() * ic->delta_v_ij;
    Eigen::Vector3d p_j = _p.back() + _v.back() * ic->dt + _gravity * ic->dt2 / 2 + _R.back() * ic->delta_p_ij;

    if (_notPushed || _isKeyframe) {
        _R.push_back(R_j);
        _v.push_back(v_j);
        _p.push_back(p_j);
        _dbg.push_back(Eigen::Vector3d::Zero());
        _dba.push_back(Eigen::Vector3d::Zero());
        _imuConstraint.push_back(ic);
        _notPushed = false;
    }
    else {
        _R.back() = R_j;
        _v.back() = v_j;
        _p.back() = p_j;
        _dbg.back() = Eigen::Vector3d::Zero();
        _dba.back() = Eigen::Vector3d::Zero();
        _imuConstraint.back() = ic;
    }
}

// void Map::pushFrame(const std::map<size_t,Feature>& features, const std::vector<size_t>& matchedFeatureIDs) {}

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

        _R[n] = _R[n] * Sophus::SO3d::exp(Eigen::Vector3d(delta_pose[i][0], delta_pose[i][1], delta_pose[i][2]));

        #ifdef USE_VIEWER
        _pViewer->pushOptimizedPosition(_p[n], i);
        #endif
    }
}

void Map::updateImuBias(Eigen::Vector3d& bg_i, Eigen::Vector3d& ba_i) {
    if (_isKeyframe) {
        bg_i += _dbg.back();
        ba_i += _dba.back();
    }
}

Sophus::SE3d Map::getBodyPose() {
    return Sophus::SE3d(_R.back(), _p.back());
}

void Map::setInitialRotation(const Eigen::Vector3d& delta_r) {
    _R[0] = _R[0] * Sophus::SO3d::exp(delta_r);
    std::cout << "Initial rotation:\n" << _R[0].matrix() << std::endl;
}

} // namespace cfsd