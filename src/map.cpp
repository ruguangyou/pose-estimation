#include "cfsd/map.hpp"

namespace cfsd {

Map::Map(const bool verbose) : _verbose(verbose) {
    _r.push_back(Eigen::Vector3d::Zero());
    _v.push_back(Eigen::Vector3d::Zero());
    _p.push_back(Eigen::Vector3d::Zero());
    _bg.push_back(Eigen::Vector3d::Zero());
    _ba.push_back(Eigen::Vector3d::Zero());

    _minRotation = Config::get<double>("keyframe_rotation");
    _minTranslation = Config::get<double>("keyframe_translation");
}

void Map::checkKeyframe() {
    int n = _r.size();

    Sophus::SE3d T_WB_i = Sophus::SE3d(Sophus::SO3d::exp(_r[n-2]), _p[n-2]);
    Sophus::SE3d T_WB_j = Sophus::SE3d(Sophus::SO3d::exp(_r[n-1]), _p[n-1]);
    Sophus::SE3d T_ji = T_WB_j * T_WB_i.inverse();

    Eigen::Vector3d dr = T_ji.so3().log();
    Eigen::Vector3d dp = T_ji.translation();

    _isKeyframe = (dr.norm() > _minRotation || dp.norm() > _minTranslation);

    if (_isKeyframe)
        std::cout << "=> this frame IS a keyframe" << std::endl;
    else
        std::cout << "=> this frame is NOT a keyframe" << std::endl;
}

void Map::pushImuConstraint(cfsd::Ptr<ImuConstraint>& ic, const Eigen::Vector3d& bg_j, const Eigen::Vector3d& ba_j, const Eigen::Vector3d& gravity) {
    // Roughly estimate the state at time j without bias_i updated.
    // This is provided as initial value for ceres optimization.
    Sophus::SO3d R_i = Sophus::SO3d::exp(_r.back());
    Eigen::Vector3d r_j = (R_i * ic->delta_R_ij).log();
    Eigen::Vector3d v_j = _v.back() + gravity * ic->dt + R_i * ic->delta_v_ij;
    Eigen::Vector3d p_j = _p.back() + _v.back() * ic->dt + gravity * ic->dt2 / 2 + R_i * ic->delta_p_ij;
    std::cout << "bg_j:\n" << bg_j << std::endl;
    std::cout << "ba_j:\n" << ba_j << std::endl;
    std::cout << "r_j:\n" << r_j << std::endl;
    std::cout << "v_j:\n" << v_j << std::endl;
    std::cout << "p_j:\n" << p_j << std::endl;

    if (_notPushed || _isKeyframe) {
        _r.push_back(r_j);
        _v.push_back(v_j);
        _p.push_back(p_j);
        _bg.push_back(bg_j);
        _ba.push_back(ba_j);
        _imuConstraint.push_back(ic);
        _notPushed = false;
    }
    else {
        _r.back() = r_j;
        _v.back() = v_j;
        _p.back() = p_j;
        _bg.back() = bg_j;
        _ba.back() = ba_j;
        _imuConstraint.back() = ic;
    }
}

// void Map::pushFrame(const std::map<size_t,Feature>& features, const std::vector<size_t>& matchedFeatureIDs) {}

int Map::getStates(double pose[WINDOWSIZE][6], double v_bga[WINDOWSIZE][9]) {
    int n = _r.size() - WINDOWSIZE;
    if (n < 0) n = 0;
    int i;
    for (i = 0 ; n < _r.size(); i++, n++) {
        pose[i][0] = _r[n](0);
        pose[i][1] = _r[n](1);
        pose[i][2] = _r[n](2);
        pose[i][3] = _p[n](0);
        pose[i][4] = _p[n](1);
        pose[i][5] = _p[n](2);

        v_bga[i][0] = _v[n](0);
        v_bga[i][1] = _v[n](1);
        v_bga[i][2] = _v[n](2);
        v_bga[i][3] = _bg[n](0);
        v_bga[i][4] = _bg[n](1);
        v_bga[i][5] = _bg[n](2);
        v_bga[i][6] = _ba[n](0);
        v_bga[i][7] = _ba[n](1);
        v_bga[i][8] = _ba[n](2);
    }

    #ifdef USE_VIEWER
    _pViewer->pushRawParameters(&pose[i-1][0]);
    #endif

    return i;
}

void Map::updateStates(double pose[WINDOWSIZE][6], double v_bga[WINDOWSIZE][9]) {
    int n = _r.size() - WINDOWSIZE;
    if (n < 0) n = 0;
    int i;
    for (i = 0 ; n < _r.size(); i++, n++) {
        _r[n](0) = pose[i][0];
        _r[n](1) = pose[i][1];
        _r[n](2) = pose[i][2];
        _p[n](0) = pose[i][3];
        _p[n](1) = pose[i][4];
        _p[n](2) = pose[i][5];

        _v[n](0) = v_bga[i][0];
        _v[n](1) = v_bga[i][1];
        _v[n](2) = v_bga[i][2];
        _bg[n](0) = v_bga[i][3];
        _bg[n](1) = v_bga[i][4];
        _bg[n](2) = v_bga[i][5];
        _ba[n](0) = v_bga[i][6];
        _ba[n](1) = v_bga[i][7];
        _ba[n](2) = v_bga[i][8];
    }

    #ifdef USE_VIEWER
    _pViewer->pushParameters(pose, i);
    #endif
}

void Map::updateImuBias(Eigen::Vector3d& bg_i, Eigen::Vector3d& ba_i) {
    if (_isKeyframe) {
        bg_i = _bg.back();
        ba_i = _ba.back();
    }
}

Sophus::SE3d Map::getBodyPose() {
    return Sophus::SE3d(Sophus::SO3d::exp(_r.back()), _p.back());
}

} // namespace cfsd