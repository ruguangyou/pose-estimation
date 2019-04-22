#ifndef MAP_HPP
#define MAP_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/structs.hpp"
#include "cfsd/frame.hpp"

// local sliding-window size
#define WINDOWSIZE 6

#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"
#endif

namespace cfsd {

class Map {
  public: 
    Map(const bool verbose);


    void pushImuConstraint(cfsd::Ptr<ImuConstraint>& ic);

    // void pushFrame(const std::map<size_t,Feature>& features, const std::vector<size_t>& matchedFeatureIDs);

    void checkKeyframe();

    void updateStates(double delta_pose[WINDOWSIZE][6], double delta_v_dbga[WINDOWSIZE][9]);

    void updateImuBias(Eigen::Vector3d& bg_i, Eigen::Vector3d& ba_i);

    // TODO................
    Sophus::SE3d getBodyPose();

    void setInitialRotation(const Eigen::Vector3d& delta_r);
  
  private:
    bool _verbose;

    // Minimum rotation and translation for selecting a keyframe.
    double _minRotation, _minTranslation;

    bool _notPushed{true};
    
  public:
    // Gravity vector.
    Eigen::Vector3d _gravity;

    // State (R, v, p) and bias (bg, ba) from time i to j
    // keyframe: *                         *
    //   camear: *       *        *        *
    //      imu: * * * * * * * * * * * * * *
    // Store keyframes' states and temperorily store current states.
    std::vector<Sophus::SO3d> _R;
    std::vector<Eigen::Vector3d> _p;
    std::vector<Eigen::Vector3d> _v;
    std::vector<Eigen::Vector3d> _dbg;
    std::vector<Eigen::Vector3d> _dba;

    std::vector<cfsd::Ptr<ImuConstraint>> _imuConstraint;

    std::vector<cfsd::Ptr<MapPoint>> _mapPoints;

    bool _isKeyframe{false};

    #ifdef USE_VIEWER
    cfsd::Ptr<Viewer> _pViewer;
    #endif
};

} // namespace cfsd

#endif // MAP_HPP