#ifndef MAP_HPP
#define MAP_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/structs.hpp"
#include "cfsd/camera-model.hpp"

// Local sliding-window size.
#define WINDOWSIZE 4
// Initialization window size.
#define INITWINDOWSIZE 4

#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"
#endif

namespace cfsd {

class Map {
  public: 
    Map(const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose);

    void pushSfm(const Eigen::Vector3d& r, const Eigen::Vector3d& p, const cfsd::Ptr<ImuConstraint>& ic);

    void repropagate(const int& start, const Eigen::Vector3d& delta_bg, const Eigen::Vector3d& delta_ba);

    void setInitialGravity(const Eigen::Vector3d& g);

    void updateInitialVelocity(const int& start, double delta_v[WINDOWSIZE][3]);

    void updateInitialRotation(const int& start, const Eigen::Vector3d& delta_r);

    void reset(const int& start);
    
    void pushImuConstraint(const cfsd::Ptr<ImuConstraint>& ic);

    void checkKeyframe();

    void manageMapPoints();

    void updateStates(double delta_pose[WINDOWSIZE][6], double delta_v_dbga[WINDOWSIZE][9]);

    void updateImuBias(Eigen::Vector3d& bg_i, Eigen::Vector3d& ba_i);

    void updateAllStates(double** delta_pose, double** delta_v_dbga);

    void pushLoopInfo(const int& curFrameID, const int& loopFrameID, const Eigen::Vector3d& rLoopToCur, const Eigen::Vector3d& pLoopToCur);

    bool getLoopInfo(const int& curFrameID, int& loopFrameID, Sophus::SO3d& R, Eigen::Vector3d& p);

    Sophus::SE3d getBodyPose();
  
  private:
    const bool _verbose;

    const cfsd::Ptr<CameraModel>& _pCameraModel;

    // Minimum rotation and translation for selecting a keyframe.
    double _minRotation{0};
    double _minTranslation{0};

    // Maximum integration time for imu, avoid accumulating too much drift.
    double _maxImuTime{0};
    double _sumImuTime{0};

    // Maximum allowable bias norm. Need reinitialization if exceeded.
    double _maxGyrBias{0};
    double _maxAccBias{0};

    bool _atBeginning{true};

    std::mutex _loopMutex{};
    
  public:
    // Gravity vector.
    Eigen::Vector3d _gravity{};

    Eigen::Vector3d _init_gravity{};

    // State (R, v, p) and bias (bg, ba) from time i to j
    // keyframe: *                         *
    //   camear: *       *        *        *
    //      imu: * * * * * * * * * * * * * *
    // Store keyframes' states and temperorily store current states.
    std::vector<cfsd::Ptr<Keyframe>> _pKeyframes{};

    // MapPoint ID : MapPoint pointer
    std::map<size_t, cfsd::Ptr<MapPoint>> _pMapPoints{};

    // Current frame ID : LoopInfo pointer
    std::unordered_map<int, cfsd::Ptr<LoopInfo>> _pLoopInfos{};


    bool _isKeyframe{false};

    bool _needReinitialize{false};

    #ifdef USE_VIEWER
    cfsd::Ptr<Viewer> _pViewer{};
    #endif
};

} // namespace cfsd

#endif // MAP_HPP