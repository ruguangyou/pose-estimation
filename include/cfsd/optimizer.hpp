#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "cfsd/cost-functions.hpp"
#include "cfsd/config.hpp"

namespace cfsd {

class Optimizer {
  public:
    Optimizer(const cfsd::Ptr<Map>& _pMap, const cfsd::Ptr<FeatureTracker>& pFeatureTracker, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose);

    // ~Optimizer();

    /* Only optimize motion (i.e. vehicle states), keep landmarks fixed.
       map points: (x x   x  x  x x)  <- fixed
                   /| |\ /|\ | /| |\
           frames: #  #  #  #  #  #  #  $    (# is keyframe, $ is latest frame)
                   | <=fixed=> |  | <=> | <- local-window to be optimizer
    */
    void motionOnlyBA();

    // Estimate initial IMU bias, align initial IMU acceleration to gravity.
    void initialGravityVelocity();
    
    void initialAlignment();

    void initialGyrBias();

    void initialAccBias();

    void loopCorrection(const int& curFrameID);

    void fullBA();

    bool linearizeReprojection(const size_t& mapPointID, const int& startFrameID, std::vector<double*>& delta_pose_img, int& errorTerms, Eigen::VectorXd& error, Eigen::MatrixXd F);

  private:
    const bool _verbose;

    cfsd::Ptr<Map> _pMap;

    cfsd::Ptr<FeatureTracker> _pFeatureTracker;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    const cfsd::Ptr<CameraModel>& _pCameraModel;

    double _pose[WINDOWSIZE][6];  // pose (rotation vector, translation vector / position)
    double _v_bga[WINDOWSIZE][9]; // velocity, bias of gyroscope, bias of accelerometer

    double _fx{0}, _fy{0}, _cx{0}, _cy{0};
    Eigen::Matrix2d _invStdT;
    
    double _priorWeight{0.0};

    bool _minimizerProgressToStdout{true};
    int _maxNumIterations{0};
    double _maxSolverTimeInSeconds{0.0};
    int _numThreads{0};
    bool _checkGradients{false};

    // std::mutex _loopMutex{};
};

} // namespace cfsd

#endif // OPTIMIZER_HPP