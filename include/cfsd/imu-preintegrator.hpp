#ifndef IMU_PREINTEGRATOR_HPP
#define IMU_PREINTEGRATOR_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/structs.hpp"
#include "cfsd/map.hpp"

namespace cfsd {

// Exponential map: v (rotation vector) -> v_hat (skew symmetric matrix) -> exp(v_hat) (rotation matrix)
// Logarithmic map: R (rotation matrix) -> log(R) (skew symmetrix matrix) -> log(R)_vee (rotation vector)

/* default IMU coordinate system => convert to camera coordinate system
            / x (roll)                          / z (yaw)
           /                                   /
          ------ y (pitch)                    ------ x (roll)
          |                                   |
          | z (yaw)                           | y (pitch)
*/

// class ImuPreintegrator : public std::enable_shared_from_this<ImuPreintegrator>
// shared_from_this() returns a std::shared_ptr<T> that shares ownership of *this with all existing std::shared_ptr that refer to *this.

class ImuPreintegrator {
  public:
    ImuPreintegrator(const cfsd::Ptr<Map> pMap, const bool verbose);

    // Set image timestamp for sync.
    void setImgTimestamp(const long& imgTimestamp);

    // Update bias after motion-only optimization is done.
    void updateBias();

    // Take measurements and perform preintegration, jacobians calculation and noise propagation.
    void process(const long& timestamp, const Eigen::Vector3d& gyr_jm1, const Eigen::Vector3d& acc_jm1);

    // Iteratively preintegrate IMU measurements.
    void iterate(Preintegration& pre, const Sophus::SO3d& dR, const Eigen::Vector3d& ub_acc_jm1);

    // Calculate right jacobian and the inverse of SO3.
    Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& omega);
    Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& omega);

    // Propagate preintegration noise.
    void propagate(Preintegration& pre, const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, const Eigen::Matrix3d& temp);

    // Calculate jacobians of R, v, p with respect to bias.
    void jacobians(Preintegration& pre, const Eigen::Matrix3d& Jr, Eigen::Matrix3d& temp);

    bool evaluate(const ImuConstraint& ic,
        const Eigen::Vector3d& r_i, const Eigen::Vector3d& v_i, const Eigen::Vector3d& p_i,
        const Eigen::Vector3d& bg_i, const Eigen::Vector3d& ba_i,
        const Eigen::Vector3d& r_j, const Eigen::Vector3d& v_j, const Eigen::Vector3d& p_j,
        const Eigen::Vector3d& bg_j, const Eigen::Vector3d& ba_j,
        double* residuals, double** jacobians);

  private:
    bool _verbose;

    cfsd::Ptr<Map> _pMap;

    // A very small number that helps determine if a rotation is close to zero.
    double _epsilon{1e-5};

    // IMU parameters:
    // Sampling frequency.
    int _samplingRate{0};
    
    // Sampling time (1 / _samplingRate); _deltaT2 = _deltaT * _deltaT
    double _deltaT{0}, _deltaT2{0};
    
    // Gravity vector.
    Eigen::Vector3d _gravity;

    // Covariance matrix of measurement discrete-time noise [n_gd, n_ad]
    Eigen::Matrix<double,6,6> _covNoise;

    // Covariance matrix of discrete-time bias [b_gd, b_ad]
    Eigen::Matrix<double,6,6> _covBias;

    // There should be 
    /* keyframe: o                        o
          image: x           x            x              x
            imu: * * * * * * * * * * * * * * * * * * * * *
                 | <-------> | (imu constraints between two consecutive frames)
                 | <--------------------> | (imu constraints between two keyframes)
    */
    Preintegration _betweenFrames, _betweenKeyframes;

    // Mutex for image timestamp.
    std::mutex _timeMutex;

    // Image timestamp.
    long _imgTimestamp{0};

    // // Mutex for the bool indicator of local optimization.
    // std::mutex _optimizeMutex;
    // // Check if it's time for local optimization, 
    // // i.e. imu timestamp matched with image timestamp.
    // // (Assumption: the optimization elapsed time should not take too long)
    // bool _isLocalOptimizable{false};
};

} // namespace cfsd

#endif // IMU_PREINTEGRATOR_HPP