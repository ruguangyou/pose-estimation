#ifndef IMU_PREINTEGRATOR_HPP
#define IMU_PREINTEGRATOR_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/structs.hpp"
#include "cfsd/map.hpp"

namespace cfsd {

// Exponential map: v (rotation vector) -> v_hat (skew symmetric matrix) -> exp(v_hat) (rotation matrix)
// Logarithmic map: R (rotation matrix) -> log(R) (skew symmetrix matrix) -> log(R)_vee (rotation vector)

// Calculate right jacobian and the inverse of SO3.
Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& omega);
Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& omega);

// class ImuPreintegrator : public std::enable_shared_from_this<ImuPreintegrator>
// shared_from_this() returns a std::shared_ptr<T> that shares ownership of *this with all existing std::shared_ptr that refer to *this.

class ImuPreintegrator {
  public:
    ImuPreintegrator(const cfsd::Ptr<Map> pMap, const bool verbose);

    // Set initial IMU bias.
    void setInitialGyrBias(const Eigen::Vector3d& delta_bg);
    void setInitialAccBias(const Eigen::Vector3d& delta_ba);

    // Update bias after motion-only optimization is done.
    void updateBias();

    void pushImuData(const long& timestamp, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc);

    void reset();

    // Take measurements and perform preintegration, jacobians calculation and noise propagation.
    bool processImu(const long& imgTimestamp);

    // Iteratively preintegrate IMU measurements.
    void integrate( const Sophus::SO3d& dR, const Eigen::Vector3d& ub_acc_jm1);

    // Propagate preintegration noise.
    void propagateNoise(const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, const Eigen::Matrix3d& temp);

    // Calculate jacobians of R, v, p with respect to bias.
    void biasJacobians(const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, Eigen::Matrix3d& temp);

  private:
    const bool _verbose;

    cfsd::Ptr<Map> _pMap;

    // // A very small number that helps determine if a rotation is close to zero.
    // double _epsilon{1e-5};

    // IMU parameters:
    // Sampling frequency.
    int _samplingRate{0};
    
    // Sampling time (1 / _samplingRate); _deltaT2 = _deltaT * _deltaT
    double _deltaT{0}, _deltaT2{0};
    long _deltaTus{0}; // _deltaT in microseconds.

    // Covariance matrix of measurement discrete-time noise [n_gd, n_ad]
    Eigen::Matrix<double,6,6> _covNoiseD;

    // Covariance matrix of continuous-time bias [b_g, b_a]
    Eigen::Matrix<double,6,6> _covBias;

     /* keyframe: o                        o
          image: x           x            x              x
            imu: * * * * * * * * * * * * * * * * * * * * *
                 | <-------> | (imu constraints between two consecutive frames)
                 | <--------------------> | (imu constraints between two keyframes)
    */
    // Covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p, delta_bg, delta_ba]
    Eigen::Matrix<double,15,15> _covPreintegration_ij;

    // Bias of gyroscope and accelerometer at time i.
    Eigen::Vector3d _bg_i, _ba_i;
    
    // Preintegrated delta_R, delta_v, delta_p, iterate from (i,j-1) to (i,j)
    Sophus::SO3d _delta_R_ij, _delta_R_ijm1; // 'm1' means minus one
    Eigen::Vector3d _delta_v_ij, _delta_v_ijm1;
    Eigen::Vector3d _delta_p_ij, _delta_p_ijm1;

    // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
    Eigen::Matrix3d _d_R_bg_ij, _d_R_bg_ijm1;
    Eigen::Matrix3d _d_v_bg_ij, _d_v_bg_ijm1;
    Eigen::Matrix3d _d_v_ba_ij, _d_v_ba_ijm1;
    Eigen::Matrix3d _d_p_bg_ij, _d_p_bg_ijm1;
    Eigen::Matrix3d _d_p_ba_ij, _d_p_ba_ijm1;

   // The time between two camera frames.
    double _dt{0};

    /* If consider the situation that starts from the very begining:
        image:          x                x                 x      
          imu: * * * * * * * * * * * * * * * * * * * * * * * * * *
              | <=====> | (this part should be dropped)
                        | ==> (should start from here)
    */
    bool _isInitialized{false};

    // Since reading image has some delay, the imu timestamp is ahead of image timestamp, so store the imu data in the queue.
    std::mutex _dataMutex;
    std::queue<std::pair<Eigen::Vector3d,Eigen::Vector3d>> _dataQueue;
    std::queue<long> _timestampQueue;

  public:
    cfsd::Ptr<ImuConstraint> _ic;

};

} // namespace cfsd

#endif // IMU_PREINTEGRATOR_HPP