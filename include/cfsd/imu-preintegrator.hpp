#ifndef IMU_PREINTEGRATOR_HPP
#define IMU_PREINTEGRATOR_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/optimizer.hpp"

namespace cfsd {

class Optimizer;

// Exponential map: v (rotation vector) -> v_hat (skew symmetric matrix) -> exp(v_hat) (rotation matrix)
// Logarithmic map: R (rotation matrix) -> log(R) (skew symmetrix matrix) -> log(R)_vee (rotation vector)

/* default IMU coordinate system => convert to camera coordinate system
            / x (roll)                          / z (yaw)
           /                                   /
          ------ y (pitch)                    ------ x (roll)
          |                                   |
          | z (yaw)                           | y (pitch)

   last year's proxy-ellipse2n (this is what will be received if using replay data collected in 2018-12-05)
           z |  / x
             | /
     y _ _ _ |/
*/


class ImuPreintegrator : public std::enable_shared_from_this<ImuPreintegrator> {
  public:
    ImuPreintegrator(const cfsd::Ptr<Optimizer>& pOptimizer, const bool verbose);

    // Test if the number of collected measurements reaches '_iters'.
    bool isProcessable();

    // Set image timestamp for sync.
    void setImgTimestamp(const long& imgTimestamp);
    
    // Reinitialize after finishing processing IMU measurements between two consecutive camera frames.
    void reinitialize();

    // Take measurements and perform preintegration, jacobians calculation and noise propagation.
    void process();

    // Iteratively preintegrate IMU measurements.
    void iterate(const SophusSO3Type& dR);

    // Calculate right jacobian and the inverse of SO3.
    EigenMatrix3Type rightJacobianSO3(const EigenVector3Type& omega);
    EigenMatrix3Type rightJacobianInverseSO3(const EigenVector3Type& omega);

    // Propagate preintegration noise.
    void propagate(const SophusSO3Type& dR, const EigenMatrix3Type& Jr, const EigenMatrix3Type& temp);

    // Calculate jacobians of R, v, p with respect to bias.
    void jacobians(const EigenMatrix3Type& Jr, EigenMatrix3Type& temp);

    // Roughly estimate the state at time j from time i without considering bias update.
    void recover();

    // Feed roughly estimated states to ceres parameters as initial values for optimization.
    void getParameters(double* qvp_i, double* qvp_j);

    // Compute residuals and jocabians, this function will be called by ceres Evaluate function.
    bool evaluate(const EigenVector3Type& r_i, const EigenVector3Type& v_i, const EigenVector3Type& p_i,
                  const EigenVector3Type& r_j, const EigenVector3Type& v_j, const EigenVector3Type& p_j,
                  const EigenVector3Type& delta_bg, const EigenVector3Type& delta_ba,
                  double* residuals, double** jacobians);
    
    // Update states from ceres optimization results.
    void updateState(double* rvp_j, double* bg_ba);

    // Store data in queue.
    void collectAccData(const long& timestamp, const float& accX, const float& accY, const float& accZ);
    void collectGyrData(const long& timestamp, const float& gyrX, const float& gyrY, const float& gyrZ);

  private:
    bool _verbose;

    // Interface to optimizer.
    cfsd::Ptr<Optimizer> _pOptimizer;

    // A very small number that helps determine if a rotation is close to zero.
    precisionType _epsilon;

    // IMU parameters:
    
    // Sampling frequency.
    int _samplingRate;
    
    // Sampling time (1 / _samplingRate)
    precisionType _deltaT;
    
    // _deltaT * _deltaT
    precisionType _deltaT2;
    
    // Noise density of accelerometer and gyroscope measurements.
    //   Continuous-time model: sigma_g, unit: [rad/(s*sqrt(Hz))] or [rad/sqrt(s)]
    //                          sigma_a, unit: [m/(s^2*sqrt(Hz))] or [m/(s*sqrt(s))]
    //   Discrete-time model: sigma_gd = sigma_g/sqrt(delta_t), unit: [rad/s]
    //                        sigma_ad = sigma_a/sqrt(delta_t), unit: [m/s^2]
    precisionType _accNoiseD, _gyrNoiseD;

    // Bias random walk noise density of accelerometer and gyroscope.
    //   Continuous-time model: sigma_bg, unit: [rad/(s^2*sqrt(Hz))] or [rad/(s*sqrt(s))]
    //                          sigma_ba, unit: [m/(s^3*sqrt(Hz))] or [m/(s^2*sqrt(s))]
    //   Discrete-time model: sigma_bgd = sigma_bg*sqrt(delta_t), unit: [rad/s]
    //                        sigma_bad = sigma_ba*sqrt(delta_t), unit: [m/s^2]
    // (Bias are modelled with a "Brownian motion" process, also termed a "Wiener process", or "random walk" in discrete-time)
    precisionType _accBiasD, _gyrBiasD;
    
    // Gravity vector.
    EigenVector3Type _gravity;

    // Store data in a queue since acc and gry data are sent separately by od4; push if new data comes in, and pop it after preintegrating.
    // IMU raw data read from ellipse2n is float type.
    
    // // Angular velocity in IMU coordinate system.
    // std::queue<EigenVector3Type> _gyrQueue;

    // // Acceleration in IMU coordinate system.
    // std::queue<EigenVector3Type> _accQueue;
    
    // // IMU data timestamps.
    // std::queue<long> _timestampQueue;

    // Mutex for image timestamp.
    std::mutex _timeMutex;

    // Image timestamp.
    long _imgTimestamp;

    // Mutex that protects queue, since od4 runs as an independent thread and will send messages continuously.
    std::mutex _gyrMutex, _accMutex;

    // Acceleration and angular velocity to be processed (from queue's front).
    EigenVector3Type _acc, _gyr;

    // Timestamp of imu measurement to be processed (from queue's front)
    long _timestamp;

    bool _isNewTs;
    bool _iterStart;

    // State (R, v, p) and bias (bg, ba) from time i to j
    //   camear: *                         *
    //      imu: * * * * * * * * * * * * * *
    //           i ----------------------> j
    SophusSO3Type _R_i, _R_j;
    EigenVector3Type _v_i, _v_j;
    EigenVector3Type _p_i, _p_j;
    EigenVector3Type _biasGyr; // Bias of gyroscope.
    EigenVector3Type _biasAcc; // Bias of accelerometer.

    // The number of imu frames between two consecutive camera frames.
    // e.g. if camera 60 Hz, imu 900 Hz, then _iter = 15
    int _iters;

    // Time between two consecutive camera frames, defined as: _deltaT * _iters
    double _dt, _dt2;

    // Preintegrated delta_R, delta_v, delta_p, iterate from (i,j-1) to (i,j)
    SophusSO3Type _delta_R_ij, _delta_R_ijm1; // 'm1' means minus one
    EigenVector3Type _delta_v_ij, _delta_v_ijm1;
    EigenVector3Type _delta_p_ij, _delta_p_ijm1;

    // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
    EigenMatrix3Type _d_R_bg_ij, _d_R_bg_ijm1;
    EigenMatrix3Type _d_v_ba_ij, _d_v_ba_ijm1;
    EigenMatrix3Type _d_v_bg_ij, _d_v_bg_ijm1;
    EigenMatrix3Type _d_p_ba_ij, _d_p_ba_ijm1;
    EigenMatrix3Type _d_p_bg_ij, _d_p_bg_ijm1;

    // Covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p, delta_bg, delta_ba]
    Eigen::Matrix<precisionType,15,15> _covPreintegration;
    
    // Covariance matrix of measurement discrete-time noise [n_gd, n_ad]
    Eigen::Matrix<precisionType,6,6> _covMeasurement;

    // Covariance matrix of discrete-time bias [b_gd, b_ad]
    Eigen::Matrix<precisionType,6,6> _covBias;

    // EKF data of ellipse2n (Euler angles? Quaternions?)
    // standard deviations?

    // GPS data for validation?

};

} // namespace cfsd

#endif // IMU_PREINTEGRATOR_HPP