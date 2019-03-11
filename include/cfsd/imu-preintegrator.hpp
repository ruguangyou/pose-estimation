#ifndef IMU_PREINTEGRATOR_HPP
#define IMU_PREINTEGRATOR_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"

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

class ImuPreintegrator {
  public:
    ImuPreintegrator(const bool verbose);

    // Reinitialize after finishing processing IMU measurements between two consecutive camera frames.
    void reinitialize();

    // Take measurements and perform preintegration, jacobians calculation and noise propagation.
    void process();

    // Iteratively preintegrate IMU measurements.
    void iterate();

    // Calculate right jacobian of SO3.
    void rightJacobianSO3(const EigenVector3Type& omega);

    // Propagate preintegration noise.
    void propagate(const EigenMatrix3Type& temp);

    // Calculate jacobians of R, v, p with respect to bias.
    void jacobians(EigenMatrix3Type& temp);


    // Data from the sensor "ellipse2n" is in IMU coordinate system; convert to camera coordinate system if needed.
    void coordinateImu2Camera();

    void collectAccData(const long& timestamp, const float& accX, const float& accY, const float& accZ);

    void collectGyrData(const long& timestamp, const float& gyrX, const float& gyrY, const float& gyrZ);

  private:
    bool _verbose;

    // A very small number that helps determine if a rotation is close to zero.
    float _epsilon;

    // IMU parameters:
    
    // Sampling frequency.
    int _samplingRate;
    
    // Sampling time (1 / _samplingRate)
    float _deltaT;
    
    // _deltaT * _deltaT
    float _deltaT2;
    
    // Noise density of accelerometer and gyroscope measurements.
    //   Continuous-time model: sigma_g, unit: [rad/(s*sqrt(Hz))] or [rad/sqrt(s)]
    //                          sigma_a, unit: [m/(s^2*sqrt(Hz))] or [m/(s*sqrt(s))]
    //   Discrete-time model: sigma_gd = sigma_g/sqrt(delta_t), unit: [rad/s]
    //                        sigma_ad = sigma_a/sqrt(delta_t), unit: [m/s^2]
    float _accNoiseD, _gyrNoiseD;

    // Bias random walk noise density of accelerometer and gyroscope.
    //   Continuous-time model: sigma_bg, unit: [rad/(s^2*sqrt(Hz))] or [rad/(s*sqrt(s))]
    //                          sigma_ba, unit: [m/(s^3*sqrt(Hz))] or [m/(s^2*sqrt(s))]
    //   Discrete-time model: sigma_bgd = sigma_bg*sqrt(delta_t), unit: [rad/s]
    //                        sigma_bad = sigma_ba*sqrt(delta_t), unit: [m/s^2]
    // (Bias are modelled with a "Brownian motion" process, also termed a "Wiener process", or "random walk" in discrete-time)
    float _accBiasD, _gyrBiasD;
    
    // Gravity vector.
    EigenVector3Type _gravity;

    // Store data in a queue since acc and gry data are sent separately by od4; push if new data comes in, and pop it after preintegrating.
    // IMU raw data read from ellipse2n is float type.
    
    // Acceleration in IMU coordinate system.
    std::queue<EigenVector3Type> _accQueue;
    
    // Angular velocity in IMU coordinate system.
    std::queue<EigenVector3Type> _gryQueue;
    
    // Timestamps.
    std::queue<long> _timestampQueue;

    // Acceleration and angular velocity to be processed (from queue's front).
    EigenVector3Type _acc, _gyr;

    // Bias of accelerometer and gyroscope.
    EigenVector3Type _biasAcc, _biasGyr;

    // Preintegrated delta_R, delta_v, delta_p.
    SophusSO3Type _delta_R_next, _delta_R_prev;
    EigenVector3Type _delta_v_next, _delta_v_prev;
    EigenVector3Type _delta_p_next, _delta_p_prev;

    // Right jacobian of SO(3).
    EigenMatrix3Type _Jr;

    // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
    EigenMatrix3Type _d_R_bg_next, _d_R_bg_prev;
    EigenMatrix3Type _d_v_ba_next, _d_v_ba_prev;
    EigenMatrix3Type _d_v_bg_next, _d_v_bg_prev;
    EigenMatrix3Type _d_p_ba_next, _d_p_ba_prev;
    EigenMatrix3Type _d_p_bg_next, _d_p_bg_prev;

    // Covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p]
    Eigen::Matrix<precisionType,9,9> _covPreintegration;
    
    // Covatiance matrix of measurement discrete-time noise [n_gd, n_ad]
    Eigen::Matrix<precisionType,6,6> _covMeasurement;

    // EKF data of ellipse2n (Euler angles? Quaternions?)
    // standard deviations?

    // GPS data for validation?

};

} // namespace cfsd

#endif // IMU_PREINTEGRATOR_HPP