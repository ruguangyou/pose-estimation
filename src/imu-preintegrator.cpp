#include "cfsd/imu-preintegrator.hpp"

namespace cfsd {

ImuPreintegrator::ImuPreintegrator(const bool verbose) : _verbose(verbose), _epsilon(std::static_cast<float>(1e-5)),
        _biasAcc(EigenVector3Type::Zero()), _biasGyr(EigenVector3Type::Zero()), _delta_R_next(SophusSO3Type()), _delta_v_next(EigenVector3Type::Zero()),
        _delta_p_next(EigenVector3Type::Zero()), _d_R_bg_next(EigenMatrix3Type::Zero()), _d_v_ba_next(EigenMatrix3Type::Zero()),
        _d_v_bg_next(EigenMatrix3Type::Zero()), _d_p_ba_next(EigenMatrix3Type::Zero()), _d_p_bg_next(EigenMatrix3Type::Zero()), 
        _covPreintegration(Eigen::Matrix<precisionType,9,9>::Zero()), _covMeasurement(Eigen::Matrix<precisionType,6,6>::Zero()) {
    _samplingRate = Config::get<int>("samplingRate");
    _deltaT = 1.0 / std::static_cast<float>(_samplingRate);
    _deltaT2 = _deltaT * _deltaT;

    float g = Config::get<int>("gNorm");
    _gravity << 0, 0, g;

    float sqrtDeltaT = std::sqrt(_deltaT);
    // Convert unit [rad/sqrt(s)] to [rad/s]
    _gyrNoiseD = Config::get<float>("gyrNoise") / sqrtDeltaT;
    // Convert unit [g*sqrt(s)] to [m/s^2].
    _accNoiseD = Config::get<float>("accNoise") * g / sqrtDeltaT);
    // Unit [rad/s]
    _gyrBiasD = Config::get<float>("gyrBias");
    // Convert unit [g] to [m/s^2]
    _accBiasD = Config::get<float>("accBias") * g;
    
    // Initialize measurement covariance.
    _covMeasurement.block<3, 3>(0, 0) = (_gyrNoiseD * _gyrNoiseD) * EigenMatrix3Type::Identity();
    _covMeasurement.block<3, 3>(3, 3) = (_accNoiseD * _accNoiseD) * EigenMatrix3Type::Identity();
}

void ImuPreintegrator::reinitialize() {}

void ImuPreintegrator::process() {
    // Only receiving acc data or gyr data will not trigger further processing.
    if (_accQueue.size() == 0 || _gyrQueue.size() == 0) return;

    // Acc and gyr to be processed.
    _acc = _accQueue.front()
    _gyr = _gyrQueue.front()

    // 
    EigenVector3Type omega = (_gyr - _biasGyr) * _deltaT;
    SophusSO3Type dR = Sophus::SO3::exp(omega);

    // How many iterations?
    iterate(dR);

    // Update _Jr (right jacobian of SO3 rotation).
    rightJacobianSO3(omega);

    // Temp variable that will be used later.
    EigenVector3Type ab = _acc - _biasAcc;
    EigenMatrix3Type ab_hat;
    ab_hat <<   0.0, -ab(2),  ab(1),
              ab(2),    0.0, -ab(0),
             -ab(1),  ab(0),    0.0;
    EigenMatrix3Type temp = _delta_R_prev * ab_hat;

    // Noise propagation.
    propagate(dR, temp);

    // Jocabians of bias.
    jacobians(temp);

    // Pop out processed measurements.
    _accQueue.pop();
    _gyrQueue.pop();
}

void ImuPreintegrator::iterate(const SophusSO3Type& dR) {
    // Assign last iteration's "next" value to this iteration's "prev".
    _delta_R_prev = _delta_R_next;
    _delta_v_prev = _delta_v_next;
    _delta_p_prev = _delta_p_next;

    // Update this iteration's "next".
    _delta_R_next = _delta_R_prev * dR.matrix();
    _delta_v_next = _delta_v_prev + _delta_R_prev * (_acc - _biasAcc) * _deltaT;
    _delta_p_next = _delta_p_prev + _delta_v_prev * _deltaT + 0.5 * _delta_R_prev * (_acc - _biasAcc) * _deltaT2;
}

void ImuPreintegrator::rightJacobianSO3(const EigenVector3Type& omega) {
    // Jr(omega) = d_exp(omega_hat) / d_omega
    // Jr(omega) = I - [1-cos(|omega|)] / (|omega|^2) * omega_hat + [|omega| - sin(|omega|)] / (|omega|^3) * omega_hat^2
    
    // theta2 = |omega|^2
    // theta = |omega|
    const float theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const float theta = std::sqrt(theta2);
    
    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < _epsilon) {
        // If theta is small enough, we view it as 0.
        _Jr = EigenMatrix3Type::Identity();
    }
    else {
        EigenMatrix3Type omega_hat;
        unit_omega_hat <<       0.0, -omega(2),  omega(1),
                           omega(2),       0.0, -omega(0),
                          -omega(1),  omega(0),       0.0;
        _Jr = EigenMatrix3Type::Identity() - (1-std::cos(theta)) / theta2 * omega_hat + (theta - std::sin(theta)) / (theta2 * theta) * omega_hat * omega_hat;
    }
}

void ImuPreintegrator::propagate(const SophusSO3Type& dR, const EigenMatrix3Type& temp) {
    // Noise propagation: n_next = A * n_prev + B * n_measurement
    // Covriance propagation: cov_next = A * cov_prev * A' + B * cov_measurement * B'

    // Construct matrix A and B.
    Eigen::Matrix<precisionType,9,9> A;
    Eigen::Matrix<precisionType,9,6> B;
    A.setZero();
    B.setZero();
    A.block<3, 3>(0, 0) = dR.matrix();
    A.block<3, 3<(3, 0) = -temp * _deltaT;  // temp = _delta_R_prev * ab_hat
    A.block<3, 3<(3, 3) = EigenMatrix3Type::Identity();
    A.block<3, 3<(6, 0) = -0.5 * temp * _deltaT2;
    A.block<3, 3<(6, 3) = _deltaT * EigenMatrix3Type::Identity();
    A.block<3, 3<(6, 6) = EigenMatrix3Type::Identity();
    B.block<3, 3>(0, 0) = _Jr * _deltaT;
    B.block<3, 3>(3, 3) = _delta_R_prev * _deltaT;
    B.block<3, 3>(6, 3) = 0.5 * _delta_R_prev * _deltaT2;

    // Update covariance matrix.
    _covPreintegration = A * _covPreintegration * A.transpose() + B * _covMeasurement * B.transpose();
}

void ImuPreintegrator::jacobians(EigenMatrix3Type& temp) {
    // Jacobians of bias.
    _d_R_bg_prev = _d_R_bg_next;
    _d_v_ba_prev = _d_v_ba_next;
    _d_v_bg_prev = _d_v_bg_next;
    _d_p_ba_prev = _d_p_ba_next;
    _d_p_bg_prev = _d_p_bg_next;

    // (temp = _delta_R_prev * ab_hat) -> (temp = _delta_R_prev * ab_hat * _d_R_bg_prev)
    temp *= _d_R_bg_prev;

    // Update jacobians of R, v, p with respect to bias.
    _d_R_bg_next = -_d_R_bg_prev - _Jr * _deltaT;
    _d_v_ba_next = -_d_v_ba_prev - _delta_R_prev * _deltaT;
    _d_v_bg_next = -_d_v_bg_prev - temp * _deltaT;
    _d_p_ba_next = _d_p_ba_prev + _d_v_ba_prev * _deltaT - 0.5 * _delta_R_prev * _deltaT2;
    _d_p_bg_next = _d_p_bg_prev + _d_v_bg_prev * _deltaT - 0.5 * temp * _deltaT2;
}


void ImuPreintegrator::coordinateImu2Camera() {

}

void ImuPreintegrator::collectAccData(const long& timestamp, const float& accX, const float& accY, const float& accZ) {
    if (_accQueue.size() == _timestampQueue.size())
        _timestampQueue.push(timestamp);
    _accQueue.push(EigenVector3Type(accX, accY, accZ));
}

void ImuPreintegrator::collectGyrData(const long& timestamp, const float& gyrX, const float& gyrY, const float& gyrZ) {
    if (_gyrQueue.size() == _timestampQueue.size())
        _timestampQueue.push(timestamp);
    _gyrQueue.push(EigenVector3Type(gyrX, gyrY, gyrZ));
}

} // namespace cfsd