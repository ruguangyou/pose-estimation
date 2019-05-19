#include "cfsd/imu-preintegrator.hpp"

namespace cfsd {

// Utility: compute right jacobian of SO3.
Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& omega) {
    // Jr(omega) = d_exp(omega_hat) / d_omega
    // Jr(omega) = I - [1-cos(|omega|)] / (|omega|^2) * omega_hat + [|omega| - sin(|omega|)] / (|omega|^3) * omega_hat^2
    Eigen::Matrix3d Jr;

    // theta2 = |omega|^2
    // theta = |omega|
    const double theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const double theta = std::sqrt(theta2);
    
    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < 1e-5) {
        // If theta is small enough, we view it as 0.
        Jr = Eigen::Matrix3d::Identity();
    }
    else {
        Eigen::Matrix3d omega_hat = Sophus::SO3d::hat(omega);
        // omega_hat <<      0.0, -omega(2),  omega(1),
        //              omega(2),       0.0, -omega(0),
        //             -omega(1),  omega(0),       0.0;
        Jr = Eigen::Matrix3d::Identity() - (1-std::cos(theta)) / theta2 * omega_hat + (theta - std::sin(theta)) / (theta2 * theta) * omega_hat * omega_hat;
    }

    if (Jr.maxCoeff() > 1e8 || Jr.minCoeff() < -1e8)
        std::cout << "Numerical unstable in calculating right jacobian of SO3!" << std::endl;

    return Jr;
}

// Utility: computer inverse of right jacobian of SO3.
Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& omega) {
    // inv(Jr(omega)) = I + 0.5 * omega + [1 / |omega|^2 - (1+cos(|omega|) / (2*|omega|*sin(|omega)))] * omega_hat^2
    Eigen::Matrix3d JrInv;

    // theta2 = |omega|^2
    // theta = |omega|
    const double theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const double theta = std::sqrt(theta2);

    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < 1e-5) {
        // If theta is small enough, we view it as 0.
        JrInv = Eigen::Matrix3d::Identity();
    }
    else {
        Eigen::Matrix3d omega_hat = Sophus::SO3d::hat(omega);
        // omega_hat <<      0.0, -omega(2),  omega(1),
        //              omega(2),       0.0, -omega(0),
        //             -omega(1),  omega(0),       0.0;
        JrInv = Eigen::Matrix3d::Identity() + omega_hat / 2 + (1 / theta2 - (1 + std::cos(theta)) / (2 * theta * std::sin(theta))) * omega_hat * omega_hat;
    }

    if (JrInv.maxCoeff() > 1e8 || JrInv.minCoeff() < -1e8)
        std::cout << "Numerical unstable in calculating right jacobian inverse of SO3!" << std::endl;

    return JrInv;
}



ImuPreintegrator::ImuPreintegrator(const cfsd::Ptr<Map> pMap, const bool verbose) :
        _pMap(pMap), _verbose(verbose),
        _covNoiseD(Eigen::Matrix<double,6,6>::Zero()), _covBias(Eigen::Matrix<double,6,6>::Zero()),
        _covPreintegration_ij(Eigen::Matrix<double,15,15>::Zero()),
        _bg_i(Eigen::Vector3d::Zero()), _ba_i(Eigen::Vector3d::Zero()),
        _delta_R_ij(Sophus::SO3d()), _delta_R_ijm1(Sophus::SO3d()),
        _delta_v_ij(Eigen::Vector3d::Zero()), _delta_v_ijm1(Eigen::Vector3d::Zero()),
        _delta_p_ij(Eigen::Vector3d::Zero()), _delta_p_ijm1(Eigen::Vector3d::Zero()),
        _d_R_bg_ij(Eigen::Matrix3d::Zero()), _d_R_bg_ijm1(Eigen::Matrix3d::Zero()),
        _d_v_bg_ij(Eigen::Matrix3d::Zero()), _d_v_bg_ijm1(Eigen::Matrix3d::Zero()),
        _d_v_ba_ij(Eigen::Matrix3d::Zero()), _d_v_ba_ijm1(Eigen::Matrix3d::Zero()),
        _d_p_bg_ij(Eigen::Matrix3d::Zero()), _d_p_bg_ijm1(Eigen::Matrix3d::Zero()),
        _d_p_ba_ij(Eigen::Matrix3d::Zero()), _d_p_ba_ijm1(Eigen::Matrix3d::Zero()),
        _dataMutex(), _dataQueue(), _timestampQueue(), _ic() {
    
    _samplingRate = Config::get<int>("samplingRate");
    _deltaT = 1.0 / (double)_samplingRate;
    _deltaT2 = _deltaT * _deltaT;
    double sqrtDeltaT = std::sqrt(_deltaT);

    _deltaTus = (long)1000000 / _samplingRate;
    
    double g = Config::get<double>("gravity");

    double gyrNoiseD, accNoiseD, gyrBias, accBias;
    #ifdef CFSD
        // Noise density of accelerometer and gyroscope measurements.
        //   Continuous-time model: sigma_g, unit: [rad/(s*sqrt(Hz))] or [rad/sqrt(s)]
        //                          sigma_a, unit: [m/(s^2*sqrt(Hz))] or [m/(s*sqrt(s))]
        //   Discrete-time model: sigma_gd = sigma_g/sqrt(delta_t), unit: [rad/s]
        //                        sigma_ad = sigma_a/sqrt(delta_t), unit: [m/s^2]
        // Convert unit [rad/sqrt(s)] to [rad/s]
        gyrNoiseD = Config::get<double>("gyrNoise") / sqrtDeltaT;
        // Convert unit [g*sqrt(s)] to [m/s^2].
        accNoiseD = Config::get<double>("accNoise") * g / sqrtDeltaT;
        // Bias random walk noise density of accelerometer and gyroscope.
        //   Continuous-time model: sigma_bg, unit: [rad/(s^2*sqrt(Hz))] or [rad/(s*sqrt(s))]
        //                          sigma_ba, unit: [m/(s^3*sqrt(Hz))] or [m/(s^2*sqrt(s))]
        //   Discrete-time model: sigma_bgd = sigma_bg*sqrt(delta_t), unit: [rad/s]
        //                        sigma_bad = sigma_ba*sqrt(delta_t), unit: [m/s^2]
        // (Bias are modelled with a "Brownian motion" process, also termed a "Wiener process", or "random walk" in discrete-time)
        // Convert unit [rad/s] to [rad/(s*sqrt(s))]
        gyrBias = Config::get<double>("gyrBias") / sqrtDeltaT;
        // Convert unit [g] to [m/(s^2*sqrt(s))]
        accBias = Config::get<double>("accBias") * g / sqrtDeltaT;
    #endif

    #ifdef KITTI
        gyrNoiseD = Config::get<double>("gyrNoise") / sqrtDeltaT;
        accNoiseD = Config::get<double>("accNoise") * g / sqrtDeltaT;
        gyrBias = Config::get<double>("gyrBias") / sqrtDeltaT;
        accBias = Config::get<double>("accBias") * g / sqrtDeltaT;
    #endif

    #ifdef EUROC
        gyrNoiseD = Config::get<double>("gyroscope_noise_density") / sqrtDeltaT; // unit: [rad/s]
        accNoiseD = Config::get<double>("accelerometer_noise_density") / sqrtDeltaT; // unit: [m/s^2]
        gyrBias = Config::get<double>("gyroscope_random_walk"); // unit: [rad/(s*sqrt(s))]
        accBias = Config::get<double>("accelerometer_random_walk"); // unit: [m/(s^2*sqrt(s))]
    #endif
    
    // For noise, cov(nd) = cov(n) / deltaT. (deltaT is the time interval between two consecutive imu measurements)
    // Covariance matrix of discrete-time noise [n_gd, n_ad]
    _covNoiseD.block<3, 3>(0, 0) = (gyrNoiseD * gyrNoiseD) * Eigen::Matrix3d::Identity(); // unit: [rad^2/s^2]
    _covNoiseD.block<3, 3>(3, 3) = (accNoiseD * accNoiseD) * Eigen::Matrix3d::Identity(); // unit: [m^2/s^4]

    // For bias, cov(bd) = cov(b) * dt_ij. (dt_ij is the time interval between two keyframes)
    // Covariance matrix of discrete-time bias [b_gd, b_ad]
    _covBias.block<3, 3>(0, 0) = (gyrBias * gyrBias) * Eigen::Matrix3d::Identity(); // unit: [rad^2/(s^3)]
    _covBias.block<3, 3>(3, 3) = (accBias * accBias) * Eigen::Matrix3d::Identity(); // unit: [m^2/(s^5)]
}

void ImuPreintegrator::pushImuData(const long& timestamp, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc) {
    std::lock_guard<std::mutex> dataLock(_dataMutex);
    _dataQueue.push(std::make_pair(gyr, acc));
    _timestampQueue.push(timestamp);
}

void ImuPreintegrator::reset() {
    _delta_R_ij = Sophus::SO3d();
    _delta_v_ij.setZero();
    _delta_p_ij.setZero();
    _d_R_bg_ij.setZero();
    _d_v_ba_ij.setZero();
    _d_v_bg_ij.setZero();
    _d_p_ba_ij.setZero();
    _d_p_bg_ij.setZero();
    _covPreintegration_ij.block<9,9>(0,0).setZero();
    _dt = 0;
}

void ImuPreintegrator::setInitialGyrBias(const Eigen::Vector3d& delta_bg) {
    _bg_i = _bg_i + delta_bg;
    std::cout << "Initial gyr bias:\n" << _bg_i << std::endl;
}

void ImuPreintegrator::setInitialAccBias(const Eigen::Vector3d& delta_ba) {
    _ba_i = _ba_i + delta_ba;
    std::cout << "Initial acc bias:\n" << _ba_i << std::endl;
}

void ImuPreintegrator::updateBias() {
    _pMap->updateImuBias(_bg_i, _ba_i);
}

bool ImuPreintegrator::processImu(const long& imgTimestamp) {
    std::lock_guard<std::mutex> dataLock(_dataMutex);
    if (!_isInitialized) {
        // For kitti.
        if (imgTimestamp < _timestampQueue.front()) {
            std::cout << "not synchronized: image timestamp is ahead of imu timestamp, wait..." << std::endl;
            return false;
        }

        // Remove imu data that is collected before initialization.
        while (std::abs(imgTimestamp - _timestampQueue.front()) > _deltaTus/2) {
            if (!_timestampQueue.size()) {
                std::cout << "not synchronized: image timestamp is ahead of imu timestamp, wait..." << std::endl;
                return false;
            }
            _timestampQueue.pop();
            _dataQueue.pop();  
        }
        std::cout << "Imu preintegrator initialized!" << std::endl << std::endl;
        _isInitialized = true;
        return _isInitialized;
    }
    
    int count = 0;
    while (std::abs(imgTimestamp - _timestampQueue.front()) > _deltaTus/2) {
        // Queue might be empty, and error occurs then!
        if (!_timestampQueue.size()) {
            std::cerr << "Error: image timestamp is ahead of imu timestamp!" << std::endl;
            return false;
        }
        Eigen::Vector3d gyr_jm1, acc_jm1;
        gyr_jm1 = _dataQueue.front().first;
        // Rotate acc measurements to align with ...(earth frame?)
        acc_jm1 = _dataQueue.front().second;
        _dataQueue.pop();
        _timestampQueue.pop();
        count++;

        // Intermediate variables that will be used later.
        Eigen::Vector3d ub_gyr_jm1 = gyr_jm1 - _bg_i; // unbiased gyr at time j-1
        Eigen::Vector3d ub_acc_jm1 = acc_jm1 - _ba_i; // unbiased acc at time j-1
        Eigen::Vector3d omega = ub_gyr_jm1 * _deltaT;
        Sophus::SO3d dR = Sophus::SO3d::exp(omega);

        // Iteratively integrate.
        integrate(dR, ub_acc_jm1);
        
        // Compute Jr (right jacobian of SO3 rotation).
        Eigen::Matrix3d Jr = rightJacobianSO3(omega);
        
        // Intermediate variable that will be used later.
        Eigen::Matrix3d temp = _delta_R_ijm1.matrix() * Sophus::SO3d::hat(ub_acc_jm1);
        
        // Noise propagation.
        propagateNoise(dR, Jr, temp);
        
        // Jacobians of bias.
        biasJacobians(dR, Jr, temp);
        
        // Update number of frames.
        _dt += _deltaT;
    }

    if (_verbose) std::cout << "number of imu measurements preintegrated: " << count << std::endl;

    // For bias, cov(bd) = cov(b) * dt_ij. (dt_ij is the time interval between two keyframes)
    _covPreintegration_ij.block<6, 6>(9, 9) = _covBias * _dt;

    _ic = std::make_shared<ImuConstraint>(_covPreintegration_ij.inverse(), _bg_i, _ba_i, _delta_R_ij, _delta_v_ij, _delta_p_ij, _d_R_bg_ij, _d_v_bg_ij, _d_v_ba_ij, _d_p_bg_ij, _d_p_ba_ij, _dt);
    
    return true;
}

void ImuPreintegrator::integrate(const Sophus::SO3d& dR, const Eigen::Vector3d& ub_acc_jm1) {
    // Assign last iteration's "ij" value to this iteration's "ijm1".
    _delta_R_ijm1 = _delta_R_ij;
    _delta_v_ijm1 = _delta_v_ij;
    _delta_p_ijm1 = _delta_p_ij;

    // Update this iteration's "ij".
    _delta_R_ij = _delta_R_ijm1 * dR;
    _delta_v_ij = _delta_v_ijm1 + _delta_R_ijm1 * ub_acc_jm1 * _deltaT;
    _delta_p_ij = _delta_p_ijm1 + _delta_v_ijm1 * _deltaT + _delta_R_ijm1 * ub_acc_jm1 * _deltaT2 / 2;
}

void ImuPreintegrator::propagateNoise(const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, const Eigen::Matrix3d& temp) {
    // Noise propagation: n_ij = A * n_ijm1 + B * n_measurement
    // Covriance propagation: cov_ij = A * cov_ijm1 * A' + B * cov_measurement * B'

    // Construct matrix A and B.
    Eigen::Matrix<double,9,9> A;
    Eigen::Matrix<double,9,6> B;
    A.setZero();
    B.setZero();
    A.block<3, 3>(0, 0) = dR.matrix().transpose();
    A.block<3, 3>(3, 0) = -temp * _deltaT;  // temp = _delta_R_ijm1.matrix() * ab_hat
    A.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    A.block<3, 3>(6, 0) = -temp * _deltaT2 / 2;
    A.block<3, 3>(6, 3) = _deltaT * Eigen::Matrix3d::Identity();
    A.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    B.block<3, 3>(0, 0) = Jr * _deltaT;
    B.block<3, 3>(3, 3) = _delta_R_ijm1.matrix() * _deltaT;
    B.block<3, 3>(6, 3) = _delta_R_ijm1.matrix() * _deltaT2 / 2;

    // Update covariance matrix.
    _covPreintegration_ij.block<9, 9>(0, 0) = A * _covPreintegration_ij.block<9, 9>(0, 0) * A.transpose() + B * _covNoiseD * B.transpose();
}

void ImuPreintegrator::biasJacobians(const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, Eigen::Matrix3d& temp) {
    // Jacobians of bias.
    _d_R_bg_ijm1 = _d_R_bg_ij;
    _d_v_ba_ijm1 = _d_v_ba_ij;
    _d_v_bg_ijm1 = _d_v_bg_ij;
    _d_p_ba_ijm1 = _d_p_ba_ij;
    _d_p_bg_ijm1 = _d_p_bg_ij;

    // (temp = _delta_R_ijm1.matrix() * ab_hat) -> (temp = _delta_R_ijm1.matrix() * ab_hat * _d_R_bg_ijm1)
    temp = temp * _d_R_bg_ijm1;

    // Update jacobians of R, v, p with respect to bias_i.
    _d_R_bg_ij = dR.matrix().transpose() * _d_R_bg_ijm1 - Jr * _deltaT;
    _d_v_bg_ij = _d_v_bg_ijm1 - temp * _deltaT;
    _d_v_ba_ij = _d_v_ba_ijm1 - _delta_R_ijm1.matrix() * _deltaT;
    _d_p_bg_ij = _d_p_bg_ijm1 + _d_v_bg_ijm1 * _deltaT - temp * _deltaT2 / 2;
    _d_p_ba_ij = _d_p_ba_ijm1 + _d_v_ba_ijm1 * _deltaT - _delta_R_ijm1.matrix() * _deltaT2 / 2;
}

} // namespace cfsd