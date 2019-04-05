#include "cfsd/imu-preintegrator.hpp"

namespace cfsd {

ImuPreintegrator::ImuPreintegrator(const cfsd::Ptr<Map> pMap, const bool verbose) :
        _pMap(pMap),
        _verbose(verbose),
        _covNoise(),
        _covBias(),
        _betweenFrames(),
        _betweenKeyframes() {
    
    _samplingRate = Config::get<int>("samplingRate");
    _deltaT = 1.0 / (double)_samplingRate;
    _deltaT2 = _deltaT * _deltaT;

    double g = Config::get<double>("gravity");
    /*  imu coordinate system
              / x
             /
            ------ y
            |
            | z
    */
    _gravity << 0, 0, g;

    double sqrtDeltaT = std::sqrt(_deltaT);
    // Noise density of accelerometer and gyroscope measurements.
    //   Continuous-time model: sigma_g, unit: [rad/(s*sqrt(Hz))] or [rad/sqrt(s)]
    //                          sigma_a, unit: [m/(s^2*sqrt(Hz))] or [m/(s*sqrt(s))]
    //   Discrete-time model: sigma_gd = sigma_g/sqrt(delta_t), unit: [rad/s]
    //                        sigma_ad = sigma_a/sqrt(delta_t), unit: [m/s^2]
    double accNoiseD, gyrNoiseD;
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
    double accBiasD, gyrBiasD;
    // Unit [rad/s]
    gyrBiasD = Config::get<double>("gyrBias");
    // Convert unit [g] to [m/s^2]
    accBiasD = Config::get<double>("accBias") * g;
    
    // Covariance matrix of discrete-time noise [n_gd, n_ad]
    _covNoise.block<3, 3>(0, 0) = (gyrNoiseD * gyrNoiseD) * Eigen::Matrix3d::Identity();
    _covNoise.block<3, 3>(3, 3) = (accNoiseD * accNoiseD) * Eigen::Matrix3d::Identity();

    // Covariance matrix of discrete-time bias [b_gd, b_ad]
    _covBias.block<3, 3>(0, 0) = (gyrBiasD * gyrBiasD) * Eigen::Matrix3d::Identity();
    _covBias.block<3, 3>(3, 3) = (accBiasD * accBiasD) * Eigen::Matrix3d::Identity();

    // Initialize preintegration covariance (r,v,p,bg,ba, 15x15).
    _betweenFrames.covPreintegration_ij.block<6, 6>(9, 9) = _covBias;
    _betweenKeyframes.covPreintegration_ij.block<6, 6>(9, 9) = _covBias;
}

void ImuPreintegrator::setImgTimestamp(const long& imgTimestamp) {
    std::lock_guard<std::mutex> lockTime(_timeMutex);
    _imgTimestamp = imgTimestamp;
}

void ImuPreintegrator::updateBias() {
    _pMap->updateImuBias(_betweenFrames);
}

void ImuPreintegrator::process(const long& timestamp, const Eigen::Vector3d& gyr_jm1, const Eigen::Vector3d& acc_jm1) {
    // Between two consecutive camera frames:
    // Intermediate variables that will be used later.
    Eigen::Vector3d ub_gyr_jm1 = gyr_jm1 - _betweenFrames.bg_i; // unbiased gyr at time j-1
    Eigen::Vector3d ub_acc_jm1 = acc_jm1 - _betweenFrames.ba_i; // unbiased acc at time j-1
    Eigen::Vector3d omega = ub_gyr_jm1 * _deltaT;
    Sophus::SO3d dR = Sophus::SO3d::exp(omega);
    // Iteratively integrate.
    iterate(_betweenFrames, dR, ub_acc_jm1);
    // Compute Jr (right jacobian of SO3 rotation).
    Eigen::Matrix3d Jr = rightJacobianSO3(omega);
    // Intermediate variable that will be used later.
    Eigen::Matrix3d temp = _betweenFrames.delta_R_ijm1.matrix() * Sophus::SO3d::hat(ub_acc_jm1);
    // Noise propagation.
    propagate(_betweenFrames, dR, Jr, temp);
    // Jacobians of bias.
    jacobians(_betweenFrames, Jr, temp);
    // Update number of frames.
    _betweenFrames.numFrames++;

    // Between two keyframes:
    // Intermediate variables that will be used later.
    ub_gyr_jm1 = gyr_jm1 - _betweenKeyframes.bg_i; // unbiased gyr at time j-1
    ub_acc_jm1 = acc_jm1 - _betweenKeyframes.ba_i; // unbiased acc at time j-1
    omega = ub_gyr_jm1 * _deltaT;
    dR = Sophus::SO3d::exp(omega);
    // Iteratively integrate.
    iterate(_betweenKeyframes, dR, ub_acc_jm1);
    // Compute Jr (right jacobian of SO3 rotation).
    Jr = rightJacobianSO3(omega);
    // Intermediate variable that will be used later.
    temp = _betweenKeyframes.delta_R_ijm1.matrix() * Sophus::SO3d::hat(ub_acc_jm1);
    // Noise propagation.
    propagate(_betweenKeyframes, dR, Jr, temp);
    // Jacobians of bias.
    jacobians(_betweenKeyframes, Jr, temp);
    // Update number of frames.
    _betweenKeyframes.numFrames++;

    //TODO.........................................
    // Check if it's time for local optimization.
    std::lock_guard<std::mutex> lockTime(_timeMutex);
    if (_betweenFrames.numFrames > 5 && std::abs(timestamp - _imgTimestamp) < 1000) {
        if (_pMap->_isKeyframe)
            _betweenKeyframes = _betweenFrames;

        _pMap->pushImuConstraint(_betweenKeyframes, _gravity, _deltaT);
        _betweenFrames.reinitialize();
    }
}

void ImuPreintegrator::iterate(Preintegration& pre, const Sophus::SO3d& dR, const Eigen::Vector3d& ub_acc_jm1) {
    // Assign last iteration's "ij" value to this iteration's "ijm1".
    pre.delta_R_ijm1 = pre.delta_R_ij;
    pre.delta_v_ijm1 = pre.delta_v_ij;
    pre.delta_p_ijm1 = pre.delta_p_ij;

    // Update this iteration's "ij".
    pre.delta_R_ij = pre.delta_R_ijm1 * dR;
    pre.delta_v_ij = pre.delta_v_ijm1 + pre.delta_R_ijm1 * ub_acc_jm1 * _deltaT;
    pre.delta_p_ij = pre.delta_p_ijm1 + pre.delta_v_ijm1 * _deltaT + pre.delta_R_ijm1 * ub_acc_jm1 * _deltaT2 / 2;
}

Eigen::Matrix3d ImuPreintegrator::rightJacobianSO3(const Eigen::Vector3d& omega) {
    // Jr(omega) = d_exp(omega_hat) / d_omega
    // Jr(omega) = I - [1-cos(|omega|)] / (|omega|^2) * omega_hat + [|omega| - sin(|omega|)] / (|omega|^3) * omega_hat^2
    Eigen::Matrix3d Jr;

    // theta2 = |omega|^2
    // theta = |omega|
    const double theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const double theta = std::sqrt(theta2);
    
    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < _epsilon) {
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

Eigen::Matrix3d ImuPreintegrator::rightJacobianInverseSO3(const Eigen::Vector3d& omega) {
    // inv(Jr(omega)) = I + 0.5 * omega + [1 / |omega|^2 - (1+cos(|omega|) / (2*|omega|*sin(|omega)))] * omega_hat^2
    Eigen::Matrix3d JrInv;

    // theta2 = |omega|^2
    // theta = |omega|
    const double theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const double theta = std::sqrt(theta2);

    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < _epsilon) {
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

void ImuPreintegrator::propagate(Preintegration& pre, const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, const Eigen::Matrix3d& temp) {
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
    B.block<3, 3>(3, 3) = pre.delta_R_ijm1.matrix() * _deltaT;
    B.block<3, 3>(6, 3) = pre.delta_R_ijm1.matrix() * _deltaT2 / 2;

    // Update covariance matrix.
    pre.covPreintegration_ij.block<9, 9>(0, 0) = A * pre.covPreintegration_ij.block<9, 9>(0, 0) * A.transpose() + B * _covNoise * B.transpose();
}

void ImuPreintegrator::jacobians(Preintegration& pre, const Eigen::Matrix3d& Jr, Eigen::Matrix3d& temp) {
    // Jacobians of bias.
    pre.d_R_bg_ijm1 = pre.d_R_bg_ij;
    pre.d_v_ba_ijm1 = pre.d_v_ba_ij;
    pre.d_v_bg_ijm1 = pre.d_v_bg_ij;
    pre.d_p_ba_ijm1 = pre.d_p_ba_ij;
    pre.d_p_bg_ijm1 = pre.d_p_bg_ij;

    // (temp = _delta_R_ijm1.matrix() * ab_hat) -> (temp = _delta_R_ijm1.matrix() * ab_hat * _d_R_bg_ijm1)
    temp = temp * pre.d_R_bg_ijm1;

    // Update jacobians of R, v, p with respect to bias_i.
    pre.d_R_bg_ij = -pre.d_R_bg_ijm1 - Jr * _deltaT;
    pre.d_v_ba_ij = -pre.d_v_ba_ijm1 - pre.delta_R_ijm1.matrix() * _deltaT;
    pre.d_v_bg_ij = -pre.d_v_bg_ijm1 - temp * _deltaT;
    pre.d_p_ba_ij = pre.d_p_ba_ijm1 + pre.d_v_ba_ijm1 * _deltaT - pre.delta_R_ijm1.matrix() * _deltaT2 / 2;
    pre.d_p_bg_ij = pre.d_p_bg_ijm1 + pre.d_v_bg_ijm1 * _deltaT - temp * _deltaT2 / 2;
}


bool ImuPreintegrator::evaluate(const ImuConstraint& ic,
        const Eigen::Vector3d& r_i, const Eigen::Vector3d& v_i, const Eigen::Vector3d& p_i,
        const Eigen::Vector3d& bg_i, const Eigen::Vector3d& ba_i,
        const Eigen::Vector3d& r_j, const Eigen::Vector3d& v_j, const Eigen::Vector3d& p_j,
        const Eigen::Vector3d& bg_j, const Eigen::Vector3d& ba_j,
        double* residuals, double** jacobians) {

    // Map double* to Eigen Matrix.
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

    // Bias estimation update at time i, i.e. bias changes by a small amount delta_b during optimization.
    Eigen::Vector3d delta_bg = bg_i - ic.bg_i;
    Eigen::Vector3d delta_ba = ba_i - ic.ba_i;

    // residual(delta_R_ij)
    Sophus::SO3d R_i = Sophus::SO3d::exp(r_i);
    Sophus::SO3d R_j = Sophus::SO3d::exp(r_j);
    Sophus::SO3d tempR = ic.delta_R_ij * Sophus::SO3d::exp(ic.d_R_bg_ij * delta_bg);
    residual.block<3, 1>(0, 0) = (tempR.inverse() * R_i.inverse() * R_j).log();

    // residual(delta_v_ij)
    Eigen::Vector3d dv = v_j - v_i - _gravity * ic.dt;
    residual.block<3, 1>(3, 0) = R_i.inverse() * dv - (ic.delta_v_ij + ic.d_v_bg_ij * delta_bg + ic.d_v_ba_ij * delta_ba);

    // residual(delta_p_ij)
    Eigen::Vector3d dp = p_j - p_i - v_i * ic.dt - _gravity * ic.dt2 / 2;
    residual.block<3, 1>(6, 0) = R_i.inverse() * dp - (ic.delta_p_ij + ic.d_p_bg_ij * delta_bg + ic.d_p_ba_ij * delta_ba);

    // residual(delta_bg_ij)
    residual.block<3, 1>(9, 0) = bg_j - bg_i;

    // residual(delta_ba_ij)
    residual.block<3, 1>(12, 0) = ba_j - ba_i;

    // |r|^2 is defined as: r' * inv(cov) * r
    // Whereas in ceres, the square of residual is defined as: |x|^2 = x' * x
    // so we should construct such x from r in order to fit ceres solver.
    
    // Use cholesky decomposition: inv(cov) = L * L'
    // |r|^2 = r' * L * L' * r = (L' * r)' * L' * r
    // define x = L' * r (matrix 15x1)
    Eigen::Matrix<double, 15, 15> Lt = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(ic.covPreintegration_ij).matrixL().transpose();
    residual = Lt * residual;

    // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
    if (!jacobians) return true;

    // Inverse of right jacobian of residual(delta_R_ij)
    Eigen::Matrix3d JrInv = rightJacobianInverseSO3(residual.block<3, 1>(0, 0));

    // Jacobian(15x9) of residual(15x1) w.r.t. ParameterBlock[0](9x1), i.e. rvp_i
    if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 15, 9>> jacobian_i(jacobians[0]);

        jacobian_i.setZero();

        // jacobian of residual(delta_R_ij) with respect to r_i
        jacobian_i.block<3, 3>(0, 0) = -JrInv * R_j.matrix().transpose() * R_i.matrix();

        // jacobian of residual(delta_v_ij) with respect to r_i
        jacobian_i.block<3, 3>(3, 0) = Sophus::SO3d::hat(R_i.matrix().transpose() * dv);

        // jacobian of residual(delta_v_ij) with respect to v_i
        jacobian_i.block<3, 3>(3, 3) = -R_i.matrix().transpose();

        // jacobian of residual(delta_p_ij) with respect to r_i
        jacobian_i.block<3, 3>(6, 0) = Sophus::SO3d::hat(R_i.matrix().transpose() * dp);

        // jacobian of residual(delta_p_ij) with respect to v_i
        jacobian_i.block<3, 3>(6, 3) = -R_i.matrix().transpose() * ic.dt2;

        // jacobian of residual(delta_p_ij) with respect to p_i
        jacobian_i.block<3, 3>(6, 6) = -Eigen::Matrix3d::Identity();

        // since cost function is defined as: L' * r
        jacobian_i = Lt * jacobian_i;

        if (jacobian_i.maxCoeff() > 1e8 || jacobian_i.minCoeff() < -1e8)
            std::cout << "Numerical unstable in calculating jacobian_i!" << std::endl;
    }

    // Jacobian(15x6) of residuals(15x1) w.r.t. ParameterBlock[1](6x1), i.e. bga_i
    if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 15, 6>> jacobian_bi(jacobians[1]);

        jacobian_bi.setZero();

        // jacobian of residual(delta_R_ij) with respect to bg_i
        jacobian_bi.block<3, 3>(0, 0) = -JrInv * Sophus::SO3d::exp(residual.block<3, 1>(0, 0)).matrix().transpose() * rightJacobianSO3(ic.d_R_bg_ij * delta_bg) * ic.d_R_bg_ij;

        // jacobian of residual(delta_v_ij) with respect to bg_i
        jacobian_bi.block<3, 3>(3, 0) = -ic.d_v_bg_ij;

        // jacobian of residual(delta_v_ij) with respect to ba_i
        jacobian_bi.block<3, 3>(3, 3) = -ic.d_v_ba_ij;

        // jacobian of residual(delta_p_ij) with respect to bg_i
        jacobian_bi.block<3, 3>(6, 0) = -ic.d_p_bg_ij;

        // jacobian of residual(delta_p_ij) with respect to ba_i
        jacobian_bi.block<3, 3>(6, 3) = -ic.d_p_ba_ij;

        // jacobian of residual(delta_bg_ij) with respect to bg_i
        jacobian_bi.block<3, 3>(9, 0) = -Eigen::Matrix3d::Identity();

        // jacobian of residual(delta_ba_ij) with respect to ba_i
        jacobian_bi.block<3, 3>(12, 3) = -Eigen::Matrix3d::Identity();

        // since cost function is defined as: L' * r
        jacobian_bi = Lt * jacobian_bi;

        if (jacobian_bi.maxCoeff() > 1e8 || jacobian_bi.minCoeff() < -1e8)
            std::cout << "Numerical unstable in calculating jacobian_bi!" << std::endl;
    }

    // Jacobian(15x9) of residuals(15x1) w.r.t. ParameterBlock[2](9x1), i.e. rvp_j
    if (jacobians[2]) {
        Eigen::Map<Eigen::Matrix<double, 15, 9>> jacobian_j(jacobians[2]);

        jacobian_j.setZero();

        // jacobian of residual(delta_R_ij) with respect to r_j
        jacobian_j.block<3, 3>(0, 0) = JrInv;

        // jacobian of residual(delta_v_ij) with respect to v_j
        jacobian_j.block<3, 3>(3, 3) = R_i.matrix().transpose();

        // jacobian of residual(delta_p_ij) with respect to p_j
        jacobian_j.block<3, 3>(6, 6) = R_i.matrix().transpose() * R_j.matrix();

        // since cost function is defined as: L' * r
        jacobian_j = Lt * jacobian_j;

        if (jacobian_j.maxCoeff() > 1e8 || jacobian_j.minCoeff() < -1e8)
            std::cout << "Numerical unstable in calculating jacobian_j!" << std::endl;
    }

    // Jacobian(15x6) of residuals(15x1) w.r.t. ParameterBlock[3](6x1), i.e. bga_j
    if (jacobians[3]) {
        Eigen::Map<Eigen::Matrix<double, 15, 6>> jacobian_bj(jacobians[3]);

        jacobian_bj.setZero();

        // jacobian of residual(delta_bg_ij) with respect to bg_j
        jacobian_bj.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();

        // jacobian of residual(delta_ba_ij) with respect to ba_j
        jacobian_bj.block<3, 3>(12, 3) = Eigen::Matrix3d::Identity();

        // since cost function is defined as: L' * r
        jacobian_bj = Lt * jacobian_bj;

        if (jacobian_bj.maxCoeff() > 1e8 || jacobian_bj.minCoeff() < -1e8)
            std::cout << "Numerical unstable in calculating jacobian_bj!" << std::endl;
    }

    return true;
}

} // namespace cfsd