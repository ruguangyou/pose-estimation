#include "cfsd/imu-preintegrator.hpp"

namespace cfsd {

ImuPreintegrator::ImuPreintegrator(const cfsd::Ptr<Optimizer>& pOptimizer, const bool verbose) :
        _pOptimizer(pOptimizer), 
        _verbose(verbose), 
        _epsilon(1e-5),
        _R_i(SophusSO3Type()),
        _v_i(EigenVector3Type::Zero()),
        _p_i(EigenVector3Type::Zero()),
        _biasGyr(EigenVector3Type::Zero()),
        _biasAcc(EigenVector3Type::Zero()),
        _delta_R_ij(SophusSO3Type()), 
        _delta_v_ij(EigenVector3Type::Zero()), 
        _delta_p_ij(EigenVector3Type::Zero()),
        _d_R_bg_ij(EigenMatrix3Type::Zero()), 
        _d_v_ba_ij(EigenMatrix3Type::Zero()), 
        _d_v_bg_ij(EigenMatrix3Type::Zero()),
        _d_p_ba_ij(EigenMatrix3Type::Zero()), 
        _d_p_bg_ij(EigenMatrix3Type::Zero()), 
        _covPreintegration(Eigen::Matrix<precisionType,15,15>::Zero()), 
        _covMeasurement(Eigen::Matrix<precisionType,6,6>::Zero()),
        _covBias(Eigen::Matrix<precisionType,6,6>::Zero()) {
    
    _samplingRate = Config::get<int>("samplingRate");
    _deltaT = 1.0 / (precisionType)_samplingRate;
    _deltaT2 = _deltaT * _deltaT;

    // _iters = _samplingRate / Config::get<int>("cameraFrequency");
    _iters = 200;
    _dt = _deltaT * _iters;
    _dt2 = _dt * _dt;

    precisionType g = Config::get<precisionType>("gravity");
    /*  camera coordinate system
              / z (yaw)
             /
            ------ x (roll)
            |
            | y (pitch)
    */
    _gravity << 0, g, 0;

    precisionType sqrtDeltaT = std::sqrt(_deltaT);
    // Convert unit [rad/sqrt(s)] to [rad/s]
    _gyrNoiseD = Config::get<precisionType>("gyrNoise") / sqrtDeltaT;
    // Convert unit [g*sqrt(s)] to [m/s^2].
    _accNoiseD = Config::get<precisionType>("accNoise") * g / sqrtDeltaT;
    // Unit [rad/s]
    _gyrBiasD = Config::get<precisionType>("gyrBias");
    // Convert unit [g] to [m/s^2]
    _accBiasD = Config::get<precisionType>("accBias") * g;
    
    // Initialize covariance.
    _covMeasurement.block<3, 3>(0, 0) = (_gyrNoiseD * _gyrNoiseD) * EigenMatrix3Type::Identity();
    _covMeasurement.block<3, 3>(3, 3) = (_accNoiseD * _accNoiseD) * EigenMatrix3Type::Identity();
    _covBias.block<3, 3>(0, 0) = (_gyrBiasD * _gyrBiasD) * EigenMatrix3Type::Identity();
    _covBias.block<3, 3>(3, 3) = (_accBiasD * _accBiasD) * EigenMatrix3Type::Identity();
    _covPreintegration.block<6, 6>(9, 9) = _covBias;
}

bool ImuPreintegrator::isProcessable() {
    std::lock_guard<std::mutex> lockGyr(_gyrMutex);
    std::lock_guard<std::mutex> lockAcc(_accMutex);

    return (_gyrQueue.size() >= _iters && _accQueue.size() >= _iters);
}

void ImuPreintegrator::reinitialize() {
    _delta_R_ij = SophusSO3Type();
    _delta_v_ij.setZero();
    _delta_p_ij.setZero();

    _d_R_bg_ij.setZero();
    _d_v_ba_ij.setZero();
    _d_v_bg_ij.setZero();
    _d_p_ba_ij.setZero();
    _d_p_bg_ij.setZero();
}

void ImuPreintegrator::process(const long& timestamp) {
    #ifdef DEBUG_IMU
    auto start = std::chrono::steady_clock::now();
    #endif

    for (int i = 0; i < _iters; i++) {
        { // Out of this local scope the locks will die.
            std::lock_guard<std::mutex> lockGyr(_gyrMutex);
            std::lock_guard<std::mutex> lockAcc(_accMutex);
        
            // Acc and gyr to be processed.
            _acc = _accQueue.front();
            _gyr = _gyrQueue.front();

            // Pop out processed measurements.
            _accQueue.pop();
            _gyrQueue.pop();
            _timestampQueue.pop();
        }

        // #ifdef DEBUG_IMU
        // std::cout << "(camera)\nacc:\n" << _acc << "gyr:\n" << _gyr << std::endl;
        // #endif
        
        // Intermediate variables that will be used later.
        EigenVector3Type omega = (_gyr - _biasGyr) * _deltaT;
        SophusSO3Type dR = SophusSO3Type::exp(omega);

        // Iteratively integrate.
        iterate(dR);

        // Compute Jr (right jacobian of SO3 rotation).
        EigenMatrix3Type Jr = rightJacobianSO3(omega);

        // Intermediate variable that will be used later.
        EigenMatrix3Type temp = _delta_R_ijm1.matrix() * SophusSO3Type::hat(_acc - _biasAcc);
        // EigenVector3Type ab = _acc - _biasAcc;
        // EigenMatrix3Type ab_hat;
        // ab_hat <<   0.0, -ab(2),  ab(1),
        //           ab(2),    0.0, -ab(0),
        //          -ab(1),  ab(0),    0.0;
        // EigenMatrix3Type temp = _delta_R_ijm1.matrix() * ab_hat;

        // Noise propagation.
        propagate(dR, Jr, temp);

        // Jacobians of bias.
        jacobians(Jr, temp);
    }

    #ifdef DEBUG_IMU
    auto end = std::chrono::steady_clock::now();
    std::cout << "Integration elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
    start = std::chrono::steady_clock::now();
    #endif

    recover();

    // Local optimization.
    // shared_from_this() returns a std::shared_ptr<T> that shares ownership of *this with all existing std::shared_ptr that refer to *this.
    _pOptimizer->localOptimize(shared_from_this());
    
    reinitialize();

    #ifdef DEBUG_IMU
    end = std::chrono::steady_clock::now();
    std::cout << "Optimization elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
    #endif
}

void ImuPreintegrator::iterate(const SophusSO3Type& dR) {
    // Assign last iteration's "ij" value to this iteration's "ijm1".
    _delta_R_ijm1 = _delta_R_ij;
    _delta_v_ijm1 = _delta_v_ij;
    _delta_p_ijm1 = _delta_p_ij;

    // Update this iteration's "ij".
    _delta_R_ij = _delta_R_ijm1 * dR;
    _delta_v_ij = _delta_v_ijm1 + _delta_R_ijm1 * (_acc - _biasAcc) * _deltaT;
    _delta_p_ij = _delta_p_ijm1 + _delta_v_ijm1 * _deltaT + _delta_R_ijm1 * (_acc - _biasAcc) * _deltaT2 / 2;
}

EigenMatrix3Type ImuPreintegrator::rightJacobianSO3(const EigenVector3Type& omega) {
    // Jr(omega) = d_exp(omega_hat) / d_omega
    // Jr(omega) = I - [1-cos(|omega|)] / (|omega|^2) * omega_hat + [|omega| - sin(|omega|)] / (|omega|^3) * omega_hat^2
    EigenMatrix3Type Jr;

    // theta2 = |omega|^2
    // theta = |omega|
    const precisionType theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const precisionType theta = std::sqrt(theta2);
    
    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < _epsilon) {
        // If theta is small enough, we view it as 0.
        Jr = EigenMatrix3Type::Identity();
    }
    else {
        EigenMatrix3Type omega_hat = SophusSO3Type::hat(omega);
        // omega_hat <<      0.0, -omega(2),  omega(1),
        //              omega(2),       0.0, -omega(0),
        //             -omega(1),  omega(0),       0.0;
        Jr = EigenMatrix3Type::Identity() - (1-std::cos(theta)) / theta2 * omega_hat + (theta - std::sin(theta)) / (theta2 * theta) * omega_hat * omega_hat;
    }

    return Jr;
}

EigenMatrix3Type ImuPreintegrator::rightJacobianInverseSO3(const EigenVector3Type& omega) {
    // inv(Jr(omega)) = I + 0.5 * omega + [1 / |omega|^2 - (1+cos(|omega|) / (2*|omega|*sin(|omega)))] * omega_hat^2
    EigenMatrix3Type JrInv;

    // theta2 = |omega|^2
    // theta = |omega|
    const precisionType theta2 = omega(0)*omega(0) + omega(1)*omega(1) + omega(2)*omega(2);
    const precisionType theta = std::sqrt(theta2);

    // The right jacobian reduces to the identity matrix for |omega| == 0.
    if (theta < _epsilon) {
        // If theta is small enough, we view it as 0.
        JrInv = EigenMatrix3Type::Identity();
    }
    else {
        EigenMatrix3Type omega_hat = SophusSO3Type::hat(omega);
        // omega_hat <<      0.0, -omega(2),  omega(1),
        //              omega(2),       0.0, -omega(0),
        //             -omega(1),  omega(0),       0.0;
        JrInv = EigenMatrix3Type::Identity() + omega_hat / 2 + (1 / theta2 - (1 + std::cos(theta)) / (2 * theta * std::sin(theta))) * omega_hat * omega_hat;
    }

    return JrInv;
}

void ImuPreintegrator::propagate(const SophusSO3Type& dR, const EigenMatrix3Type& Jr, const EigenMatrix3Type& temp) {
    // Noise propagation: n_ij = A * n_ijm1 + B * n_measurement
    // Covriance propagation: cov_ij = A * cov_ijm1 * A' + B * cov_measurement * B'

    // Construct matrix A and B.
    Eigen::Matrix<precisionType,9,9> A;
    Eigen::Matrix<precisionType,9,6> B;
    A.setZero();
    B.setZero();
    A.block<3, 3>(0, 0) = dR.matrix().transpose();
    A.block<3, 3>(3, 0) = -temp * _deltaT;  // temp = _delta_R_ijm1.matrix() * ab_hat
    A.block<3, 3>(3, 3) = EigenMatrix3Type::Identity();
    A.block<3, 3>(6, 0) = -temp * _deltaT2 / 2;
    A.block<3, 3>(6, 3) = _deltaT * EigenMatrix3Type::Identity();
    A.block<3, 3>(6, 6) = EigenMatrix3Type::Identity();
    B.block<3, 3>(0, 0) = Jr * _deltaT;
    B.block<3, 3>(3, 3) = _delta_R_ijm1.matrix() * _deltaT;
    B.block<3, 3>(6, 3) = _delta_R_ijm1.matrix() * _deltaT2 / 2;

    // Update covariance matrix.
    _covPreintegration.block<9, 9>(0, 0) = A * _covPreintegration.block<9, 9>(0, 0) * A.transpose() + B * _covMeasurement * B.transpose();
}

void ImuPreintegrator::jacobians(const EigenMatrix3Type& Jr, EigenMatrix3Type& temp) {
    // Jacobians of bias.
    _d_R_bg_ijm1 = _d_R_bg_ij;
    _d_v_ba_ijm1 = _d_v_ba_ij;
    _d_v_bg_ijm1 = _d_v_bg_ij;
    _d_p_ba_ijm1 = _d_p_ba_ij;
    _d_p_bg_ijm1 = _d_p_bg_ij;

    // (temp = _delta_R_ijm1.matrix() * ab_hat) -> (temp = _delta_R_ijm1.matrix() * ab_hat * _d_R_bg_ijm1)
    temp = temp * _d_R_bg_ijm1;

    // Update jacobians of R, v, p with respect to bias.
    _d_R_bg_ij = -_d_R_bg_ijm1 - Jr * _deltaT;
    _d_v_ba_ij = -_d_v_ba_ijm1 - _delta_R_ijm1.matrix() * _deltaT;
    _d_v_bg_ij = -_d_v_bg_ijm1 - temp * _deltaT;
    _d_p_ba_ij = _d_p_ba_ijm1 + _d_v_ba_ijm1 * _deltaT - _delta_R_ijm1.matrix() * _deltaT2 / 2;
    _d_p_bg_ij = _d_p_bg_ijm1 + _d_v_bg_ijm1 * _deltaT - temp * _deltaT2 / 2;
}

void ImuPreintegrator::recover() {
    // Roughly estimate the state at time j without considering bias update.
    // This is provided as initial value for ceres optimization.
    _R_j = _R_i * _delta_R_ij;
    _v_j = _v_i + _gravity * _dt + _R_i * _delta_v_ij;
    _p_j = _p_i + _v_i * _dt + _gravity * _dt2 / 2 + _R_i * _delta_p_ij;
    
    // _R_j = _R_i * _delta_R_ij * SophusSO3Type::exp(_d_R_bg_ij * _delta_bg);
    // _v_j = _v_i + _gravity * (_deltaT * n) + _R_i.matrix() * (_delta_v_ij + _d_v_bg_ij * _delta_bg + _d_v_ba_ij * _delta_ba);
    // _p_j = _p_i + _v_i * (_deltaT * n) + 0.5 * _gravity * (_deltaT2 * n * n) + _R_i * (_delta_p_ij + _d_p_bg_ij * _delta_bg + _d_p_ba_ij * _delta_ba);
}

void ImuPreintegrator::getParameters(double* rvp_i, double* rvp_j) {
    EigenVector3Type r_i = _R_i.log();
    EigenVector3Type r_j = _R_j.log();

    rvp_i[0] = r_i(0);
    rvp_i[1] = r_i(1);
    rvp_i[2] = r_i(2);
    rvp_i[3] = _v_i(0);
    rvp_i[4] = _v_i(1);
    rvp_i[5] = _v_i(2);
    rvp_i[6] = _p_i(0);
    rvp_i[7] = _p_i(1);
    rvp_i[8] = _p_i(2);
    
    rvp_j[0] = r_j(0);
    rvp_j[1] = r_j(1);
    rvp_j[2] = r_j(2);
    rvp_j[3] = _v_j(0);
    rvp_j[4] = _v_j(1);
    rvp_j[5] = _v_j(2);
    rvp_j[6] = _p_j(0);
    rvp_j[7] = _p_j(1);
    rvp_j[8] = _p_j(2);

    #ifdef DEBUG_IMU
    std::cout << "position i: " << rvp_i[6] << ", " << rvp_i[7] << ", " << rvp_i[8] << std::endl;
    std::cout << "position j: " << rvp_j[6] << ", " << rvp_j[7] << ", " << rvp_j[8] << std::endl;
    #endif
}

bool ImuPreintegrator::evaluate(
        const EigenVector3Type& r_i, const EigenVector3Type& v_i, const EigenVector3Type& p_i,
        const EigenVector3Type& r_j, const EigenVector3Type& v_j, const EigenVector3Type& p_j,
        const EigenVector3Type& delta_bg, const EigenVector3Type& delta_ba,
        double* residuals, double** jacobians) {

    // Map double* to Eigen Matrix.
    Eigen::Map<Eigen::Matrix<precisionType, 15, 1>> residual(residuals);

    // residual(delta_R_ij)
    SophusSO3Type R_i = SophusSO3Type::exp(r_i);
    SophusSO3Type R_j = SophusSO3Type::exp(r_j);
    SophusSO3Type tempR = _delta_R_ij * SophusSO3Type::exp(_d_R_bg_ij * delta_bg);
    residual.block<3, 1>(0, 0) = (tempR.inverse() * R_i.inverse() * R_j).log();

    // residual(delta_v_ij)
    EigenVector3Type dv = v_j - v_i - _gravity * _dt;
    residual.block<3, 1>(3, 0) = R_i.inverse() * dv - (_delta_v_ij + _d_v_bg_ij * delta_bg + _d_v_ba_ij * delta_ba);

    // residual(delta_p_ij)
    EigenVector3Type dp = p_j - p_i - v_i * _dt - _gravity * _dt2 / 2;
    residual.block<3, 1>(6, 0) = R_i.inverse() * dp - (_delta_p_ij + _d_p_bg_ij * delta_bg + _d_p_ba_ij * delta_ba);

    // residual(delta_bg_ij)
    residual.block<3, 1>(9, 0) = delta_bg;

    // residual(delta_ba_ij)
    residual.block<3, 1>(12, 0) = delta_ba;

    // |r|^2 is defined as: r' * inv(cov) * r
    // Whereas in ceres, the square of residual is defined as: |x|^2 = x' * x
    // so we should construct such x from r in order to fit ceres solver.
    
    // Use cholesky decomposition: inv(cov) = L * L'
    // |r|^2 = r' * L * L' * r = (L' * r)' * L' * r
    // define x = L' * r (matrix 15x1)
    Eigen::Matrix<precisionType, 15, 15> L = Eigen::LLT<Eigen::Matrix<precisionType, 15, 15>>(_covPreintegration).matrixL();
    residual = L.transpose() * residual;

    // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
    if (!jacobians) return true;

    // Inverse of right jacobian of residual(delta_R_ij)
    EigenMatrix3Type JrInv = rightJacobianInverseSO3(residual.block<3, 1>(0, 0));

    // Jacobian(15x9) of residual(15x1) w.r.t. ParameterBlock[0](9x1), i.e. rvp_i
    if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<precisionType, 15, 9>> jacobian_i(jacobians[0]);

        jacobian_i.setZero();

        // jacobian of residual(delta_R_ij) with respect to r_i
        jacobian_i.block<3, 3>(0, 0) = -JrInv * R_j.matrix().transpose() * R_i.matrix();

        // jacobian of residual(delta_v_ij) with respect to r_i
        jacobian_i.block<3, 3>(3, 0) = SophusSO3Type::hat(R_i.matrix().transpose() * dv);

        // jacobian of residual(delta_v_ij) with respect to v_i
        jacobian_i.block<3, 3>(3, 3) = -R_i.matrix().transpose();

        // jacobian of residual(delta_p_ij) with respect to r_i
        jacobian_i.block<3, 3>(6, 0) = SophusSO3Type::hat(R_i.matrix().transpose() * dp);

        // jacobian of residual(delta_p_ij) with respect to v_i
        jacobian_i.block<3, 3>(6, 3) = -R_i.matrix().transpose() * _dt2;

        // jacobian of residual(delta_p_ij) with respect to p_i
        jacobian_i.block<3, 3>(6, 6) = -EigenMatrix3Type::Identity();
    }

    // Jacobian(15x9) of residuals(15x1) w.r.t. ParameterBlock[1](9x1), i.e. rvp_j
    if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<precisionType, 15, 9>> jacobian_j(jacobians[1]);

        jacobian_j.setZero();

        // jacobian of residual(delta_R_ij) with respect to r_j
        jacobian_j.block<3, 3>(0, 0) = JrInv;

        // jacobian of residual(delta_v_ij) with respect to v_j
        jacobian_j.block<3, 3>(3, 3) = R_i.matrix().transpose();

        // jacobian of residual(delta_p_ij) with respect to p_j
        jacobian_j.block<3, 3>(6, 6) = R_i.matrix().transpose() * R_j.matrix();
    }

    // Jacobian(15x6) of residuals(15x1) w.r.t. ParameterBlock[2](6x1), i.e. bg_ba
    if (jacobians[2]) {
        Eigen::Map<Eigen::Matrix<precisionType, 15, 6>> jacobian_bias(jacobians[2]);

        jacobian_bias.setZero();

        // jacobian of residual(delta_R_ij) with respect to delta_bg
        jacobian_bias.block<3, 3>(0, 0) = -JrInv * SophusSO3Type::exp(residual.block<3, 1>(0, 0)).matrix().transpose() * rightJacobianSO3(_d_R_bg_ij * delta_bg) * _d_R_bg_ij;

        // jacobian of residual(delta_v_ij) with respect to delta_bg
        jacobian_bias.block<3, 3>(3, 0) = -_d_v_bg_ij;

        // jacobian of residual(delta_v_ij) with respect to delta_ba
        jacobian_bias.block<3, 3>(3, 3) = -_d_v_ba_ij;

        // jacobian of residual(delta_p_ij) with respect to delta_bg
        jacobian_bias.block<3, 3>(6, 0) = -_d_p_bg_ij;

        // jacobian of residual(delta_p_ij) with respect to delta_ba
        jacobian_bias.block<3, 3>(6, 3) = -_d_p_ba_ij;

        // jacobian of residual(delta_bg_ij) with respect to delta_bg
        jacobian_bias.block<3, 3>(9, 0) = EigenMatrix3Type::Identity();

        // jacobian of residual(delta_ba_ij) with respect to delta_ba
        jacobian_bias.block<3, 3>(12, 3) = EigenMatrix3Type::Identity();
    }

    return true;
}

void ImuPreintegrator::updateState(double* rvp_j, double* bg_ba) {
    _R_i = SophusSO3Type::exp(EigenVector3Type(rvp_j[0], rvp_j[1], rvp_j[2]));
    _v_i = EigenVector3Type(rvp_j[3], rvp_j[4], rvp_j[5]);
    _p_i = EigenVector3Type(rvp_j[6], rvp_j[7], rvp_j[8]);

    _biasGyr += EigenVector3Type(bg_ba[0], bg_ba[1], bg_ba[2]);
    _biasAcc += EigenVector3Type(bg_ba[3], bg_ba[4], bg_ba[5]);

    #ifdef DEBUG_IMU
    std::cout << "bias gyr: " << _biasGyr(0) << ", " << _biasGyr(1) << ", " << _biasGyr(2) << std::endl;
    std::cout << "bias acc: " << _biasAcc(0) << ", " << _biasAcc(1) << ", " << _biasAcc(2) << std::endl;
    #endif
}

void ImuPreintegrator::collectAccData(const long& timestamp, const float& accX, const float& accY, const float& accZ) {
    std::lock_guard<std::mutex> lockAcc(_accMutex);

    if (_accQueue.size() == _timestampQueue.size())
        _timestampQueue.push(timestamp);
    
    EigenVector3Type acc;
    /*  IMU coordinate system => camera coordinate system
              / x (roll)                  / z (yaw)
             /                           /
            ------ y (pitch)            ------ x (roll)
            |                           |
            | z (yaw)                   | y (pitch)
        
        last year's proxy-ellipse2n (this is what will be received if using replay data collected in 2018-12-05)
                 z |  / x
                   | /
           y _ _ _ |/
    */

    // why accX is negative? even when the vehicle is still?
    // acc << -accY, -accZ, accX;
    acc << -accY, -accZ, accX; // converted from last year's coordinate system
    // acc << -accY, -accZ, accX+0.37; // compensation for accX for now, need calibration later
    
    _accQueue.push(acc);
}

void ImuPreintegrator::collectGyrData(const long& timestamp, const float& gyrX, const float& gyrY, const float& gyrZ) {
    std::lock_guard<std::mutex> lockGyr(_gyrMutex);

    if (_gyrQueue.size() == _timestampQueue.size())
        _timestampQueue.push(timestamp);
    
    EigenVector3Type gyr;
    /*  IMU coordinate system => camera coordinate system
              / x (roll)                  / z (yaw)
             /                           /
            ------ y (pitch)            ------ x (roll)
            |                           |
            | z (yaw)                   | y (pitch)
        
        last year's proxy-ellipse2n (this is what will be received if using replay data collected in 2018-12-05)
                 z |  / x
                   | /
           y _ _ _ |/
    */
    gyr << -gyrY, -gyrZ, gyrX; // converted from last year's coordinate system
    
    _gyrQueue.push(gyr);
}

} // namespace cfsd