#include <iostream>
#include <chrono>
#include <cmath>

// Eigen
#include <Eigen/Core>       // Matrix and Array classes, basic linear algebra (including triangular and selfadjoint products), array manipulation
#include <Eigen/Geometry>   // Transform, Translation, Scaling, Rotation2D and 3D rotations (Quaternion, AngleAxis)
// Sophus
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

class ImuPreintegrator {
  public:  
    // Variables need to be calculated in preintegration theory.
    struct Preintegration {
        Preintegration() : covPreintegration_ij(Eigen::Matrix<double,15,15>::Zero()),
                           bg_i(Eigen::Vector3d::Zero()), ba_i(Eigen::Vector3d::Zero()),
                           delta_R_ij(Sophus::SO3d()), delta_R_ijm1(Sophus::SO3d()),
                           delta_v_ij(Eigen::Vector3d::Zero()), delta_v_ijm1(Eigen::Vector3d::Zero()),
                           delta_p_ij(Eigen::Vector3d::Zero()), delta_p_ijm1(Eigen::Vector3d::Zero()),
                           d_R_bg_ij(Eigen::Matrix3d::Zero()), d_R_bg_ijm1(Eigen::Matrix3d::Zero()),
                           d_v_bg_ij(Eigen::Matrix3d::Zero()), d_v_bg_ijm1(Eigen::Matrix3d::Zero()),
                           d_v_ba_ij(Eigen::Matrix3d::Zero()), d_v_ba_ijm1(Eigen::Matrix3d::Zero()),
                           d_p_bg_ij(Eigen::Matrix3d::Zero()), d_p_bg_ijm1(Eigen::Matrix3d::Zero()),
                           d_p_ba_ij(Eigen::Matrix3d::Zero()), d_p_ba_ijm1(Eigen::Matrix3d::Zero()),
                           numFrames(0) {}

        void reinitialize() {
            delta_R_ijm1 = Sophus::SO3d();
            delta_v_ijm1.setZero();
            delta_p_ijm1.setZero();
            d_R_bg_ijm1.setZero();
            d_v_ba_ijm1.setZero();
            d_v_bg_ijm1.setZero();
            d_p_ba_ijm1.setZero();
            d_p_bg_ijm1.setZero();

            delta_R_ij = Sophus::SO3d();
            delta_v_ij.setZero();
            delta_p_ij.setZero();
            d_R_bg_ij.setZero();
            d_v_ba_ij.setZero();
            d_v_bg_ij.setZero();
            d_p_ba_ij.setZero();
            d_p_bg_ij.setZero();

            covPreintegration_ij.block<9,9>(0,0).setZero();
            numFrames = 0;
        }

        // Covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p, delta_bg, delta_ba]
        Eigen::Matrix<double,15,15> covPreintegration_ij;

        // Bias of gyroscope and accelerometer at time i.
        Eigen::Vector3d bg_i, ba_i;
        
        // Preintegrated delta_R, delta_v, delta_p, iterate from (i,j-1) to (i,j)
        Sophus::SO3d delta_R_ij, delta_R_ijm1; // 'm1' means minus one
        Eigen::Vector3d delta_v_ij, delta_v_ijm1;
        Eigen::Vector3d delta_p_ij, delta_p_ijm1;

        // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
        Eigen::Matrix3d d_R_bg_ij, d_R_bg_ijm1;
        Eigen::Matrix3d d_v_bg_ij, d_v_bg_ijm1;
        Eigen::Matrix3d d_v_ba_ij, d_v_ba_ijm1;
        Eigen::Matrix3d d_p_bg_ij, d_p_bg_ijm1;
        Eigen::Matrix3d d_p_ba_ij, d_p_ba_ijm1;
    
        // The number of imu frames between two camera frames.
        int numFrames;
    };

  public:
    ImuPreintegrator(const bool verbose);
    
    // Reinitialize after finishing processing IMU measurements between two consecutive camera frames.
    void reinitialize();

    // Take measurements and perform preintegration, jacobians calculation and noise propagation.
    void process(const long& timestamp, const Eigen::Vector3d& gyr_jm1, const Eigen::Vector3d& acc_jm1);

    // Iteratively preintegrate IMU measurements.
    void iterate(const Sophus::SO3d& dR, const Eigen::Vector3d& ub_acc_jm1);

    // Calculate right jacobian and the inverse of SO3.
    Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& omega);
    Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& omega);

    // Propagate preintegration noise.
    void propagate(const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, const Eigen::Matrix3d& temp, const Eigen::Matrix3d& tempKey);

    // Calculate jacobians of R, v, p with respect to bias.
    void jacobians(const Eigen::Matrix3d& Jr, Eigen::Matrix3d& temp, Eigen::Matrix3d& tempKey);

  private:
    bool _verbose;

    // A very small number that helps determine if a rotation is close to zero.
    double _epsilon{1e-5};

    // IMU parameters:
    // Sampling frequency.
    int _samplingRate{200};
    
    // Sampling time (1 / _samplingRate); _deltaT2 = _deltaT * _deltaT
    double _deltaT{0.005}, _deltaT2{0.0025};
    
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

    int iters{0};
    int iterKeys{0};
};

ImuPreintegrator::ImuPreintegrator(const bool verbose) :
        _verbose(verbose),
        _covNoise(),
        _covBias(),
        _betweenFrames(),
        _betweenKeyframes() {

    double g = 9.81;
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
    gyrNoiseD = 0.000617;
    // Convert unit [g*sqrt(s)] to [m/s^2].
    accNoiseD = 0.000806;

    // Bias random walk noise density of accelerometer and gyroscope.
    //   Continuous-time model: sigma_bg, unit: [rad/(s^2*sqrt(Hz))] or [rad/(s*sqrt(s))]
    //                          sigma_ba, unit: [m/(s^3*sqrt(Hz))] or [m/(s^2*sqrt(s))]
    //   Discrete-time model: sigma_bgd = sigma_bg*sqrt(delta_t), unit: [rad/s]
    //                        sigma_bad = sigma_ba*sqrt(delta_t), unit: [m/s^2]
    // (Bias are modelled with a "Brownian motion" process, also termed a "Wiener process", or "random walk" in discrete-time)
    double accBiasD, gyrBiasD;
    // Unit [rad/s]
    gyrBiasD = 3.3937e-5;
    // Convert unit [g] to [m/s^2]
    accBiasD = 0.000014 * g;
    
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


void ImuPreintegrator::process(const long& timestamp, const Eigen::Vector3d& gyr_jm1, const Eigen::Vector3d& acc_jm1) {
    // Intermediate variables that will be used later.
    Eigen::Vector3d ub_gyr_jm1 = gyr_jm1 - _betweenFrames.bg_i; // unbiased gyr at time j-1
    Eigen::Vector3d ub_acc_jm1 = acc_jm1 - _betweenFrames.ba_i; // unbiased acc at time j-1
    Eigen::Vector3d omega = ub_gyr_jm1 * _deltaT;
    Sophus::SO3d dR = Sophus::SO3d::exp(omega);

    // Iteratively integrate.
    iterate(dR, ub_acc_jm1);

    // Compute Jr (right jacobian of SO3 rotation).
    Eigen::Matrix3d Jr = rightJacobianSO3(omega);

    // Intermediate variable that will be used later.
    Eigen::Matrix3d temp = _betweenFrames.delta_R_ijm1.matrix() * Sophus::SO3d::hat(ub_acc_jm1);
    Eigen::Matrix3d tempKey = _betweenKeyframes.delta_R_ijm1.matrix() * Sophus::SO3d::hat(ub_acc_jm1);

    // Noise propagation.
    propagate(dR, Jr, temp, tempKey);

    // Jacobians of bias.
    jacobians(Jr, temp, tempKey);

    _betweenFrames.numFrames++;

    iters++; iterKeys++;

    if (iters == 20) {
        _betweenFrames.reinitialize();
        iters = 0;
    }

    // if (iterKeys == 60) {
    //     _betweenKeyframes.reinitialize();
    //     iterKeys = 0;
    // }
}

void ImuPreintegrator::iterate(const Sophus::SO3d& dR, const Eigen::Vector3d& ub_acc_jm1) {
    // Between two consecutive camera frames.
    // Assign last iteration's "ij" value to this iteration's "ijm1".
    _betweenFrames.delta_R_ijm1 = _betweenFrames.delta_R_ij;
    _betweenFrames.delta_v_ijm1 = _betweenFrames.delta_v_ij;
    _betweenFrames.delta_p_ijm1 = _betweenFrames.delta_p_ij;
    // Update this iteration's "ij".
    _betweenFrames.delta_R_ij = _betweenFrames.delta_R_ijm1 * dR;
    _betweenFrames.delta_v_ij = _betweenFrames.delta_v_ijm1 + _betweenFrames.delta_R_ijm1 * ub_acc_jm1 * _deltaT;
    _betweenFrames.delta_p_ij = _betweenFrames.delta_p_ijm1 + _betweenFrames.delta_v_ijm1 * _deltaT + _betweenFrames.delta_R_ijm1 * ub_acc_jm1 * _deltaT2 / 2;

    // Between two keyframes.
    // _betweenKeyframes.delta_R_ijm1 = _betweenKeyframes.delta_R_ij;
    // _betweenKeyframes.delta_v_ijm1 = _betweenKeyframes.delta_v_ij;
    // _betweenKeyframes.delta_p_ijm1 = _betweenKeyframes.delta_p_ij;
    // _betweenKeyframes.delta_R_ij = _betweenKeyframes.delta_R_ijm1 * dR;
    // _betweenKeyframes.delta_v_ij = _betweenKeyframes.delta_v_ijm1 + _betweenKeyframes.delta_R_ijm1 * ub_acc_jm1 * _deltaT;
    // _betweenKeyframes.delta_p_ij = _betweenKeyframes.delta_p_ijm1 + _betweenKeyframes.delta_v_ijm1 * _deltaT + _betweenKeyframes.delta_R_ijm1 * ub_acc_jm1 * _deltaT2 / 2;
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

void ImuPreintegrator::propagate(const Sophus::SO3d& dR, const Eigen::Matrix3d& Jr, const Eigen::Matrix3d& temp, const Eigen::Matrix3d& tempKey) {
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
    B.block<3, 3>(3, 3) = _betweenFrames.delta_R_ijm1.matrix() * _deltaT;
    B.block<3, 3>(6, 3) = _betweenFrames.delta_R_ijm1.matrix() * _deltaT2 / 2;

    // Update covariance matrix between two frames.
    _betweenFrames.covPreintegration_ij.block<9, 9>(0, 0) = A * _betweenFrames.covPreintegration_ij.block<9, 9>(0, 0) * A.transpose() + B * _covNoise * B.transpose();

    // Update covarianve matrix between two keyframes.
    // A.block<3, 3>(3, 0) = -tempKey * _deltaT;  // tempKey = _betweenKeyframes.delta_R_ijm1.matrix() * ab_hat
    // A.block<3, 3>(6, 0) = -tempKey * _deltaT2 / 2;
    // B.block<3, 3>(3, 3) = _betweenKeyframes.delta_R_ijm1.matrix() * _deltaT;
    // B.block<3, 3>(6, 3) = _betweenKeyframes.delta_R_ijm1.matrix() * _deltaT2 / 2;
    // _betweenKeyframes.covPreintegration_ij.block<9, 9>(0, 0) = A * _betweenKeyframes.covPreintegration_ij.block<9, 9>(0, 0) * A.transpose() + B * _covNoise * B.transpose();
}

void ImuPreintegrator::jacobians(const Eigen::Matrix3d& Jr, Eigen::Matrix3d& temp, Eigen::Matrix3d& tempKey) {
    // Jacobians of bias between two consecutive frames.
    _betweenFrames.d_R_bg_ijm1 = _betweenFrames.d_R_bg_ij;
    _betweenFrames.d_v_ba_ijm1 = _betweenFrames.d_v_ba_ij;
    _betweenFrames.d_v_bg_ijm1 = _betweenFrames.d_v_bg_ij;
    _betweenFrames.d_p_ba_ijm1 = _betweenFrames.d_p_ba_ij;
    _betweenFrames.d_p_bg_ijm1 = _betweenFrames.d_p_bg_ij;
    // (temp = _delta_R_ijm1.matrix() * ab_hat) -> (temp = _delta_R_ijm1.matrix() * ab_hat * _d_R_bg_ijm1)
    temp = temp * _betweenFrames.d_R_bg_ijm1;
    // Update jacobians of R, v, p with respect to bias_i.
    _betweenFrames.d_R_bg_ij = -_betweenFrames.d_R_bg_ijm1 - Jr * _deltaT;
    _betweenFrames.d_v_ba_ij = -_betweenFrames.d_v_ba_ijm1 - _betweenFrames.delta_R_ijm1.matrix() * _deltaT;
    _betweenFrames.d_v_bg_ij = -_betweenFrames.d_v_bg_ijm1 - temp * _deltaT;
    _betweenFrames.d_p_ba_ij = _betweenFrames.d_p_ba_ijm1 + _betweenFrames.d_v_ba_ijm1 * _deltaT - _betweenFrames.delta_R_ijm1.matrix() * _deltaT2 / 2;
    _betweenFrames.d_p_bg_ij = _betweenFrames.d_p_bg_ijm1 + _betweenFrames.d_v_bg_ijm1 * _deltaT - temp * _deltaT2 / 2;

    // // Jacobians of bias between two keyframes.
    // _betweenKeyframes.d_R_bg_ijm1 = _betweenKeyframes.d_R_bg_ij;
    // _betweenKeyframes.d_v_ba_ijm1 = _betweenKeyframes.d_v_ba_ij;
    // _betweenKeyframes.d_v_bg_ijm1 = _betweenKeyframes.d_v_bg_ij;
    // _betweenKeyframes.d_p_ba_ijm1 = _betweenKeyframes.d_p_ba_ij;
    // _betweenKeyframes.d_p_bg_ijm1 = _betweenKeyframes.d_p_bg_ij;
    // // (temp = _delta_R_ijm1.matrix() * ab_hat) -> (temp = _delta_R_ijm1.matrix() * ab_hat * _d_R_bg_ijm1)
    // tempKey = tempKey * _betweenKeyframes.d_R_bg_ijm1;
    // // Update jacobians of R, v, p with respect to bias_i.
    // _betweenKeyframes.d_R_bg_ij = -_betweenKeyframes.d_R_bg_ijm1 - Jr * _deltaT;
    // _betweenKeyframes.d_v_ba_ij = -_betweenKeyframes.d_v_ba_ijm1 - _betweenKeyframes.delta_R_ijm1.matrix() * _deltaT;
    // _betweenKeyframes.d_v_bg_ij = -_betweenKeyframes.d_v_bg_ijm1 - tempKey * _deltaT;
    // _betweenKeyframes.d_p_ba_ij = _betweenKeyframes.d_p_ba_ijm1 + _betweenKeyframes.d_v_ba_ijm1 * _deltaT - _betweenKeyframes.delta_R_ijm1.matrix() * _deltaT2 / 2;
    // _betweenKeyframes.d_p_bg_ij = _betweenKeyframes.d_p_bg_ijm1 + _betweenKeyframes.d_v_bg_ijm1 * _deltaT - tempKey * _deltaT2 / 2;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./preintegrate [number of iteration]" << std::endl;
        return -1;
    }
    
    int iter = std::atoi(argv[1]);

    ImuPreintegrator ip(false);

    Eigen::Vector3d gyr, acc;
    gyr << 0.001, 0.002, 0.003;
    acc << 0.3, 0.01, -9.8;

    double sum_t = 0;
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; i++) {
        start = std::chrono::steady_clock::now();
        ip.process(0, gyr, acc);
        end = std::chrono::steady_clock::now();
        // std::cout << "One process elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
        sum_t += std::chrono::duration<double, std::milli>(end-start).count();    
    }
    std::cout << std::endl << "Average elapsed time (" << iter << " iters): " << sum_t / (double)iter << "ms, FPS: " << (double)iter / sum_t * 1000 << std::endl << std::endl;

    return 0;
}