#ifndef COST_FUNCTIONS_HPP
#define COST_FUNCTIONS_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/camera-model.hpp"
#include "cfsd/map.hpp"
#include "cfsd/feature-tracker.hpp"
#include "cfsd/imu-preintegrator.hpp"

#include <ceres/ceres.h>

namespace cfsd {

struct ImageCostFunction : public ceres::SizedCostFunction<2, /* residuals */
                                                           6  /* pose (r, p) at time j */> {
                                                           // 3  /* landmark position */> {
    ImageCostFunction(const cfsd::Ptr<CameraModel>& pCameraModel, const Eigen::Vector3d& point, const cv::Point2d& pixel) : _pCameraModel(pCameraModel), _point(point), _pixel(pixel) {}

    virtual ~ImageCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: pose_j
        // pose: [rx,ry,rz, px,py,pz]
        Eigen::Vector3d r_j(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d p_j(parameters[0][3], parameters[0][4], parameters[0][5]);
        // T_WB = (r_j, p_j), is transformation from body to world frame.
        // T_BW = T_WB.inverse()
        Sophus::SO3d R_BW = Sophus::SO3d::exp(r_j);
        Sophus::SE3d T_BW = Sophus::SE3d(R_BW, p_j).inverse();

        // 3D landmark homogeneous coordinates w.r.t camera frame.
        Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * T_BW * _point;
        double x1 = point_wrt_cam(0);
        double y1 = point_wrt_cam(1);
        double z1 = point_wrt_cam(2);
        Eigen::Matrix<double,4,1> point_homo;
        point_homo << x1, y1, z1, 1;

        Eigen::Vector3d pixel_homo;
        pixel_homo = _pCameraModel->_P_L * point_homo;
        residuals[0] = pixel_homo(0)/pixel_homo(2) - _pixel.x;
        residuals[1] = pixel_homo(1)/pixel_homo(2) - _pixel.y;
        
        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        double fx = _pCameraModel->_P_L(0,0);
        double fy = _pCameraModel->_P_L(1,1);
        Eigen::Matrix<double,2,3> d1;
        d1 << fx / z1, 0, -fx * x1 / (z1*z1), 0, fy / z1, -fy * y1 / (z1*z1);
        
        // Jacobian(2x6) of residual(2x1) w.r.t. ParameterBlock[0](6x1), i.e. pose_j
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6>> jacobian_pose(jacobians[0]);

            Eigen::Matrix<double,3,6> d2;
            d2.block<3,3>(0,0) = -Sophus::SO3d::hat(R_BW * (_point - p_j));
            d2.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            jacobian_pose = d1 * _pCameraModel->_T_BC.so3().matrix() * d2;
        }

        // // Jacobian(2x3) of residual(2x1) w.r.t. ParameterBlock[1](3x1), i.e. landmark
        // if (jacobians[1]) {
        //     Eigen::Map<Eigen::Matrix<double, 2, 3>> jacobian_landmark(jacobians[1]);

        //     jacobian_landmark = d1 * _pCameraModel->_T_BC.so3().matrix() * R_BW.matrix();
        // }
        return true;
    }

  private:
    cfsd::Ptr<CameraModel> _pCameraModel;
    Eigen::Vector3d _point; // Landmark w.r.t world frame.
    cv::Point2d _pixel; // Pixel coordinates.
};


struct ImuCostFunction : public ceres::SizedCostFunction<15, /* residuals */
                                                          6, /* increment of pose (r, p) at time i */
                                                          9, /* increment of velocity, delta_bg and delta_ba at time i */
                                                          6, /* increment of pose (r, p) at time j */
                                                          9  /* increment of velocity, delta_bg and delta_ba at time j */> {
    ImuCostFunction(const cfsd::Ptr<Map>& pMap, const int& idx) : _pMap(pMap), _idx(idx) {}

    virtual ~ImuCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_pose_i, delta_v_dbga_i, delta_pose_j, delta_v_dbga_j
        // pose: [rx,ry,rz, px,py,pz]

        // Rotation.
        Eigen::Vector3d delta_r_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d delta_r_j(parameters[2][0], parameters[2][1], parameters[2][2]);

        // Position.
        Eigen::Vector3d delta_p_i(parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d delta_p_j(parameters[2][3], parameters[2][4], parameters[2][5]);

        // Velocity.
        Eigen::Vector3d delta_v_i(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d delta_v_j(parameters[3][0], parameters[3][1], parameters[3][2]);

        // Gyroscope bias.
        Eigen::Vector3d delta_dbg_i(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d delta_dbg_j(parameters[3][3], parameters[3][4], parameters[3][5]);

        // Accelerometer bias.
        Eigen::Vector3d delta_dba_i(parameters[1][6], parameters[1][7], parameters[1][8]);
        Eigen::Vector3d delta_dba_j(parameters[3][6], parameters[3][7], parameters[3][8]);

        cfsd::Ptr<ImuConstraint> ic = _pMap->_imuConstraint[_pMap->_imuConstraint.size() - _idx];

        // Map double* to Eigen Matrix.
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

        int index_i = _pMap->_dbg.size() - _idx - 1;
        int index_j = index_i + 1;

        // Bias estimation update at time i, i.e. bias changes by a small amount delta_b during optimization.
        Eigen::Vector3d updated_delta_bg_i = _pMap->_dbg[index_i] + delta_dbg_i;
        Eigen::Vector3d updated_delta_ba_i = _pMap->_dba[index_i] + delta_dba_i;

        // residual(delta_R_ij)
        Sophus::SO3d R_i = _pMap->_R[index_i];
        Sophus::SO3d R_j = _pMap->_R[index_j];
        Sophus::SO3d updated_R_i = R_i * Sophus::SO3d::exp(delta_r_i);
        Sophus::SO3d updated_R_j = R_j * Sophus::SO3d::exp(delta_r_j);
        Sophus::SO3d tempR = ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * updated_delta_bg_i);
        residual.block<3, 1>(0, 0) = (tempR.inverse() * updated_R_i.inverse() * updated_R_j).log();

        // residual(delta_v_ij)
        Eigen::Vector3d updated_v_i = _pMap->_v[index_i] + delta_v_i;
        Eigen::Vector3d updated_v_j = _pMap->_v[index_j] + delta_v_j;
        Eigen::Vector3d updated_dv = updated_v_j - updated_v_i - _pMap->_gravity * ic->dt;
        residual.block<3, 1>(3, 0) = updated_R_i.inverse() * updated_dv - (ic->delta_v_ij + ic->d_v_bg_ij * updated_delta_bg_i + ic->d_v_ba_ij * updated_delta_ba_i);

        // residual(delta_p_ij)
        Eigen::Vector3d updated_p_i = _pMap->_p[index_i] + updated_R_i * delta_p_i;
        Eigen::Vector3d updated_p_j = _pMap->_p[index_j] + updated_R_j * delta_p_j;
        Eigen::Vector3d updated_dp = updated_p_j - updated_p_i - updated_v_i * ic->dt - _pMap->_gravity * ic->dt2 / 2;
        residual.block<3, 1>(6, 0) = updated_R_i.inverse() * updated_dp - (ic->delta_p_ij + ic->d_p_bg_ij * updated_delta_bg_i + ic->d_p_ba_ij * updated_delta_ba_i);

        // residual(delta_bg_ij)
        // Eigen::Vector3d bg_i = ic->bg_i + updated_delta_bg_i;
        // Eigen::Vector3d bg_j = ic->bg_i + _pMap->_dbg[index_j] + delta_dbg_j;
        // residual.block<3, 1>(9, 0) = bg_j - bg_i;
        residual.block<3, 1>(9, 0) = _pMap->_dbg[index_j] + delta_dbg_j - updated_delta_bg_i;

        // residual(delta_ba_ij)
        // Eigen::Vector3d ba_i = ic->ba_i + updated_delta_ba_i;
        // Eigen::Vector3d ba_j = ic->ba_i + _pMap->_dba[index_j] + delta_dba_j;
        // residual.block<3, 1>(12, 0) = ba_j - ba_i;
        residual.block<3, 1>(12, 0) = _pMap->_dba[index_j] + delta_dba_j - updated_delta_ba_i;

        // |r|^2 is defined as: r' * inv(cov) * r
        // Whereas in ceres, the square of residual is defined as: |x|^2 = x' * x
        // so we should construct such x from r in order to fit ceres solver.
        
        // Use cholesky decomposition: inv(cov) = L * L'
        // |r|^2 = r' * L * L' * r = (L' * r)' * (L' * r)
        // define x = L' * r (matrix 15x1)
        Eigen::Matrix<double, 15, 15> Lt = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(ic->invCovPreintegration_ij).matrixL().transpose();
        residual = Lt * residual;

        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        // Inverse of right jacobian of residual (delta_R_ij) calculated without adding delta increment.
        Eigen::Vector3d residual_R = ((ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * _pMap->_dbg[index_i])).inverse() * R_i.inverse() * R_j).log();
        Eigen::Matrix3d JrInv = rightJacobianInverseSO3(residual_R);

        // Jacobian(15x6) of residual(15x1) w.r.t. ParameterBlock[0](6x1), i.e. delta_pose_i
        if (jacobians[0]) {
            // The default storage order is ColMajor in Eigen.
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_rp_i(jacobians[0]);

            jacobian_rp_i.setZero();

            // jacobian of residual(delta_R_ij) with respect to delta_r_i
            jacobian_rp_i.block<3, 3>(0, 0) = -JrInv * R_j.matrix().transpose() * R_i.matrix();

            // jacobian of residual(delta_v_ij) with respect to delta_r_i
            Eigen::Vector3d dv = _pMap->_v[index_j] - _pMap->_v[index_i] - _pMap->_gravity * ic->dt;
            jacobian_rp_i.block<3, 3>(3, 0) = Sophus::SO3d::hat(R_i.inverse() * dv);

            // jacobian of residual(delta_p_ij) with respect to delta_r_i
            Eigen::Vector3d dp = _pMap->_p[index_j] - _pMap->_p[index_i] - _pMap->_v[index_i] * ic->dt - _pMap->_gravity * ic->dt2 / 2;
            jacobian_rp_i.block<3, 3>(6, 0) = Sophus::SO3d::hat(R_i.inverse() * dp);

            // jacobian of residual(delta_p_ij) with respect to delta_p_i
            jacobian_rp_i.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();

            // since cost function is defined as: L' * r
            jacobian_rp_i = Lt * jacobian_rp_i;

            if (jacobian_rp_i.maxCoeff() > 1e8 || jacobian_rp_i.minCoeff() < -1e8)
                std::cout << "Numerical unstable in calculating jacobian_i!" << std::endl;
        }

        // Jacobian(15x9) of residuals(15x1) w.r.t. ParameterBlock[1](9x1), i.e. delta_v_dbga_i
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_vb_i(jacobians[1]);

            jacobian_vb_i.setZero();

            // jacobian of residual(delta_R_ij) with respect to delta_dbg_i
            jacobian_vb_i.block<3, 3>(0, 3) = -JrInv * Sophus::SO3d::exp(residual_R).matrix().transpose() * rightJacobianSO3(ic->d_R_bg_ij * _pMap->_dbg[index_i]) * ic->d_R_bg_ij;
            
            // jacobian of residual(delta_v_ij) with respect to delta_v_i
            jacobian_vb_i.block<3, 3>(3, 0) = -R_i.matrix().transpose();

            // jacobian of residual(delta_v_ij) with respect to delta_dbg_i
            jacobian_vb_i.block<3, 3>(3, 3) = -ic->d_v_bg_ij;

            // jacobian of residual(delta_v_ij) with respect to delta_dba_i
            jacobian_vb_i.block<3, 3>(3, 6) = -ic->d_v_ba_ij;

            // jacobian of residual(delta_p_ij) with respect to delta_v_i
            jacobian_vb_i.block<3, 3>(6, 0) = -R_i.matrix().transpose() * ic->dt;

            // jacobian of residual(delta_p_ij) with respect to delta_dbg_i
            jacobian_vb_i.block<3, 3>(6, 3) = -ic->d_p_bg_ij;

            // jacobian of residual(delta_p_ij) with respect to delta_dba_i
            jacobian_vb_i.block<3, 3>(6, 6) = -ic->d_p_ba_ij;

            // jacobian of residual(delta_bg_ij) with respect to delta_dbg_i
            jacobian_vb_i.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();

            // jacobian of residual(delta_ba_ij) with respect to delta_dba_i
            jacobian_vb_i.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();

            // since cost function is defined as: L' * r
            jacobian_vb_i = Lt * jacobian_vb_i;

            if (jacobian_vb_i.maxCoeff() > 1e8 || jacobian_vb_i.minCoeff() < -1e8)
                std::cout << "Numerical unstable in calculating jacobian_bi!" << std::endl;
        }

        // Jacobian(15x6) of residuals(15x1) w.r.t. ParameterBlock[2](6x1), i.e. delta_pose_j
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_rp_j(jacobians[2]);

            jacobian_rp_j.setZero();

            // jacobian of residual(delta_R_ij) with respect to delta_r_j
            jacobian_rp_j.block<3, 3>(0, 0) = JrInv;

            // jacobian of residual(delta_p_ij) with respect to delta_p_j
            jacobian_rp_j.block<3, 3>(6, 3) = R_i.matrix().transpose() * R_j.matrix();

            // since cost function is defined as: L' * r
            jacobian_rp_j = Lt * jacobian_rp_j;

            if (jacobian_rp_j.maxCoeff() > 1e8 || jacobian_rp_j.minCoeff() < -1e8)
                std::cout << "Numerical unstable in calculating jacobian_j!" << std::endl;
        }

        // Jacobian(15x9) of residuals(15x1) w.r.t. ParameterBlock[3](9x1), i.e. delta_v_dbga_j
        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_vb_j(jacobians[3]);

            jacobian_vb_j.setZero();

            // jacobian of residual(delta_v_ij) with respect to delta_v_j
            jacobian_vb_j.block<3, 3>(3, 0) = R_i.matrix().transpose();

            // jacobian of residual(delta_bg_ij) with respect to delta_dbg_j
            jacobian_vb_j.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();

            // jacobian of residual(delta_ba_ij) with respect to delta_dba_j
            jacobian_vb_j.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

            // since cost function is defined as: L' * r
            jacobian_vb_j = Lt * jacobian_vb_j;

            if (jacobian_vb_j.maxCoeff() > 1e8 || jacobian_vb_j.minCoeff() < -1e8)
                std::cout << "Numerical unstable in calculating jacobian_bj!" << std::endl;
        }

        return true;
    }

  private:
    cfsd::Ptr<Map> _pMap;
    int _idx;
};


struct BiasGyrCostFunction : public ceres::SizedCostFunction<3, /* residuals of delta_R_ij */
                                                             3  /* delta_dbg_i */> {
    BiasGyrCostFunction(const cfsd::Ptr<ImuConstraint>& ic, const Eigen::Vector3d& delta_bg) : _ic(ic), _delta_bg_i(delta_bg) {}

    virtual ~BiasGyrCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_dbg_i
        Eigen::Vector3d delta_dbg_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        
        Eigen::Map<Eigen::Vector3d> residual(residuals);

        residual = (_ic->delta_R_ij * Sophus::SO3d::exp(_ic->d_R_bg_ij * (_delta_bg_i + delta_dbg_i))).inverse().log();

        Eigen::Matrix3d Lt = Eigen::LLT<Eigen::Matrix3d>(_ic->invCovPreintegration_ij.block<3,3>(0,0)).matrixL().transpose();

        residual = Lt * residual;

        if (!jacobians) return true;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_dbg(jacobians[0]);

            Eigen::Vector3d residual_R = (_ic->delta_R_ij * Sophus::SO3d::exp(_ic->d_R_bg_ij * _delta_bg_i)).inverse().log();
            
            jacobian_dbg = -rightJacobianInverseSO3(residual_R) * Sophus::SO3d::exp(residual_R).matrix().transpose() * rightJacobianSO3(_ic->d_R_bg_ij * _delta_bg_i) * _ic->d_R_bg_ij;

            jacobian_dbg = Lt * jacobian_dbg;
        }
        return true;
    }

  private:
    cfsd::Ptr<ImuConstraint> _ic;
    Eigen::Vector3d _delta_bg_i;
};


struct AlignmentCostFunction : public ceres::SizedCostFunction<6, /* residuals of delta_v_ij, delta_p_ij */
                                                               3, /* delta_r_i */
                                                               3  /* delta_dbg_i */> {
    AlignmentCostFunction(const cfsd::Ptr<ImuConstraint>& ic, const Eigen::Vector3d& delta_bg, const Eigen::Vector3d& delta_ba, const Eigen::Vector3d& g) : _ic(ic), _delta_bg_i(delta_bg), _delta_ba_i(delta_ba), _gravity(g) {}

    virtual ~AlignmentCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_r_i, delta_dba_i
        Eigen::Vector3d delta_r_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d delta_dba_i(parameters[1][0], parameters[1][1], parameters[1][2]);
        
        Eigen::Map<Eigen::Matrix<double,6,1>> residual(residuals);

        // Assume R_i is an identity matrix.
        Sophus::SO3d updated_R_i = Sophus::SO3d::exp(delta_r_i);
        Eigen::Vector3d updated_delta_ba_i = _delta_ba_i + delta_dba_i;

        // residual(delta_v_ij)
        residual.block<3, 1>(0, 0) = updated_R_i.inverse() * (-_gravity * _ic->dt) - (_ic->delta_v_ij + _ic->d_v_bg_ij * _delta_bg_i + _ic->d_v_ba_ij * updated_delta_ba_i);

        // residual(delta_p_ij)
        residual.block<3, 1>(3, 0) = updated_R_i.inverse() * (-_gravity * _ic->dt2 / 2) - (_ic->delta_p_ij + _ic->d_p_bg_ij * _delta_bg_i + _ic->d_p_ba_ij * updated_delta_ba_i);

        Eigen::Matrix<double,6,6> Lt = Eigen::LLT<Eigen::Matrix<double,6,6>>(_ic->invCovPreintegration_ij.block<6,6>(3,3)).matrixL().transpose();

        residual = Lt * residual;

        if (!jacobians) return true;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian_r(jacobians[0]);

            jacobian_r.block<3,3>(0,0) = Sophus::SO3d::hat(-_gravity * _ic->dt);

            jacobian_r.block<3,3>(3,0) = Sophus::SO3d::hat(-_gravity * _ic->dt2 / 2);

            jacobian_r = Lt * jacobian_r;
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian_dba(jacobians[1]);

            jacobian_dba.block<3,3>(0,0) = -_ic->d_v_ba_ij;

            jacobian_dba.block<3,3>(3,0) = -_ic->d_p_ba_ij;

            jacobian_dba = Lt * jacobian_dba;
        }

        return true;
    }

  private:
    cfsd::Ptr<ImuConstraint> _ic;
    Eigen::Vector3d _delta_bg_i;
    Eigen::Vector3d _delta_ba_i;
    Eigen::Vector3d _gravity;
};


// struct AccCostFunctor {
//     template<typename T>
//     bool operator() (const T* const rvec, const T* const ba, T* residuals) const {
//         Eigen::Matrix<T,3,1> _rvec(rvec[0], rvec[1], rvec[2]);
//         Eigen::Matrix<T,3,1> _ba(ba[0], ba[1], ba[2]);
//         Sophus::SO3<T> R = Sophus::SO3<T>::exp(_rvec);
//         Eigen::Matrix<T,3,1> correctedAcc = R * (_acc - _ba);

//         Eigen::Map<Eigen::Matrix<T,3,1>> residual(residuals);
//         residual = (R * (_acc - _ba) + _g);

//         return true;
//     }

//     AccCostFunctor(const Eigen::Vector3d& acc, const Eigen::Vector3d& gravity) : _acc(acc), _g(gravity) {}

//     Eigen::Vector3d _acc;
//     Eigen::Vector3d _g;
// };


// struct GyrCostFunctor {
//     template<typename T>
//     bool operator() (const T* const bg, T* residuals) const {
//         Eigen::Matrix<T,3,1> _bg(bg[0], bg[1], bg[2]);
//         Eigen::Map<Eigen::Matrix<T,3,1>> residual(residuals);
//         residual = _gyr - _bg;
//         return true;
//     }

//     GyrCostFunctor(const Eigen::Vector3d& gyr) : _gyr(gyr) {}

//     Eigen::Vector3d _gyr;
// };


} // namespace cfsd

#endif // COST_FUNCTIONS_HPP