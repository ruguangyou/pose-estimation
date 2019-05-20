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

struct FullBAFunction : public ceres::CostFunction {
    FullBAFunction(const int& n, const Eigen::MatrixXd& error, const Eigen::MatrixXd& F, const Eigen::MatrixXd& E)
        : _numResiduals(error.size()), _numParameterBlocks(n), _error(error), _F(F), _E(E) {

        set_num_residuals(_numResiduals);
        
        std::vector<int32_t> prameterBlockSizes;
        for (int i = 0; i < n-1; i++)
            prameterBlockSizes.push_back(6);
        prameterBlockSizes.push_back(3);
        *mutable_parameter_block_sizes() = prameterBlockSizes;
    }

    virtual ~FullBAFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_pose (delta_r, delta_p)
        Eigen::VectorXd delta_pose(6*(_numParameterBlocks-1));
        int i;
        for (i = 0; i < _numParameterBlocks-1; i++) {
            // delta_r
            delta_pose.segment<3>(6*i) = Eigen::Vector3d(parameters[i][0], parameters[i][1], parameters[i][2]);
            // delta_p
            delta_pose.segment<3>(6*i+3) = Eigen::Vector3d(parameters[i][3], parameters[i][4], parameters[i][5]);
        }
        Eigen::Vector3d delta_x(parameters[i][0], parameters[i][1], parameters[i][2]);

        Eigen::Map<Eigen::VectorXd> residual(residuals, _numResiduals);

        residual = _error + _F * delta_pose + _E * delta_x;;

        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        for (i = 0; i < _numParameterBlocks-1; i++) {
            if (jacobians[i]) {
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> jacobian_pose(jacobians[i], _numResiduals, 6);
                
                jacobian_pose = _F.block(0,6*i, _numResiduals,6);
            }
        }
        if (jacobians[i]) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> jacobian_pose(jacobians[i], _numResiduals, 3);
            
            jacobian_pose = _E;
        }

        return true;
    }

  private:
    int _numResiduals{0};
    int _numParameterBlocks{0};
    Eigen::MatrixXd _error;
    Eigen::MatrixXd _F;
    Eigen::MatrixXd _E;
};

struct ImageCostFunction : public ceres::CostFunction {
    ImageCostFunction(const int& n, const Eigen::MatrixXd& error, const Eigen::MatrixXd& F)
        : _numResiduals(error.size()), _numParameterBlocks(n), _error(error), _F(F){

        set_num_residuals(_numResiduals);
        
        std::vector<int32_t> prameterBlockSizes;
        for (int i = 0; i < n; i++)
            prameterBlockSizes.push_back(6);
        *mutable_parameter_block_sizes() = prameterBlockSizes;
    }

    virtual ~ImageCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_pose (delta_r, delta_p)
        Eigen::VectorXd delta(6*_numParameterBlocks);
        for (int i = 0; i < _numParameterBlocks; i++) {
            // delta_r
            delta.segment<3>(6*i) = Eigen::Vector3d(parameters[i][0], parameters[i][1], parameters[i][2]);
            // delta_p
            delta.segment<3>(6*i+3) = Eigen::Vector3d(parameters[i][3], parameters[i][4], parameters[i][5]);
        }

        Eigen::Map<Eigen::VectorXd> residual(residuals, _numResiduals);

        residual = _error + _F * delta;

        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        for (int i = 0; i < _numParameterBlocks; i++) {
            if (jacobians[i]) {
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> jacobian_pose(jacobians[i], _numResiduals, 6);

                jacobian_pose = _F.block(0,6*i, _numResiduals,6);
            }
        }

        return true;
    }

  private:
    int _numResiduals{0};
    int _numParameterBlocks{0};
    Eigen::MatrixXd _error;
    Eigen::MatrixXd _F;
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

        cfsd::Ptr<Keyframe>& frame_i = _pMap->_pKeyframes[_idx];
        cfsd::Ptr<Keyframe>& frame_j = _pMap->_pKeyframes[_idx+1];

        // Map double* to Eigen Matrix.
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

        cfsd::Ptr<ImuConstraint>& ic = frame_j->pImuConstraint;

        // Bias estimation update at time i, i.e. bias changes by a small amount delta_b during optimization.
        Eigen::Vector3d updated_delta_bg_i = frame_i->dbg + delta_dbg_i;
        Eigen::Vector3d updated_delta_ba_i = frame_i->dba + delta_dba_i;

        // residual(delta_R_ij)
        Sophus::SO3d& R_i = frame_i->R;
        Sophus::SO3d& R_j = frame_j->R;
        Sophus::SO3d updated_R_i = R_i * Sophus::SO3d::exp(delta_r_i);
        Sophus::SO3d updated_R_j = R_j * Sophus::SO3d::exp(delta_r_j);
        Sophus::SO3d tempR = ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * updated_delta_bg_i);
        residual.block<3, 1>(0, 0) = (tempR.inverse() * updated_R_i.inverse() * updated_R_j).log();

        // residual(delta_v_ij)
        Eigen::Vector3d updated_v_i = frame_i->v + delta_v_i;
        Eigen::Vector3d updated_v_j = frame_j->v + delta_v_j;
        Eigen::Vector3d updated_dv = updated_v_j - updated_v_i - _pMap->_gravity * ic->dt;
        residual.block<3, 1>(3, 0) = updated_R_i.inverse() * updated_dv - (ic->delta_v_ij + ic->d_v_bg_ij * updated_delta_bg_i + ic->d_v_ba_ij * updated_delta_ba_i);

        // residual(delta_p_ij)
        Eigen::Vector3d updated_p_i = frame_i->p + R_i * delta_p_i;
        Eigen::Vector3d updated_p_j = frame_j->p + R_j * delta_p_j;
        Eigen::Vector3d updated_dp = updated_p_j - updated_p_i - updated_v_i * ic->dt - _pMap->_gravity * ic->dt2 / 2;
        residual.block<3, 1>(6, 0) = updated_R_i.inverse() * updated_dp - (ic->delta_p_ij + ic->d_p_bg_ij * updated_delta_bg_i + ic->d_p_ba_ij * updated_delta_ba_i);

        // residual(delta_bg_ij)
        // residual.block<3, 1>(9, 0) = bg_j - bg_i;
        residual.block<3, 1>(9, 0) = frame_j->dbg + delta_dbg_j - updated_delta_bg_i;

        // residual(delta_ba_ij)
        // residual.block<3, 1>(12, 0) = ba_j - ba_i;
        residual.block<3, 1>(12, 0) = frame_j->dba + delta_dba_j - updated_delta_ba_i;

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
        Eigen::Vector3d residual_R = ((ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * frame_i->dbg)).inverse() * R_i.inverse() * R_j).log();
        Eigen::Matrix3d JrInv = rightJacobianInverseSO3(residual_R);

        // Jacobian(15x6) of residual(15x1) w.r.t. ParameterBlock[0](6x1), i.e. delta_pose_i
        if (jacobians[0]) {
            // The default storage order is ColMajor in Eigen.
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_rp_i(jacobians[0]);

            jacobian_rp_i.setZero();

            // jacobian of residual(delta_R_ij) with respect to delta_r_i
            jacobian_rp_i.block<3, 3>(0, 0) = -JrInv * R_j.matrix().transpose() * R_i.matrix();

            // jacobian of residual(delta_v_ij) with respect to delta_r_i
            Eigen::Vector3d dv = frame_j->v - frame_i->v - _pMap->_gravity * ic->dt;
            jacobian_rp_i.block<3, 3>(3, 0) = Sophus::SO3d::hat(R_i.inverse() * dv);

            // jacobian of residual(delta_p_ij) with respect to delta_r_i
            Eigen::Vector3d dp = frame_j->p - frame_i->p - frame_i->v * ic->dt - _pMap->_gravity * ic->dt2 / 2;
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
            jacobian_vb_i.block<3, 3>(0, 3) = -JrInv * Sophus::SO3d::exp(residual_R).matrix().transpose() * rightJacobianSO3(ic->d_R_bg_ij * frame_i->dbg) * ic->d_R_bg_ij;
            
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
    const cfsd::Ptr<Map>& _pMap;
    int _idx;
};


struct BiasGyrCostFunction : public ceres::SizedCostFunction<3, /* residuals of delta_R_ij */
                                                             3  /* delta_dbg */> {
    BiasGyrCostFunction(const cfsd::Ptr<ImuConstraint>& ic, const Sophus::SO3d& R_i, const Sophus::SO3d& R_j) : _ic(ic), _R_i(R_i), _R_j(R_j) {}

    virtual ~BiasGyrCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_dbg
        Eigen::Vector3d delta_dbg_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        
        Eigen::Map<Eigen::Vector3d> residual(residuals);

        residual = ((_ic->delta_R_ij * Sophus::SO3d::exp(_ic->d_R_bg_ij * delta_dbg_i)).inverse() * _R_i.inverse() * _R_j).log();

        Eigen::Matrix3d Lt = Eigen::LLT<Eigen::Matrix3d>(_ic->invCovPreintegration_ij.block<3,3>(0,0)).matrixL().transpose();

        residual = Lt * residual;

        if (!jacobians) return true;

        Eigen::Vector3d residual_R = (_ic->delta_R_ij.inverse() * _R_i.inverse() * _R_j).log();

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_dbg(jacobians[0]);
            
            jacobian_dbg = -rightJacobianInverseSO3(residual_R) * Sophus::SO3d::exp(residual_R).matrix().transpose() * _ic->d_R_bg_ij;

            jacobian_dbg = Lt * jacobian_dbg;
        }

        return true;
    }

  private:
    const cfsd::Ptr<ImuConstraint>& _ic;
    const Sophus::SO3d& _R_i;
    const Sophus::SO3d& _R_j;
};


struct GravityVelocityCostFunction : public ceres::SizedCostFunction<6, /* residuals of delta_v_ij, delta_p_ij */
                                                                     3, /* delta gravity */
                                                                     3, /* delta_v_i */
                                                                     3  /* delta_v_j */> {
    GravityVelocityCostFunction(const cfsd::Ptr<ImuConstraint>& ic, const Sophus::SO3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& p_j) 
        : _ic(ic), _R_i(R_i), _p_i(p_i), _p_j(p_j) {}

    virtual ~GravityVelocityCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_g, delta_v_i, delta_v_j
        Eigen::Vector3d delta_g(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d delta_v_i(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d delta_v_j(parameters[2][0], parameters[2][1], parameters[2][2]);
        // Assume initial gravity, v_i, v_j value is zero.
        
        Eigen::Map<Eigen::Matrix<double,6,1>> residual(residuals);

        // residual(delta_v_ij)
        residual.block<3, 1>(0, 0) = _R_i.inverse() * (delta_v_j - delta_v_i - delta_g * _ic->dt) - _ic->delta_v_ij; // delta_v_ij has already been propagated

        // residual(delta_p_ij)
        residual.block<3, 1>(3, 0) = _R_i.inverse() * (_p_j - _p_i - delta_v_i * _ic->dt - delta_g * _ic->dt2 / 2) - _ic->delta_p_ij;

        Eigen::Matrix<double,6,6> Lt = Eigen::LLT<Eigen::Matrix<double,6,6>>(_ic->invCovPreintegration_ij.block<6,6>(3,3)).matrixL().transpose();

        residual = Lt * residual;

        if (!jacobians) return true;

        Eigen::Matrix3d R_temp = -_R_i.inverse().matrix();

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> jacobian_g(jacobians[0]);

            // delta_v_ij w.r.t delta_g
            jacobian_g.block<3,3>(0,0) = R_temp * _ic->dt;

            // delta_p_ij w.r.t delta_g
            jacobian_g.block<3,3>(3,0) = R_temp * _ic->dt2 / 2;

            jacobian_g = Lt * jacobian_g;
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> jacobian_v_i(jacobians[1]);

            // delta_v_ij w.r.t delta_v_i
            jacobian_v_i.block<3,3>(0,0) = R_temp;

            // delta_p_ij w.r.t delta_v_i
            jacobian_v_i.block<3,3>(3,0) = R_temp * _ic->dt;

            jacobian_v_i = Lt * jacobian_v_i;
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> jacobian_v_j(jacobians[2]);

            // delta_v_ij w.r.t delta_v_j
            jacobian_v_j.block<3,3>(0,0) = -R_temp;

            // delta_p_ij w.r.t delta_v_j
            jacobian_v_j.block<3,3>(3,0) = Eigen::Matrix3d::Zero();

            jacobian_v_j = Lt * jacobian_v_j;
        }

        return true;
    }

  private:
    const cfsd::Ptr<ImuConstraint>& _ic;
    const Sophus::SO3d& _R_i;
    const Eigen::Vector3d& _p_i;
    const Eigen::Vector3d& _p_j;
};


struct AlignmentCostFunction : public ceres::SizedCostFunction<3, /* residuals of gravity refinement */
                                                               2  /* delta_r around non-gravitational axis */> {
    AlignmentCostFunction(const Eigen::Vector3d& init_g, const Eigen::Vector3d& unit_g) : _init_g(init_g), _unit_g(unit_g) {}

    virtual ~AlignmentCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        /*  cfsd imu coordinate system
                      / x
                     /
                    ------ y
                    |
                    | z

            euroc imu coordinate system
                  x |  / z
                    | /
                    ------ y
        
            kitti imu coordinate system
                  z |  / x
                    | /
              y -----
        */

        Eigen::Vector3d delta_r;

        #ifdef CFSD
        delta_r << parameters[0][0], parameters[0][1], 0.0;
        #endif
        
        #ifdef EUROC
        delta_r << 0.0, parameters[0][0], parameters[0][1];
        #endif

        #ifdef KITTI
        delta_r << parameters[0][0], parameters[0][1], 0.0;
        #endif
        
        Eigen::Map<Eigen::Matrix<double,3,1>> residual(residuals);

        residual.block<3, 1>(0, 0) = _unit_g - Sophus::SO3d::exp(delta_r) * _init_g;

        if (!jacobians) return true;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double,3,2,Eigen::RowMajor>> jacobian_r(jacobians[0]);

            #ifdef CFSD
            jacobian_r = Sophus::SO3d::hat(_init_g).leftCols<2>();
            #endif
            
            #ifdef EUROC
            jacobian_r = Sophus::SO3d::hat(_init_g).rightCols<2>();
            #endif

            #ifdef KITTI
            jacobian_r = Sophus::SO3d::hat(_init_g).leftCols<2>();
            #endif
        }

        return true;
    }

  private:
    const Eigen::Vector3d& _init_g;
    const Eigen::Vector3d& _unit_g;
};


struct AccCostFunction : public ceres::SizedCostFunction<6, /* residuals of delta_v_ij, delta_p_ij */
                                                         3  /* delta_dba_i */> {
    AccCostFunction(const cfsd::Ptr<ImuConstraint>& ic, const Sophus::SO3d& R_i, const Eigen::Vector3d& v_i, const Eigen::Vector3d& v_j, const Eigen::Vector3d& p_i, const Eigen::Vector3d& p_j, const Eigen::Vector3d& g)
        : _ic(ic), _R_i(R_i), _v_i(v_i), _v_j(v_j), _p_i(p_i), _p_j(p_j), _gravity(g) {}

    virtual ~AccCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: delta_dba_i
        Eigen::Vector3d delta_dba_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        
        Eigen::Map<Eigen::Matrix<double,6,1>> residual(residuals);

        // residual(delta_v_ij)
        residual.block<3, 1>(0, 0) = _R_i.inverse() * (_v_j - _v_i -_gravity * _ic->dt) - (_ic->delta_v_ij + _ic->d_v_ba_ij * delta_dba_i);

        // residual(delta_p_ij)
        residual.block<3, 1>(3, 0) = _R_i.inverse() * (_p_j - _p_i - _v_i * _ic->dt - _gravity * _ic->dt2 / 2) - (_ic->delta_p_ij + _ic->d_p_ba_ij * delta_dba_i);

        Eigen::Matrix<double,6,6> Lt = Eigen::LLT<Eigen::Matrix<double,6,6>>(_ic->invCovPreintegration_ij.block<6,6>(3,3)).matrixL().transpose();

        residual = Lt * residual;

        if (!jacobians) return true;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian_dba_i(jacobians[0]);

            jacobian_dba_i.setZero();

            // delta_v_ij w.r.t delta_dba_i
            jacobian_dba_i.block<3,3>(0,0) = -_ic->d_v_ba_ij;

            // delta_p_ij w.r.t delta_dba_i
            jacobian_dba_i.block<3,3>(3,0) = -_ic->d_p_ba_ij;

            jacobian_dba_i = Lt * jacobian_dba_i;
        }

        return true;
    }

  private:
    const cfsd::Ptr<ImuConstraint>& _ic;
    const Sophus::SO3d& _R_i;
    const Eigen::Vector3d& _v_i;
    const Eigen::Vector3d& _v_j;
    const Eigen::Vector3d& _p_i;
    const Eigen::Vector3d& _p_j;
    const Eigen::Vector3d& _gravity;
};

struct ImuCostFunction4DOF : public ceres::SizedCostFunction<9, /* residuals */
                                                             4, /* increment of yaw and position at time i */
                                                             4  /* increment of yaw and position at time j */> {
    ImuCostFunction4DOF(const cfsd::Ptr<Map>& pMap, const int& idx) : _pMap(pMap), _idx(idx) {}

    virtual ~ImuCostFunction4DOF() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        /*  cfsd imu coordinate system
                      / x
                     /
                    ------ y
                    |
                    | z

            euroc imu coordinate system
                  x |  / z
                    | /
                    ------ y
        
            kitti imu coordinate system
                  z |  / x
                    | /
              y -----
        */

        // Rotation.
        Eigen::Vector3d delta_r_i, delta_r_j;

        #ifdef CFSD
        delta_r_i << 0.0, 0.0, parameters[0][0];
        delta_r_j << 0.0, 0.0, parameters[1][0];
        #endif
        
        #ifdef EUROC
        delta_r_i << parameters[0][0], 0.0, 0.0;
        delta_r_j << parameters[1][0], 0.0, 0.0;
        #endif

        #ifdef KITTI
        delta_r_i << 0.0, 0.0, parameters[0][0];
        delta_r_j << 0.0, 0.0, parameters[1][0];
        #endif

        // Position.
        Eigen::Vector3d delta_p_i(parameters[0][1], parameters[0][2], parameters[0][3]);
        Eigen::Vector3d delta_p_j(parameters[1][1], parameters[1][2], parameters[1][3]);

        cfsd::Ptr<Keyframe>& frame_i = _pMap->_pKeyframes[_idx];
        cfsd::Ptr<Keyframe>& frame_j = _pMap->_pKeyframes[_idx+1];

        // Map double* to Eigen Matrix.
        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);

        cfsd::Ptr<ImuConstraint>& ic = frame_j->pImuConstraint;

        // residual(delta_R_ij)
        Sophus::SO3d& R_i = frame_i->R;
        Sophus::SO3d& R_j = frame_j->R;
        Sophus::SO3d updated_R_i = R_i * Sophus::SO3d::exp(delta_r_i);
        Sophus::SO3d updated_R_j = R_j * Sophus::SO3d::exp(delta_r_j);
        Sophus::SO3d tempR = ic->delta_R_ij * Sophus::SO3d::exp(ic->d_R_bg_ij * frame_i->dbg);
        residual.block<3, 1>(0, 0) = (tempR.inverse() * updated_R_i.inverse() * updated_R_j).log();

        // residual(delta_v_ij)
        residual.block<3, 1>(3, 0) = updated_R_i.inverse() * (frame_j->v - frame_i->v - _pMap->_gravity * ic->dt) - (ic->delta_v_ij + ic->d_v_bg_ij * frame_i->dbg + ic->d_v_ba_ij * frame_i->dba);

        // residual(delta_p_ij)
        Eigen::Vector3d updated_p_i = frame_i->p + R_i * delta_p_i;
        Eigen::Vector3d updated_p_j = frame_j->p + R_j * delta_p_j;
        Eigen::Vector3d updated_dp = updated_p_j - updated_p_i - frame_i->v * ic->dt - _pMap->_gravity * ic->dt2 / 2;
        residual.block<3, 1>(6, 0) = updated_R_i.inverse() * updated_dp - (ic->delta_p_ij + ic->d_p_bg_ij * frame_i->dbg + ic->d_p_ba_ij * frame_i->dba);

        // |r|^2 is defined as: r' * inv(cov) * r
        // Whereas in ceres, the square of residual is defined as: |x|^2 = x' * x
        // so we should construct such x from r in order to fit ceres solver.
        
        // Use cholesky decomposition: inv(cov) = L * L'
        // |r|^2 = r' * L * L' * r = (L' * r)' * (L' * r)
        // define x = L' * r (matrix 9x1)
        Eigen::Matrix<double, 9, 9> Lt = Eigen::LLT<Eigen::Matrix<double, 9, 9>>(ic->invCovPreintegration_ij.block<9,9>(0,0)).matrixL().transpose();
        residual = Lt * residual;

        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        // Inverse of right jacobian of residual (delta_R_ij) calculated without adding delta increment.
        Eigen::Vector3d residual_R = (tempR.inverse() * R_i.inverse() * R_j).log();
        Eigen::Matrix3d JrInv = rightJacobianInverseSO3(residual_R);

        // Jacobian(9x4) of residual(9x1) w.r.t. ParameterBlock[0](4x1), i.e. delta_yaw_p_i
        if (jacobians[0]) {
            // The default storage order is ColMajor in Eigen.
            Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> jacobian_yaw_p_i(jacobians[0]);

            jacobian_yaw_p_i.setZero();

            // jacobian of residual(delta_R_ij) with respect to delta_r_i
            Eigen::Matrix3d J_R_r = -JrInv * R_j.matrix().transpose() * R_i.matrix();
            // jacobian of residual(delta_v_ij) with respect to delta_r_i
            Eigen::Matrix3d J_v_r = Sophus::SO3d::hat(R_i.inverse() * (frame_j->v - frame_i->v - _pMap->_gravity * ic->dt));
            // jacobian of residual(delta_p_ij) with respect to delta_r_i
            Eigen::Matrix3d J_p_r = Sophus::SO3d::hat(R_i.inverse() * (frame_j->p - frame_i->p - frame_i->v * ic->dt - _pMap->_gravity * ic->dt2 / 2));

            #ifdef CFSD
            jacobian_yaw_p_i.block<3,1>(0,0) = J_R_r.rightCols<1>();
            jacobian_yaw_p_i.block<3,1>(3,0) = J_v_r.rightCols<1>();
            jacobian_yaw_p_i.block<3,1>(6,0) = J_p_r.rightCols<1>();
            #endif
            
            #ifdef EUROC
            jacobian_yaw_p_i.block<3,1>(0,0) = J_R_r.leftCols<1>();
            jacobian_yaw_p_i.block<3,1>(3,0) = J_v_r.leftCols<1>();
            jacobian_yaw_p_i.block<3,1>(6,0) = J_p_r.leftCols<1>();
            #endif

            #ifdef KITTI
            jacobian_yaw_p_i.block<3,1>(0,0) = J_R_r.rightCols<1>();
            jacobian_yaw_p_i.block<3,1>(3,0) = J_v_r.rightCols<1>();
            jacobian_yaw_p_i.block<3,1>(6,0) = J_p_r.rightCols<1>();
            #endif

            // jacobian of residual(delta_p_ij) with respect to delta_p_i
            jacobian_yaw_p_i.block<3, 3>(6, 1) = -Eigen::Matrix3d::Identity();

            // since cost function is defined as: L' * r
            jacobian_yaw_p_i = Lt * jacobian_yaw_p_i;

            if (jacobian_yaw_p_i.maxCoeff() > 1e8 || jacobian_yaw_p_i.minCoeff() < -1e8)
                std::cout << "Numerical unstable in calculating jacobian_i!" << std::endl;
        }

        // Jacobian(9x4) of residuals(9x1) w.r.t. ParameterBlock[1](4x1), i.e. delta_yaw_p_j
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> jacobian_yaw_p_j(jacobians[1]);

            jacobian_yaw_p_j.setZero();

            // jacobian of residual(delta_R_ij) with respect to delta_r_j
            #ifdef CFSD
            jacobian_yaw_p_j.block<3,1>(0,0) = JrInv.rightCols<1>();
            #endif
            
            #ifdef EUROC
            jacobian_yaw_p_j.block<3,1>(0,0) = JrInv.leftCols<1>();
            #endif

            #ifdef KITTI
            jacobian_yaw_p_j.block<3,1>(0,0) = JrInv.rightCols<1>();
            #endif

            // jacobian of residual(delta_p_ij) with respect to delta_p_j
            jacobian_yaw_p_j.block<3, 3>(6, 1) = R_i.matrix().transpose() * R_j.matrix();

            // since cost function is defined as: L' * r
            jacobian_yaw_p_j = Lt * jacobian_yaw_p_j;

            if (jacobian_yaw_p_j.maxCoeff() > 1e8 || jacobian_yaw_p_j.minCoeff() < -1e8)
                std::cout << "Numerical unstable in calculating jacobian_j!" << std::endl;
        }

        return true;
    }

  private:
    const cfsd::Ptr<Map>& _pMap;
    int _idx;
};

struct ImageCostFunction4DOF : public ceres::CostFunction {
    ImageCostFunction4DOF(const int& n, const Eigen::MatrixXd& error, const Eigen::MatrixXd& F)
        : _numResiduals(error.size()), _numParameterBlocks(n), _error(error), _F(F){

        set_num_residuals(_numResiduals);
        
        std::vector<int32_t> prameterBlockSizes;
        for (int i = 0; i < n; i++)
            prameterBlockSizes.push_back(4);
        *mutable_parameter_block_sizes() = prameterBlockSizes;
    }

    virtual ~ImageCostFunction4DOF() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        /*  cfsd imu coordinate system
                      / x
                     /
                    ------ y
                    |
                    | z

            euroc imu coordinate system
                  x |  / z
                    | /
                    ------ y
        
            kitti imu coordinate system
                  z |  / x
                    | /
              y -----
        */
        Eigen::VectorXd delta(6*_numParameterBlocks);
        for (int i = 0; i < _numParameterBlocks; i++) {
            // delta_r
            #ifdef CFSD
            delta.segment<3>(6*i) = Eigen::Vector3d(0.0, 0.0, parameters[i][0]);
            #endif
            
            #ifdef EUROC
            delta.segment<3>(6*i) = Eigen::Vector3d(parameters[i][0], 0.0, 0.0);
            #endif

            #ifdef KITTI
            delta.segment<3>(6*i) = Eigen::Vector3d(0.0, 0.0, parameters[i][0]);
            #endif

            // delta_p
            delta.segment<3>(6*i+3) = Eigen::Vector3d(parameters[i][1], parameters[i][2], parameters[i][3]);
        }

        Eigen::Map<Eigen::VectorXd> residual(residuals, _numResiduals);

        residual = _error + _F * delta;

        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        for (int i = 0; i < _numParameterBlocks; i++) {
            if (jacobians[i]) {
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> jacobian_yaw_p(jacobians[i], _numResiduals, 4);

                #ifdef CFSD
                jacobian_yaw_p.leftCols<1>() = _F.col(6*i+2);
                #endif
                
                #ifdef EUROC
                jacobian_yaw_p.leftCols<1>() = _F.col(6*i);
                #endif

                #ifdef KITTI
                jacobian_yaw_p.leftCols<1>() = _F.col(6*i+2);
                #endif

                jacobian_yaw_p.rightCols<3>() = _F.block(0,6*i+3, _numResiduals,3);
            }
        }

        return true;
    }

  private:
    int _numResiduals{0};
    int _numParameterBlocks{0};
    Eigen::MatrixXd _error;
    Eigen::MatrixXd _F;
};

} // namespace cfsd

#endif // COST_FUNCTIONS_HPP