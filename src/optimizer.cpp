#include "cfsd/optimizer.hpp"

namespace cfsd {

class ImuCostFunction : public ceres::SizedCostFunction<15, /* residuals */
                                                         9, /* r, v, p at time i */
                                                         9, /* r, v, p at time j */
                                                         6 /* bias of gyr and acc */> {
  public:
    ImuCostFunction(const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator) : _pImuPreintegrator(pImuPreintegrator) {}

    virtual ~ImuCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: rvp_i, rvp_j, bg_ba
        // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]

        // Quaternion: w + xi + yj + zk
        // When creating an Eigen quaternion through the constructor the elements are accepted in w, x, y, z order;
        // whereas Eigen stores the elements in memory as [x, y, z, w] where the real part is last.
        // The commonly used memory layout for quaternion is [w, x, y, z], so is it in Ceres.
        // EigenQuaternionType q_i(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        // EigenQuaternionType q_i(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        // Update: instead of using quaternion, use rotation vector which can be mapped to SO3 space using Exp map.
        // Rotation vector.
        EigenVector3Type r_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        EigenVector3Type r_j(parameters[1][0], parameters[1][1], parameters[1][2]);

        // Velocity.
        EigenVector3Type v_i(parameters[0][3], parameters[0][4], parameters[0][5]);
        EigenVector3Type v_j(parameters[1][3], parameters[1][4], parameters[1][5]);

        // Position.
        EigenVector3Type p_i(parameters[0][6], parameters[0][7], parameters[0][8]);
        EigenVector3Type p_j(parameters[1][6], parameters[1][7], parameters[1][8]);

        // Gyroscope bias.
        EigenVector3Type delta_bg(parameters[2][0], parameters[2][1], parameters[2][2]);

        // Accelerometer bias.
        EigenVector3Type delta_ba(parameters[2][3], parameters[2][4], parameters[2][5]);

        // Call ImuPreintegrator::evaluate() to compute residuals and jacobians.
        return _pImuPreintegrator->evaluate(r_i, v_i, p_i, r_j, v_j, p_j, delta_bg, delta_ba, residuals, jacobians);
    }

  private:
    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;
};


Optimizer::Optimizer(const bool verbose) : _verbose(verbose) {}

// Assume vehicle state at time i is identity R and zero v, p, bg, ba,
// and given the relative transformation from i to j,
// optimize the state at time j.
void Optimizer::localOptimize(const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator) {
    // Parameters to be optimized: rotation, velocity, position, deltaBiasGyr, deltaBiasAcc
    double rvp_i[9]; // r, v, p at time i
    double rvp_j[9]; // r, v, p at time j
    double bg_ba[6] = {0, 0, 0, 0, 0, 0}; // delta_bg, delta_ba from i to j

    // Read initial values from ImuPreintegrator.
    pImuPreintegrator->getParameters(rvp_i, rvp_j);

    // Build the problem.
    ceres::Problem problem;

    // Set up cost function (a.k.a. residuals).
    ceres::CostFunction* costFunction = new ImuCostFunction(pImuPreintegrator);
    problem.AddResidualBlock(costFunction, nullptr, rvp_i, rvp_j, bg_ba);

    // Set the solver.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // todo...
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    // Run the solver.
    ceres::Solve(options, &problem, &summary);
    
    if (_verbose) {
        // Show the report.
        std::cout << summary.BriefReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;
    }

    // Update state values in ImuPreintegrator.
    pImuPreintegrator->updateState(rvp_i, rvp_j, bg_ba);
}


} // namespace cfsd