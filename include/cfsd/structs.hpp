#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include "cfsd/common.hpp"

namespace cfsd {

struct Feature {
    Feature() {}
    Feature(const int& frameCount, const cv::Point2d& pixelL, const cv::Point2d& pixelR, const cv::Mat& descriptorL, const cv::Mat& descriptorR, Eigen::Vector3d& position, const int& age)
      : descriptorL(descriptorL), descriptorR(descriptorR), position(position), age(age) {
        seenByFrames.push_back(frameCount);
        pixelsL.push_back(pixelL);
        pixelsR.push_back(pixelR);
    }

    // ID of Frames which can see this feature.
    std::vector<int> seenByFrames;

    // Pixel coordinate of this feature in each frame.
    std::vector<cv::Point2d> pixelsL;
    std::vector<cv::Point2d> pixelsR;
    
    // Feature descriptor, comes from the earliest frame, i.e. _seenByFrames[0].
    cv::Mat descriptorL;
    cv::Mat descriptorR;

    // 3D landmark position w.r.t world frame.
    Eigen::Vector3d position;

    int age;
};

struct MapPoint {
    MapPoint() {}
    MapPoint(const cfsd::Ptr<Feature>& f) 
        : seenByFrames(f->seenByFrames), pixelsL(f->pixelsL), pixelsR(f->pixelsR), position(f->position) {}

    // ID of Frames which can see this feature.
    std::vector<int> seenByFrames;

    // Pixel coordinate of this feature in each frame.
    std::vector<cv::Point2d> pixelsL;
    std::vector<cv::Point2d> pixelsR;

    // 3D landmark position w.r.t world frame.
    Eigen::Vector3d position;
};

struct ImuConstraint {
    ImuConstraint() {}
    ImuConstraint(const Eigen::Matrix<double,15,15>& covPreintegration_ij, const Eigen::Vector3d& bg_i, const Eigen::Vector3d& ba_i,
                  const Sophus::SO3d& delta_R_ij, const Eigen::Vector3d& delta_v_ij, const Eigen::Vector3d& delta_p_ij,
                  const Eigen::Matrix3d& d_R_bg_ij, const Eigen::Matrix3d& d_v_bg_ij, const Eigen::Matrix3d& d_v_ba_ij, 
                  const Eigen::Matrix3d& d_p_bg_ij, const Eigen::Matrix3d& d_p_ba_ij, const double& dt) : 
        covPreintegration_ij(covPreintegration_ij), bg_i(bg_i), ba_i(ba_i), delta_R_ij(delta_R_ij), delta_v_ij(delta_v_ij), delta_p_ij(delta_p_ij),
        d_R_bg_ij(d_R_bg_ij), d_v_bg_ij(d_v_bg_ij), d_v_ba_ij(d_v_ba_ij), d_p_bg_ij(d_p_bg_ij), d_p_ba_ij(d_p_ba_ij), dt(dt) { dt2 = dt* dt; }

    // Covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p, delta_bg, delta_ba]
    Eigen::Matrix<double,15,15> covPreintegration_ij;

    // Bias of gyroscope and accelerometer at time i.
    Eigen::Vector3d bg_i, ba_i;
    
    // Preintegrated delta_R, delta_v, delta_p.
    Sophus::SO3d delta_R_ij;
    Eigen::Vector3d delta_v_ij;
    Eigen::Vector3d delta_p_ij;

    // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
    Eigen::Matrix3d d_R_bg_ij;
    Eigen::Matrix3d d_v_bg_ij;
    Eigen::Matrix3d d_v_ba_ij;
    Eigen::Matrix3d d_p_bg_ij;
    Eigen::Matrix3d d_p_ba_ij;

    // The time between two camera frames, dt2 = dt^2.
    double dt{0}, dt2{0};
};

} // namespace cfsd

#endif // STRUCTS_HPP
