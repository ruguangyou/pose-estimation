#ifndef CAMERA_MODEL_HPP
#define CAMERA_MODEL_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace cfsd {

/* CFSD coordinate system convension:
    
    world frame (the z direction should be parallel with gravitational direction)
       / x
      /
     ------ y
     |
     | z

    T_WB = [R_WB, W_P]
    transformation from body frame to world frame, consisits of rotation from B to W, and position of B w.r.t W.
    Note: if IMU is tilted at beginning, there should be a initial rotation from B to W.

    body/imu frame
       / x (roll)
      /
     ------ y (pitch)
     |
     | z (yaw)

    T_BC = [R_BC, B_P]
    transformation from camera frame to body frame, consists of rotation from C to B, and position of C w.r.t B.

    camera frame (rectified)
       / z
      /
     ------ x
     |
     | y

    * imu preintegration calculates T_WB
    * triangulated points from stereo pairs are w.r.t camera frame
    * would like to store the 3D landmarks w.r.t world frame, so we should apply two transformations to these points, i.e. T_WB * T_BC * C_points
*/

class CameraModel {
  public:
    CameraModel() : _imageSize(), _K1(), _K2(), _D1(), _D2(), _R(), _T(), _R1(), _R2(), _P1(), _P2(), _Q(), _K_L(), _P_L(), _P_R(), _T_BC(), _T_CB() {
        _stdX = Config::get<double>("stdX");
        _stdY = Config::get<double>("stdY");

        _imageSize.height = Config::get<int>("imageHeight");
        _imageSize.width = Config::get<int>("imageWidth");
        
        _K1 = Config::get<cv::Mat>("camLeft");
        _D1 = Config::get<cv::Mat>("distLeft");
        _K2 = Config::get<cv::Mat>("camRight");
        _D2 = Config::get<cv::Mat>("distRight");
        _R = Config::get<cv::Mat>("rotationLeftToRight");
        // cv::Rodrigues(Config::get<cv::Mat>("rotationLeftToRight"), _R);
        _T = Config::get<cv::Mat>("translationLeftToRight");

        // CALIB_ZERO_DISPARITY: with this flag, the function makes the principal points of each camera have the same pixel coordinates in the rectified views.
        // Operation flags: may be zero or CALIB_ZERO_DISPARITY
        //    If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. 
        //    If the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
        // Free scaling parameter alpha: 
        //    If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. 
        //    alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). 
        //    alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). 
        cv::Rect validRoi[2];
        cv::stereoRectify(_K1, _D1, _K2, _D2, _imageSize, _R, _T, _R1, _R2, _P1, _P2, _Q, cv::CALIB_ZERO_DISPARITY, 0, _imageSize, &validRoi[0], &validRoi[1]);

        // Computes the undistortion and rectification transformation map.
        cv::initUndistortRectifyMap(_K1, _D1, _R1, _P1, _imageSize, CV_16SC2, _rmap[0][0], _rmap[0][1]);
        cv::initUndistortRectifyMap(_K2, _D2, _R2, _P2, _imageSize, CV_16SC2, _rmap[1][0], _rmap[1][1]);
        std::cout << "Camera init undistort-rectify-map done!" << std::endl << std::endl;

        _K_L = _P1.colRange(0,3);
        cv::cv2eigen(_P1, _P_L);
        cv::cv2eigen(_P2, _P_R);

        // Transformation between body/imu and camera.
        // TODO...................
        // Body frame to camera frame.
        cv::Mat cv_R_CB = Config::get<cv::Mat>("rotationImuToCamera");
        cv::Mat cv_t_CB = Config::get<cv::Mat>("translationImuToCamera");
        Eigen::Matrix3d eigen_R_CB;
        Eigen::Vector3d eigen_t_CB;
        cv::cv2eigen(cv_R_CB, eigen_R_CB);
        cv::cv2eigen(cv_t_CB, eigen_t_CB);
        _T_CB = Sophus::SE3d(Eigen::Quaterniond(eigen_R_CB), eigen_t_CB);
        _T_BC = _T_CB.inverse();
    }

    // Standard deviation of pixel measurement.
    double _stdX{0};
    double _stdY{0};

    // Convention: first camera is left, second camera is right.
    
    // Image size.
    cv::Size _imageSize;
    
    // Camera intrinsic parameters.
    // camera matrix:
    cv::Mat _K1, _K2;
    // distortion coefficient [k1, k2, p1, p2, k3] (k is radial distortion, p is tangential distortion):
    cv::Mat _D1, _D2;

    // Camera extrinsic parameters.
    // rotation matrix and translation vector from the coordinate systems of the first to the second cameras:
    cv::Mat _R, _T;

    // Rectified camera parameters.
    // 3x3 rectification transform (rotation matrix) for the first and second camera:
    cv::Mat _R1, _R2;
    // 3x4 projection matrix in the new (rectified) coordinate systems for the first and second camera.
    cv::Mat _P1, _P2;
    // 4×4 disparity-to-depth mapping matrix (see reprojectImageTo3D ).
    cv::Mat _Q;

    // 3x3 camera matrix after undistort and rectification.
    cv::Mat _K_L;

    // Undistortion and rectification transformation map.
    cv::Mat _rmap[2][2];

    // 3x4 projection matrix in the rectified coordinate systems in Eigen format.
    Eigen::Matrix<double, 3, 4> _P_L, _P_R;

    // camera & imu extrinsics
    Sophus::SE3d _T_BC; // camera frame to body frame.
    Sophus::SE3d _T_CB; // body frame to camera frame.

    // coordinates transformation
    /* coordinate system convension defined here:
        imu and camera, take imu coordinates as reference
        the stereo camera, take left camera as reference
        (so for right camera point, it is first transformed to left camera
        coordinate using extrinsic parameters, then to imu coordinate) */
    // inline Eigen::Vector2d camLeft2pixel (const Eigen::Vector3d& p_c) {
    //     return Eigen::Vector2d(p_c(0,0) / p_c(2,0) * _eigenCamLeft(0,0) + _eigenCamLeft(0,2),
    //                             p_c(1,0) / p_c(2,0) * _eigenCamLeft(1,1) + _eigenCamLeft(1,2));
    // }
    // inline Eigen::Vector2d camRight2pixel(const Eigen::Vector3d& p_c) {
    //     return Eigen::Vector2d(p_c(0,0) / p_c(2,0) * _eigenCamRight(0,0) + _eigenCamRight(0,2),
    //                             p_c(1,0) / p_c(2,0) * _eigenCamRight(1,1) + _eigenCamRight(1,2));
    // }
    // inline Eigen::Vector3d pixel2camLeft (const Eigen::Vector2d& p_p, const double depth) {
    //     return Eigen::Vector3d((p_p(0,0) - _eigenCamLeft(0,2)) / _eigenCamLeft(0,0) * depth,
    //                             (p_p(1,0) - _eigenCamLeft(1,2)) / _eigenCamLeft(1,1) * depth,
    //                             depth);
    // }
    // inline Eigen::Vector3d pixel2camRight(const Eigen::Vector2d& p_p, const double depth) {
    //     return Eigen::Vector3d((p_p(0,0) - _eigenCamRight(0,2)) / _eigenCamRight(0,0) * depth,
    //                             (p_p(1,0) - _eigenCamRight(1,2)) / _eigenCamRight(1,1) * depth,
    //                             depth);
    // }
    // inline Eigen::Vector3d camLeft2world (const Eigen::Vector3d& p_c, const Sophus::SE3d& T_c_w) { return T_c_w * p_c; }
    // inline Eigen::Vector3d world2camLeft (const Eigen::Vector3d& p_w, const Sophus::SE3d& T_c_w) { return T_c_w.inverse() * p_w; }
    // inline Eigen::Vector3d camLeft2camRight(const Eigen::Vector3d& p_c) { return _left2right * p_c; }
    // inline Eigen::Vector3d camRight2camLeft(const Eigen::Vector3d& p_c) { return _right2left * p_c; }
    // inline Eigen::Vector3d camLeft2imu(const Eigen::Vector3d& p_c, const Sophus::SE3d& T_c_i) { return _left2imu * p_c; }
    // inline Eigen::Vector3d imu2camLeft(const Eigen::Vector3d& p_i, const Sophus::SE3d& T_c_i) { return _left2imu.inverse() * p_i; }

};

} // namespace cfsd

#endif // CAMERA_MODEL_HPP
