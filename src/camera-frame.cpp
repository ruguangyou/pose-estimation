#include "cfsd/camera-frame.hpp"

namespace cfsd {

// default construtor
CameraFrame::CameraFrame() {}

CameraFrame::~CameraFrame() {}

CameraFrame::CameraFrame(long id, double timestamp, cv::Mat img)
    : _id(id), _timestamp(timestamp) {
        _imgLeft = img(cv::Range(0, img.rows), cv::Range(0, img.cols/2));
        _imgRight = img(cv::Range(0, img.rows), cv::Range(img.cols/2, img.cols));
    }

CameraFrame::Ptr CameraFrame::create(double timestamp, cv::Mat img) {
    static long factory_id = 0;
    return CameraFrame::Ptr(new CameraFrame(factory_id++, timestamp, img));
}

void CameraFrame::setIntrinsics(cv::Mat& camLeft, cv::Mat& distLeft, cv::Mat& camRight, cv::Mat& distRight) {
    cv::cv2eigen(camLeft, _camLeft);
    cv::cv2eigen(distLeft, _distLeft);
    cv::cv2eigen(camRight, _camRight);
    cv::cv2eigen(distRight, _distRight);
}

void CameraFrame::setExtrinsics(cv::Mat& R, cv::Mat& t) {
    Eigen::Matrix3d _R;
    Eigen::Vector3d _t;
    cv::cv2eigen(R, _R);
    cv::cv2eigen(t, _t);
    _left2right = Sophus::SE3d(_R, _t);
    _right2left = _left2right.inverse();
}

// maybe this part could be put in the main program, so that the copying cost will be reduced later when passing parameters
// image pre-processing
// void CameraFrame::cropImage() {
//     // crop the image, remove the part we are not interested, e.g. sky
//     int startRow = _imgLeft.rows / 2;
//     int endRow = _imgLeft.rows * 8 / 9;
//     int startCol = 0;
//     int endCol = _imgLeft.cols;
//     _imgLeft = _imgLeft(Range(startRow, endRow), Range(startCol, endCol));
//     _imgRight = _imgRight(Range(startRow, endRow), Range(startCol, endCol));
// }

void CameraFrame::undistort() {
    // the images should be undistorted
    cv::undistort(_imgLeft, _imgLeft, _camLeft, _distLeft);
    cv::undistort(_imgRight, _imgRight, _camRight, _distRight);
}

// coordinate tranformation
Eigen::Vector2d CameraFrame::camLeft2pixel(const Eigen::Vector3d& p_c) {
    return Eigen::Vector2d(p_c(0,0) / p_c(2,0) * _camLeft(0,0) + _camLeft(0,2),
                           p_c(1,0) / p_c(2,0) * _camLeft(1,1) + _camLeft(1,2));
}

Eigen::Vector2d CameraFrame::camRight2pixel(const Eigen::Vector3d& p_c) {
    return Eigen::Vector2d(p_c(0,0) / p_c(2,0) * _camRight(0,0) + _camRight(0,2),
                           p_c(1,0) / p_c(2,0) * _camRight(1,1) + _camRight(1,2));
}

Eigen::Vector3d CameraFrame::pixel2camLeft(const Eigen::Vector2d& p_p, const double depth) {
    return Eigen::Vector3d((p_p(0,0) - _camLeft(0,2)) / _camLeft(0,0) * depth,
                           (p_p(1,0) - _camLeft(1,2)) / _camLeft(1,1) * depth,
                           depth);
}

Eigen::Vector3d CameraFrame::pixel2camRight(const Eigen::Vector2d& p_p, const double depth) {
    return Eigen::Vector3d((p_p(0,0) - _camRight(0,2)) / _camRight(0,0) * depth,
                           (p_p(1,0) - _camRight(1,2)) / _camRight(1,1) * depth,
                           depth);
}

Eigen::Vector3d CameraFrame::camLeft2world(const Eigen::Vector2d& p_c, const Sophus::SE3d& T_c_w) {
    // T_c_w is the transform from left camera to world
    return T_c_w * p_c;
}

Eigen::Vector2d CameraFrame::world2camLeft(const Eigen::Vector3d& p_w, const Sophus::SE3d& T_c_w) {
    return T_c_w.inverse() * p_w;
}

Eigen::Vector3d CameraFrame::camRight2camLeft(const Eigen::Vector3d& p_c) {
    return _left2right * p_c;
}

Eigen::Vector3d CameraFrame::camRight2camLeft(const Eigen::Vector3d& p_c) {
    return _right2left * p_c;
}

// Eigen::Vector3d CameraFrame::camLeft2imu(const Eigen::Vector3d& p_c) {
//     return _left2imu * p_c;
// }

// Eigen::Vector3d CameraFrame::imu2camLeft(const Eigen::Vector3d& p_i) {
//     return _left2imu.inverse() * p_i;
// }


} // namespace cfsd