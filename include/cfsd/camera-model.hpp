#ifndef CAMERA_MODEL_HPP
#define CAMERA_MODEL_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"

namespace cfsd {

class CameraModel {
  public:
    CameraModel() {
        _cvCamLeft = Config::get<cv::Mat>("camLeft");
        _cvCamRight = Config::get<cv::Mat>("camRight");
        cv::cv2eigen(_cvCamLeft, _eigenCamLeft);
        cv::cv2eigen(_cvCamRight, _eigenCamRight);

        cv::Mat radDistLeft = Config::get<cv::Mat>("radDistLeft");
        cv::Mat tanDistLeft = Config::get<cv::Mat>("tanDistLeft");
        cv::vconcat(radDistLeft, tanDistLeft, _cvDistLeft);
        cv::cv2eigen(_cvDistLeft, _eigenDistLeft);

        cv::Mat radDistRight = Config::get<cv::Mat>("radDistRight");
        cv::Mat tanDistRight = Config::get<cv::Mat>("tanDistRight");
        cv::vconcat(radDistRight, tanDistRight, _cvDistRight);
        cv::cv2eigen(_cvDistRight, _eigenDistRight);

        EigenMatrix3Type R;
        EigenVector3Type t;
        cv::cv2eigen(Config::get<cv::Mat>("rotationLeftToRight"), R);
        cv::cv2eigen(Config::get<cv::Mat>("translationLeftToRight"), t);
        // std::cout << R.determinant() << std::endl;
        // std::cout << R * R.transpose() << std::endl;

        EigenQuaternionType q = EigenQuaternionType(R);
        // std::cout << q.norm() << std::endl;

        // _sophusLeft2Right = SophusSE3Type(R, t);  // Note: this would fail because R*R' is not exactly identity matrix
        _sophusLeft2Right = SophusSE3Type(q, t);     // however, convert R to quaternion will not have such problem
        // std::cout << _sophusLeft2Right.matrix3x4() << std::endl;
        _sophusRight2Left = _sophusLeft2Right.inverse();
        cv::eigen2cv(_sophusLeft2Right.matrix3x4(), _cvLeft2Right);
        cv::eigen2cv(_sophusRight2Left.matrix3x4(), _cvRight2Left);
    }

    // Camera intrinsic parameters.
    // OpenCV format
    cvMatrix3Type _cvCamLeft;
    cvMatrix3Type _cvCamRight;
    cvVector4Type _cvDistLeft;
    cvVector4Type _cvDistRight;
    // Eigen format
    EigenMatrix3Type _eigenCamLeft;
    EigenMatrix3Type _eigenCamRight;
    EigenVector4Type _eigenDistLeft;
    EigenVector4Type _eigenDistRight;

    // Camera extrinsic parameters.
    // OpenCV format
    cvMatrix34Type _cvLeft2Right;
    cvMatrix34Type _cvRight2Left;
    // Sophus format
    SophusSE3Type _sophusLeft2Right;
    SophusSE3Type _sophusRight2Left;


    // camera & imu extrinsics
    // static SophusSE3Type _left2imu; // left camera to imu

    // coordinates transformation
    /* coordinate system convension defined here:
        imu and camera, take imu coordinates as reference
        the stereo camera, take left camera as reference
        (so for right camera point, it is first transformed to left camera
        coordinate using extrinsic parameters, then to imu coordinate) */
    // inline EigenVector2Type camLeft2pixel (const EigenVector3Type& p_c) {
    //     return EigenVector2Type(p_c(0,0) / p_c(2,0) * _eigenCamLeft(0,0) + _eigenCamLeft(0,2),
    //                             p_c(1,0) / p_c(2,0) * _eigenCamLeft(1,1) + _eigenCamLeft(1,2));
    // }
    // inline EigenVector2Type camRight2pixel(const EigenVector3Type& p_c) {
    //     return EigenVector2Type(p_c(0,0) / p_c(2,0) * _eigenCamRight(0,0) + _eigenCamRight(0,2),
    //                             p_c(1,0) / p_c(2,0) * _eigenCamRight(1,1) + _eigenCamRight(1,2));
    // }
    // inline EigenVector3Type pixel2camLeft (const EigenVector2Type& p_p, const double depth) {
    //     return EigenVector3Type((p_p(0,0) - _eigenCamLeft(0,2)) / _eigenCamLeft(0,0) * depth,
    //                             (p_p(1,0) - _eigenCamLeft(1,2)) / _eigenCamLeft(1,1) * depth,
    //                             depth);
    // }
    // inline EigenVector3Type pixel2camRight(const EigenVector2Type& p_p, const double depth) {
    //     return EigenVector3Type((p_p(0,0) - _eigenCamRight(0,2)) / _eigenCamRight(0,0) * depth,
    //                             (p_p(1,0) - _eigenCamRight(1,2)) / _eigenCamRight(1,1) * depth,
    //                             depth);
    // }
    // inline EigenVector3Type camLeft2world (const EigenVector3Type& p_c, const SophusSE3Type& T_c_w) { return T_c_w * p_c; }
    // inline EigenVector3Type world2camLeft (const EigenVector3Type& p_w, const SophusSE3Type& T_c_w) { return T_c_w.inverse() * p_w; }
    // inline EigenVector3Type camLeft2camRight(const EigenVector3Type& p_c) { return _left2right * p_c; }
    // inline EigenVector3Type camRight2camLeft(const EigenVector3Type& p_c) { return _right2left * p_c; }
    // inline EigenVector3Type camLeft2imu(const EigenVector3Type& p_c, const SophusSE3Type& T_c_i) { return _left2imu * p_c; }
    // inline EigenVector3Type imu2camLeft(const EigenVector3Type& p_i, const SophusSE3Type& T_c_i) { return _left2imu.inverse() * p_i; }

};

} // namespace cfsd

#endif // CAMERA_MODEL_HPP
