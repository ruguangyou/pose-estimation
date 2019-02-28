#ifndef CAMERA_FRAME_HPP
#define CAMERA_FRAME_HPP

#include "cfsd/common.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

class CameraFrame {
  public:
    // camera intrinsic parameters
    static EigenMatrix3Type _camLeft;
    static EigenMatrix3Type _camRight;
    static EigenVector4Type _distLeft;
    static EigenVector4Type _distRight;
    // camera extrinsic parameters
    static SophusSE3Type _right2left;
    static SophusSE3Type _left2right;
    // camera & imu extrinsics
    // static SophusSE3Type _left2imu; // left camera to imu

    using Ptr = std::shared_ptr<CameraFrame>;

    // constructor and deconstructor
    CameraFrame();
    ~CameraFrame();

    CameraFrame(long id, long timestamp, cv::Mat img);

    // factory function
    static CameraFrame::Ptr create(long timestamp, cv::Mat img);

    // setting camera parameters
    static void setIntrinsics(cv::Mat& camLeft, cv::Mat& distLeft, cv::Mat& camRight, cv::Mat& distRight);
    static void setExtrinsics(cv::Mat& R, cv::Mat& t);

    // image pre-processing
    // void extractROI();
    void undistort();
    // void rectify();
    

  private:
    unsigned long _id;
    long _timestamp;
    // left and right images from stereo camera
    cv::Mat _imgLeft, _imgRight;

  public:
    // getter funtions
    inline long getTimestamp() const { return _timestamp; }
    inline const cv::Mat& getImgLeft() const { return _imgLeft; }
    inline const cv::Mat& getImgRight() const { return _imgRight; }
    inline const EigenMatrix3Type& getCamLeft() const { return _camLeft; }
    inline const EigenMatrix3Type& getCamRight() const { return _camRight; }
    inline const EigenVector4Type& getDistLeft() const { return _distLeft; }
    inline const EigenVector4Type& getDistRight() const { return _distRight; }
    inline const SophusSE3Type& getRightToLeft() const { return _right2left; }
    inline const SophusSE3Type& getLeftToRight() const { return _left2right; }

    // inline EigenVector3Type getCameraCenter() const; // the coordinated is relative to left camera or imu??

    // coordinates transformation
    /* coordinate system convension defined here:
        imu and camera, take imu coordinates as reference
        the stereo camera, take left camera as reference
        (so for right camera point, it is first transformed to left camera
        coordinate using extrinsic parameters, then to imu coordinate) */
    inline EigenVector2Type camLeft2pixel (const EigenVector3Type& p_c) {
        return EigenVector2Type(p_c(0,0) / p_c(2,0) * _camLeft(0,0) + _camLeft(0,2),
                                p_c(1,0) / p_c(2,0) * _camLeft(1,1) + _camLeft(1,2));
    }
    inline EigenVector2Type camRight2pixel(const EigenVector3Type& p_c) {
        return EigenVector2Type(p_c(0,0) / p_c(2,0) * _camRight(0,0) + _camRight(0,2),
                                p_c(1,0) / p_c(2,0) * _camRight(1,1) + _camRight(1,2));
    }
    inline EigenVector3Type pixel2camLeft (const EigenVector2Type& p_p, const double depth) {
        return EigenVector3Type((p_p(0,0) - _camLeft(0,2)) / _camLeft(0,0) * depth,
                                (p_p(1,0) - _camLeft(1,2)) / _camLeft(1,1) * depth,
                                depth);
    }
    inline EigenVector3Type pixel2camRight(const EigenVector2Type& p_p, const double depth) {
        return EigenVector3Type((p_p(0,0) - _camRight(0,2)) / _camRight(0,0) * depth,
                                (p_p(1,0) - _camRight(1,2)) / _camRight(1,1) * depth,
                                depth);
    }
    inline EigenVector3Type camLeft2world (const EigenVector3Type& p_c, const SophusSE3Type& T_c_w) { return T_c_w * p_c; }
    inline EigenVector3Type world2camLeft (const EigenVector3Type& p_w, const SophusSE3Type& T_c_w) { return T_c_w.inverse() * p_w; }
    inline EigenVector3Type camLeft2camRight(const EigenVector3Type& p_c) { return _left2right * p_c; }
    inline EigenVector3Type camRight2camLeft(const EigenVector3Type& p_c) { return _right2left * p_c; }
    // inline EigenVector3Type camLeft2imu(const EigenVector3Type& p_c, const SophusSE3Type& T_c_i) { return _left2imu * p_c; }
    // inline EigenVector3Type imu2camLeft(const EigenVector3Type& p_i, const SophusSE3Type& T_c_i) { return _left2imu.inverse() * p_i; }

};

} // namespace cfsd

#endif // CAMERA_FRAME_HPP
