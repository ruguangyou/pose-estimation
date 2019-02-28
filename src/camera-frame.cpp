#include "cfsd/camera-frame.hpp"

namespace cfsd {

// static parameters need initilization
EigenMatrix3Type CameraFrame::_camLeft(EigenMatrix3Type::Zero());
EigenMatrix3Type CameraFrame::_camRight(EigenMatrix3Type::Zero());
EigenVector4Type CameraFrame::_distLeft(EigenVector4Type::Zero());
EigenVector4Type CameraFrame::_distRight(EigenVector4Type::Zero());
SophusSE3Type CameraFrame::_right2left(EigenMatrix3Type::Identity(), EigenVector3Type::Zero());
SophusSE3Type CameraFrame::_left2right(EigenMatrix3Type::Identity(), EigenVector3Type::Zero());

// default construtor
CameraFrame::CameraFrame() {}

CameraFrame::~CameraFrame() {}

CameraFrame::CameraFrame(long id, long timestamp, cv::Mat img)
    : _id(id), _timestamp(timestamp) {
        _imgLeft = img(cv::Range(0, img.rows), cv::Range(0, img.cols/2));
        _imgRight = img(cv::Range(0, img.rows), cv::Range(img.cols/2, img.cols));
    }

CameraFrame::Ptr CameraFrame::create(long timestamp, cv::Mat img) {
    static long frameCounter = 0;
    return CameraFrame::Ptr(new CameraFrame(frameCounter++, timestamp, img));
}

void CameraFrame::setIntrinsics(cv::Mat& camLeft, cv::Mat& distLeft, cv::Mat& camRight, cv::Mat& distRight) {
    cv::cv2eigen(camLeft, _camLeft);
    cv::cv2eigen(distLeft, _distLeft);
    cv::cv2eigen(camRight, _camRight);
    cv::cv2eigen(distRight, _distRight);
}

void CameraFrame::setExtrinsics(cv::Mat& R, cv::Mat& t) {
    EigenMatrix3Type _R;
    EigenVector3Type _t;
    cv::cv2eigen(R, _R);
    cv::cv2eigen(t, _t);
    _left2right = SophusSE3Type(_R, _t);
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
    cv::Mat camLeft, distLeft, camRight, distRight;
    cv::eigen2cv(_camLeft, camLeft);
    cv::eigen2cv(_distLeft, distLeft);
    cv::eigen2cv(_camRight, camRight);
    cv::eigen2cv(_distRight, distRight);
    cv::undistort(_imgLeft, _imgLeft, camLeft, distLeft);
    cv::undistort(_imgRight, _imgRight, camRight, distRight);
}

} // namespace cfsd