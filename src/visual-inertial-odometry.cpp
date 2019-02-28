#include "cfsd/visual-inertial-odometry.hpp"

namespace cfsd {

VisualInertialOdometry::VisualInertialOdometry(bool verbose, bool debug) : _state(INITIALIZING), _verbose(verbose), _debug(debug) {
    _featureTracker =  FeatureTracker::create(_verbose, _debug);
    // _map = Map::create(_verbose, _debug);

    cv::Mat camLeft   = Config::get<cv::Mat>("camLeft");
    cv::Mat camRight  = Config::get<cv::Mat>("camRight");
    cv::Mat radDistLeft  = Config::get<cv::Mat>("radDistLeft");
    cv::Mat radDistRight = Config::get<cv::Mat>("radDistRight");
    cv::Mat tanDistLeft = Config::get<cv::Mat>("tanDistLeft");
    cv::Mat tanDistRight = Config::get<cv::Mat>("tanDistRight");

    cv::Mat distLeft, distRight;
    cv::hconcat(radDistLeft, tanDistLeft, distLeft);
    cv::hconcat(radDistRight, tanDistRight, distRight);
    if (_debug) {
        std::cout << "distLeft: " << distLeft << std::endl;
        std::cout << "distRight: " << distRight << std::endl;
    }
    CameraFrame::setIntrinsics(camLeft, distLeft, camRight, distRight);

    cv::Mat rotationL2R = Config::get<cv::Mat>("rotationLeftToRight");
    cv::Mat translationL2R = Config::get<cv::Mat>("translationLeftToRight");
    CameraFrame::setExtrinsics(rotationL2R, translationL2R);
}

VisualInertialOdometry::Ptr VisualInertialOdometry::create(bool verbose = false, bool debug = false) {
    return VisualInertialOdometry::Ptr(new VisualInertialOdometry(verbose, debug));
}

void VisualInertialOdometry::processFrame(long timestamp, cv::Mat& img) {
    CameraFrame::Ptr camFrame = CameraFrame::create(timestamp, img);
    // camFrame->undistort();
    switch (_state) {
        case INITIALIZING:
        {
            // set the first frame as key frame
            KeyFrame::Ptr keyFrame = KeyFrame::create(timestamp, camFrame, _verbose, _debug);
            // temperarily, take the left camera of first frame as reference coordinate system
            SophusSE3Type curCamPose(EigenMatrix3Type::Identity(), EigenVector3Type::Zero());
            if (_verbose) {
                std::cout << std::endl << "timestamp: " << timestamp << std::endl;
                std::cout << "Current camera position:\n" << curCamPose.inverse().translation() << std::endl;
                std::cout << "Current camera orientation (roll,pitch,yaw):\n" << curCamPose.so3().log() << std::endl;
            }
            keyFrame->setCamPose(curCamPose);
            keyFrame->matchAndTriangulate();
            _featureTracker->setKeyFrame(keyFrame);
            _keyFrames.push_back(keyFrame);
            _state = RUNNING;
            _camPoses.push_back(curCamPose);
            break;
        }
        case RUNNING:
        {
            // try to match current frame and 
            _featureTracker->setCamFrame(camFrame);
            _featureTracker->extractKeypoints();
            _featureTracker->matchKeypoints();
            if (_featureTracker->curIsKeyFrame()) {
                // if current frame is key frame, update the _keyFrame in _featureTracker
                SophusSE3Type curCamPose;
                _featureTracker->computeCamPose(curCamPose);
                if (_verbose) {
                    std::cout << "timestamp: " << timestamp << std::endl;
                    std::cout << "Current camera position:\n  " << curCamPose.inverse().translation() << std::endl;
                    std::cout << "Current camera orientation (roll,pitch,yaw):\n  " << curCamPose.so3().log() << std::endl;
                }
                KeyFrame::Ptr keyFrame = KeyFrame::create(timestamp, camFrame, _verbose, _debug);
                keyFrame->setCamPose(curCamPose);
                keyFrame->matchAndTriangulate();
                _featureTracker->setKeyFrame(keyFrame);
                _frameMatches.push_back(_featureTracker->getDMatch());
                _keyFrames.push_back(keyFrame);
                // _camPositions.push_back(curCamPose.inverse().translation());
                // _camEulerAngles.push_back(curCamPose.so3().matrix().eulerAngles(0,1,2)); // (0,1,2) -> (roll,pitch,yaw)
                // _camEulerAngles.push_back(curCamPose.so3().log());
                _camPoses.push_back(curCamPose);
            }
            break;
        }
        case LOST:
        {
            // if lost the track of features ...
            // need relocalization
            break;
        }
    }
}

} // namespace cfsd
