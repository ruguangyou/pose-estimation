#include "cfsd/visual-inertial-odometry.hpp"

namespace cfsd {

VisualInertialOdometry::VisualInertialOdometry() : _state(INITIALIZING) {}

VisualInertialOdometry::~VisualInertialOdometry() {}

VisualInertialOdometry::Ptr VisualInertialOdometry::create() {
    return VisualInertialOdometry::Ptr(new VisualInertialOdometry());
}

void VisualInertialOdometry::processFrame() {
    switch (_state) {
        case INITIALIZING:
    }
}

void VisualInertialOdometry::addKeyFrame() {
    // add a new key frame if curIsKeyFrame() of FeatureTracker returns true
    CameraFrame::Ptr camFrame = _featureTracker->getCamFrame();
    KeyFrame::Ptr keyFrame = KeyFrame::create(camFrame->getTimestamp(), camFrame);
    
    _keyFrames.push_back(keyFrame);
}



} // namespace cfsd
