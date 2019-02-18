#ifndef IMU_FRAME_HPP
#define IMU_FRAME_HPP

#include "cfsd/common.hpp"

namespace cfsd {

class ImuFrame {
  public:
    using Ptr = std::shared_ptr<ImuFrame>;
    ImuFrame();
    ~ImuFrame();

    static ImuFrame::Ptr create();

    // preintegration
    void preintegration();


    void nextPose();

  private:
    unsigned long _id;
    double _timestamp;
    
    // acceleration, angular velocity
    float _accX, _accY, _accZ;
    float _gryX, _gryY, _gryZ;

    // preintegrated ...


    // IMU pose
    Sophus::SE3d _SE3;
    
};

} // namespace cfsd

#endif // IMU_FRAME_HPP