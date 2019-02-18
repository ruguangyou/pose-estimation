#ifndef IMU_PREINTEGRATOR_HPP
#define IMU_PREINTEGRATOR_HPP

#include "cfsd/common.hpp"
#include "cfsd/imu-frame.hpp"

namespace cfsd {

class ImuPreintegrator {
  public:
    using Ptr = std::shared_ptr<ImuPreintegrator>;
    ImuPreintegrator();
    ~ImuPreintegrator();

    static ImuPreintegrator::Ptr create();

  private:

};

} // namespace cfsd

#endif // IMU_PREINTEGRATOR_HPP