#ifndef COMMON_HPP
#define COMMON_HPP

// commonly include files: std, OpenCV, Eigen, Sophus

// Eigen
#include <Eigen/Core>       // Matrix and Array classes, basic linear algebra (including triangular and selfadjoint products), array manipulation
#include <Eigen/Geometry>   // Transform, Translation, Scaling, Rotation2D and 3D rotations (Quaternion, AngleAxis)

// Sophus
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp> // cv::eigen2cv, cv::cv2eigen

#ifdef DEBUG_IMG
#include <opencv2/highgui/highgui.hpp>
#endif

// std utilities
#include <iostream>         // several standard stream objects
#include <memory>           // higher level memory management utilities (std::shared_ptr, std::unique_ptr, ...)
#include <chrono>           // C++ time utilities
#include <string>           // std::basic_string class template
#include <algorithm>        // algorithms that operate on ranges (std::move, std::min, std::max, ...)
#include <cmath>             // common mathematics functions

// std thread support
#include <thread>           // std::thread class and supporting functions
#include <mutex>            // mutual exclusion primitives

// std containers library
#include <vector>           // std::vector container
#include <queue>            // std::queue and std::priority_queue container
#include <list>             // std::list container
#include <map>              // std::map and std::multimap associative containers
#include <unordered_map>    // std::unordered_map and std::unordered_multimap unordered associative containers
#include <set>              // std::set and std::multiset associative containers

// pre-defined type (easy to change between float and double type)
typedef double precisionType;
typedef Eigen::Matrix3d EigenMatrix3Type;
typedef Eigen::Vector3d EigenVector3Type;
typedef Eigen::Vector2d EigenVector2Type;
typedef Eigen::Matrix<double,4,1> EigenVector4Type;
typedef Eigen::Quaterniond EigenQuaternionType;
typedef Sophus::SE3d SophusSE3Type;
typedef Sophus::SO3d SophusSO3Type;
typedef cv::Point3d cvPoint3Type;
typedef cv::Point2d cvPoint2Type;
typedef cv::Matx33d cvMatrix3Type;
typedef cv::Matx41d cvVector4Type;
typedef cv::Matx34d cvMatrix34Type;

// pre-defined smart pointer type
namespace cfsd {
template<typename T>
using Ptr = std::shared_ptr<T>; // the same as `typedef std::shared_ptr<VisualInertialOdometry> Ptr`, however `typedef` is limited and cannot use `template`
} // namespace

#endif // COMMON_HPP