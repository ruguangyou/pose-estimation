#ifndef LOOP_CLOSURE_HPP
#define LOOP_CLOSURE_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/camera-model.hpp"
#include "cfsd/map.hpp"
#include "cfsd/optimizer.hpp"

// DBoW2 include path: /usr/local/include/DBoW2
#include <DBoW2.h>

namespace cfsd {

class LoopClosure {
  public:
    LoopClosure(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<Optimizer>& pOptimizer, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose);

    void run();

    // Change the structure descriptors, put each row of Mat into vector.
    void changeStructure(const cv::Mat& mat, std::vector<cv::Mat>& vec);

    void addImage(const cv::Mat& descriptorsMat);

    int detectLoop(const cv::Mat& descriptorsMat, const int& frameID);

    void setToCloseLoop(const int& minLoopFrameID, const int& curFrameID);
    void setToCloseLoop();

    bool computeLoopInfo(const int& refFrameID, const int& curFrameID, Eigen::Vector3d& r, Eigen::Vector3d& p);

  private:
    OrbDatabase _db;

    std::mutex _dbMutex{};
    
    cfsd::Ptr<Map> _pMap;

    cfsd::Ptr<Optimizer> _pOptimizer;

    const cfsd::Ptr<CameraModel>& _pCameraModel;

    bool _verbose;

    int _minFrameInterval{0};

    double _minScore{0};

    int _solvePnP{0};

    bool _toCloseLoop{false};

    int _wait{0}; // wait until the "current frame" goes out the sliding window, to avoid concurrent updating against motion-only BA.

    int _loopFrameID{-1}, _lastLoopFrameID{-1}, _curFrameID{-1};

    std::mutex _dataMutex{};
};

} // namespace cfsd

#endif // LOOP_CLOSURE_HPP