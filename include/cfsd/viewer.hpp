// Compile only if there is USE_VIEWER flag defined in compiler.
#ifdef USE_VIEWER

#ifndef VIEWER_HPP
#define VIEWER_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/structs.hpp"

#include <pangolin/pangolin.h>

// local sliding-window size
#ifndef WINDOWSIZE
#define WINDOWSIZE 4
#endif

namespace cfsd {

class Viewer {
  public:
    Viewer();

    // Execute in an independent thread for rendering.
    void run();

    void genOpenGlMatrix(const Eigen::Matrix3f& R, const Eigen::Vector3d& p, pangolin::OpenGlMatrix& M);

    void followBody(pangolin::OpenGlRenderState& s_cam);

    void pushRawPosition(const Eigen::Vector3d& p, const int& offset);
    void pushPosition(const Eigen::Vector3d& p, const int& offset);
    void pushPose(const Eigen::Matrix3d& R);
    void pushLandmark(const std::map<size_t, cfsd::Ptr<MapPoint>>& pMPs);
    void pushLoopConnection(const int& refFrameID, const int& curFrameID);
    void pushFullBAPosition(const Eigen::Vector3d& p);
    
    void drawCoordinate();
    void drawRawPosition();
    void drawPosition();
    void drawPose(pangolin::OpenGlMatrix &M);
    void drawLandmark();
    void drawLoopConnection();
    void drawFullBAPosition();

  private:
    // Viewer settings (refer to ORB-SLAM2).
    int viewScale{0};
    float pointSize{0}, landmarkSize{0}, lineWidth{0}, cameraSize{0}, cameraLineWidth{0};
    float viewpointX{0}, viewpointY{0}, viewpointZ{0}, viewpointF{0};
    int axisDirection{0};
    int background{0};

    // States.
    std::vector<Eigen::Vector3d> positions{};
    std::vector<Eigen::Vector3d> rawPositions{};
    std::map<size_t, cfsd::Ptr<MapPoint>> pMapPoints{};
    std::vector<std::pair<int,int>> loopConnection{};
    std::vector<Eigen::Vector3d> fullBAPositions{};

    Eigen::Matrix3f pose{};
    pangolin::OpenGlMatrix T_WB{};

    // Set true if Optimizer pass parameters to this Viewer.
    bool readyToDrawPosition{false}, readyToDrawRawPosition{false};
    bool readyToDrawPose{false}, readyToDrawRawPose{false};
    bool readyToDrawLandmark{false};
    bool readyToDrawLoop{false};
    bool readyToDrawFullBAPosition{false};

    std::mutex positionMutex{}, rawPositionMutex{}, poseMutex{}, landmarkMutex{}, loopMutex{}, fullBAPositionMutex{};

    int idx{0};
};

} // namespace cfsd

#endif // VIEWER_HPP

#endif // USE_VIEWER