// Compile only if there is USE_VIEWER flag defined in compiler.
#ifdef USE_VIEWER

#ifndef VIEWER_HPP
#define VIEWER_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"

#include <pangolin/pangolin.h>

// local sliding-window size
#define WINDOWSIZE 6

namespace cfsd {

class Viewer {
  public:
    Viewer();

    // Execute in an independent thread for rendering.
    void run();

    void drawCoordinate();

    // Push and draw.
    void pushRawPosition(const Eigen::Vector3d& p, const int& offset);
    void drawRawPosition();

    void pushOptimizedPosition(const Eigen::Vector3d& p, const int& offset);
    void drawOptimizedPosition();

    void pushLandmark(const double& x, const double& y, const double& z);
    void drawLandmark();

    void setStop();

  private:
    // Viewer settings (refer to ORB-SLAM2).
    int viewScale{0};
    float pointSize{0}, landmarkSize{0}, lineWidth{0};
    float viewpointX{0}, viewpointY{0}, viewpointZ{0}, viewpointF{0};
    int background{0};

    // States (position).
    std::vector<float> xsOptimized, ysOptimized, zsOptimized;
    std::vector<float> xsRaw, ysRaw, zsRaw;
    std::vector<float> pointsX, pointsY, pointsZ;

    // Set true if Optimizer pass parameters to this Viewer.
    bool readyToDrawOptimized{false};
    bool readyToDrawRaw{false};
    bool readyToDrawLandmark{false};

    std::mutex dataMutex, rawDataMutex, landmarkMutex;

    int idx{0};
};

} // namespace cfsd

#endif // VIEWER_HPP

#endif // USE_VIEWER