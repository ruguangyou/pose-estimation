// Compile only if there is USE_VIEWER flag defined in compiler.
#ifdef USE_VIEWER

#ifndef VIEWER_HPP
#define VIEWER_HPP

#include "cfsd/config.hpp"
#include <vector>
#include <thread>
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
    void pushParameters(double pose[WINDOWSIZE][6], int size);

    void drawPosition();

    void pushRawParameters(double* pose_i);

    void drawRawPosition();

    void pushLandmark(const double& x, const double& y, const double& z);

    void drawLandmark();

    void setStop();

  private:
    // Viewer settings (refer to ORB-SLAM2).
    int viewScale;
    float pointSize;
    float viewpointX, viewpointY, viewpointZ, viewpointF;

    // States (position).
    std::vector<float> xs, ys, zs;
    std::vector<float> xsRaw, ysRaw, zsRaw;
    std::vector<float> pointsX, pointsY, pointsZ;

    // Set true if Optimizer pass parameters to this Viewer.
    bool readyToDraw, readyToDrawRaw, readyToDrawLandmark;

    std::mutex dataMutex, rawDataMutex, landmarkMutex;
};

} // namespace cfsd

#endif // VIEWER_HPP

#endif // USE_VIEWER