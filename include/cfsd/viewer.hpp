// Compile only if there is USE_VIEWER flag defined in compiler.
#ifdef USE_VIEWER

#ifndef VIEWER_HPP
#define VIEWER_HPP

#include "cfsd/config.hpp"
#include <vector>
#include <thread>
#include <pangolin/pangolin.h>

namespace cfsd {

class Viewer {
  public:
    Viewer();

    // Execute in an independent thread for rendering.
    void run();

    // API for Optimizer.
    void setParameters(double* rvp_i, double* rvp_j);

    void drawPosition();

    void setRawParameters(double* rvp_i, double* rvp_j);

    void drawRawPosition();

  private:
    // Set true if Optimizer pass parameters to this Viewer.
    bool readyToDraw, readyToDrawRaw;

    // Viewer settings (refer to ORB-SLAM2).
    int viewScale;
    float pointSize;
    float viewpointX, viewpointY, viewpointZ, viewpointF;

    // States.
    std::vector<float> xs, ys, zs;
    std::vector<float> xsRaw, ysRaw, zsRaw;

    std::mutex dataMutex, rawDataMutex;

};

} // namespace cfsd

#endif // VIEWER_HPP

#endif // USE_VIEWER