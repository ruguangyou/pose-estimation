#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"

namespace cfsd {

Viewer::Viewer() {
    viewScale = Config::get<int>("viewScale");
    pointSize = Config::get<float>("pointSize");
    viewpointX = Config::get<float>("viewpointX");
    viewpointY = Config::get<float>("viewpointY");
    viewpointZ = Config::get<float>("viewpointZ");
    viewpointF = Config::get<float>("viewpointF");
}

void Viewer::run() {
    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue sepcific OpenGL might be needed
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const int PANEL_WIDTH = 220;

    // Add named Panel and bind to variables beginning 'menu'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));

    // Safe and efficient binding of named variables
    // Specialisations mean no conversions take place for exact types and conversions between scalar types are cheap
    pangolin::Var<bool> menuSaveWindow("menu.Save Window", false, false);
    pangolin::Var<bool> menuSaveObject("menu.Save Object", false, false);
    pangolin::Var<bool> menuShowCoordinate("menu.Show Coordinate", true, true);
    pangolin::Var<bool> menuShowRawPosition("menu.Show Raw Position", true, true);
    pangolin::Var<bool> menuShowOptimizedPosition("menu.Show Optimized Position", true, true);
    pangolin::Var<bool> menuShowLandmark("menu.Show Landmark", true, true);
    // ...
    pangolin::Var<bool> menuReset("menu.Reset", false, false);
    pangolin::Var<bool> menuExit("menu.Exit", false, false);

    // Define Camera Render Object (for view / sceno browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
        // pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisNegZ)
        pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisZ)
    );
    /*  our camera coordinate system (where AxisNegY is the up direction)
              / z (yaw)
             /
            ------ x (roll)
            |
            | y (pitch)
    */

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // Defalut hooks for existing (Esc) and fullscreen (tab)
    while (!pangolin::ShouldQuit()) {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Make the background white (default black)
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // Activate efficiently by object
        d_cam.Activate(s_cam);

        if (pangolin::Pushed(menuSaveWindow))
            pangolin::SaveWindowOnRender("window");

        if (pangolin::Pushed(menuSaveObject))
            d_cam.SaveOnRender("object");

        if (menuShowCoordinate)
            drawCoordinate();

        if (menuShowRawPosition)
            drawRawPosition();

        if (menuShowOptimizedPosition)
            drawOptimizedPosition();

        if (menuShowLandmark)
            drawLandmark();
        
        if (pangolin::Pushed(menuReset)) {
            std::lock_guard<std::mutex> lockData(dataMutex);
            std::lock_guard<std::mutex> lockRawData(rawDataMutex);
            xsOptimized.clear(); ysOptimized.clear(); zsOptimized.clear();
            xsRaw.clear(); ysRaw.clear(); zsRaw.clear();
            pointsX.clear(); pointsY.clear(); pointsZ.clear();
            readyToDrawOptimized = false; readyToDrawRaw = false;
        }

        // Swap frames and Precess Events
        pangolin::FinishFrame();

        if (pangolin::Pushed(menuExit)) break;
    }
}

void Viewer::drawCoordinate() {
    float len = 4.0f;
    glLineWidth(2);
    glBegin(GL_LINES);
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(len, 0.0f, 0.0f);

    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, len, 0.0f);
    
    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, len);
    glEnd();
}

void Viewer::pushRawPosition(const Eigen::Vector3d& p, const int& offset) {
    std::lock_guard<std::mutex> lockRawData(rawDataMutex);

    int i = idx + offset;
    if (xsRaw.size() <= i) {
        xsRaw.push_back(static_cast<float>(p(0) * viewScale));
        ysRaw.push_back(static_cast<float>(p(1) * viewScale));
        zsRaw.push_back(static_cast<float>(p(2) * viewScale));
    }
    else {
        xsRaw[i] = static_cast<float>(p(0) * viewScale);
        ysRaw[i] = static_cast<float>(p(1) * viewScale);
        zsRaw[i] = static_cast<float>(p(2) * viewScale);
    }
    // if (xsRaw.size() >= WINDOWSIZE && offset == WINDOWSIZE - 1) idx++;

    readyToDrawRaw = true;
}

void Viewer::drawRawPosition() {
    std::lock_guard<std::mutex> lockRawData(rawDataMutex);

    if (!readyToDrawRaw) return;

    glPointSize(pointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < xsRaw.size(); i++)
        glVertex3f(xsRaw[i], ysRaw[i], zsRaw[i]);
    glEnd();
}

void Viewer::pushOptimizedPosition(const Eigen::Vector3d& p, const int& offset) {
    // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]
    std::lock_guard<std::mutex> lockData(dataMutex);

    int i = idx + offset;
    if (xsOptimized.size() <= i) {
        xsOptimized.push_back(static_cast<float>(p(0) * viewScale));
        ysOptimized.push_back(static_cast<float>(p(1) * viewScale));
        zsOptimized.push_back(static_cast<float>(p(2) * viewScale));
    }
    else {
        xsOptimized[i] = static_cast<float>(p(0) * viewScale);
        ysOptimized[i] = static_cast<float>(p(1) * viewScale);
        zsOptimized[i] = static_cast<float>(p(2) * viewScale);
    }
    if (xsOptimized.size() >= WINDOWSIZE && offset == WINDOWSIZE - 1) idx++;

    readyToDrawOptimized = true;
}

void Viewer::drawOptimizedPosition() {
    std::lock_guard<std::mutex> lockData(dataMutex);

    if (!readyToDrawOptimized) return;

    glPointSize(pointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (int i = 0; i < xsOptimized.size(); i++)
        glVertex3f(xsOptimized[i], ysOptimized[i], zsOptimized[i]);
    glEnd();

    // glLineWidth(2);
    // glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
    // glBegin(GL_LINES);
    // for (int i = 0; i < xs.size(); i++)
    //     glVertex3f(xs[i], ys[i], zs[i]);
    // glEnd();
}

void Viewer::pushLandmark(const double& x, const double& y, const double& z) {
    std::lock_guard<std::mutex> lockLandmark(landmarkMutex);

    pointsX.push_back(x);
    pointsY.push_back(y);
    pointsZ.push_back(z);

    readyToDrawLandmark = true;
}

void Viewer::drawLandmark() {
    std::lock_guard<std::mutex> lockLandmark(landmarkMutex);

    if (!readyToDrawLandmark) return;

    glPointSize(pointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.0f, 1.0f);
    for (int i = 0; i < pointsX.size(); i++)
        glVertex3f(pointsX[i], pointsY[i], pointsZ[i]);
    glEnd();
}

} // namespace cfsd

#endif // USE_VIEWER