#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"

namespace cfsd {

Viewer::Viewer() : readyToDraw(false), readyToDrawRaw(false), readyToDrawLandmark(false) {
    viewScale = Config::get<int>("viewScale");
    pointSize = Config::get<float>("pointSize");
    viewpointX = Config::get<float>("viewpointX");
    viewpointY = Config::get<float>("viewpointY");
    viewpointZ = Config::get<float>("viewpointZ");
    viewpointF = Config::get<float>("viewpointF");

    xs.push_back(0);
    ys.push_back(0);
    zs.push_back(0);
    xsRaw.push_back(0);
    ysRaw.push_back(0);
    zsRaw.push_back(0);
}

void Viewer::run() {
    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue sepcific OpenGL might be needed
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const int PANEL_WIDTH = 175;

    // Add named Panel and bind to variables beginning 'menu'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));

    // Safe and efficient binding of named variables
    // Specialisations mean no conversions take place for exact types and conversions between scalar types are cheap
    pangolin::Var<bool> menuSaveWindow("menu.Save Window", false, false);
    pangolin::Var<bool> menuSaveObject("menu.Save Object", false, false);
    pangolin::Var<bool> menuShowRawPosition("menu.Show Raw Position", true, true);
    pangolin::Var<bool> menuShowPosition("menu.Show Position", true, true);
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

        drawCoordinate();

        if (pangolin::Pushed(menuSaveWindow))
            pangolin::SaveWindowOnRender("window");

        if (pangolin::Pushed(menuSaveObject))
            d_cam.SaveOnRender("object");

        if (menuShowRawPosition)
            drawRawPosition();

        if (menuShowPosition)
            drawPosition();

        if (menuShowLandmark)
            drawLandmark();
        
        if (pangolin::Pushed(menuReset)) {
            std::lock_guard<std::mutex> lockData(dataMutex);
            std::lock_guard<std::mutex> lockRawData(rawDataMutex);
            xs.clear(); ys.clear(); zs.clear();
            xsRaw.clear(); ysRaw.clear(); zsRaw.clear();
            pointsX.clear(); pointsY.clear(); pointsZ.clear();
            readyToDraw = false; readyToDrawRaw = false;
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

void Viewer::pushParameters(double pose[WINDOWSIZE][6], int size) {
    // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]
    std::lock_guard<std::mutex> lockData(dataMutex);

    for (int i = 0; i < size-1; i++) {
        int idx = xs.size() - size + i;
        if (idx < 0) idx = 0;
        xs[idx] = static_cast<float>(pose[i][3] * viewScale);
        ys[idx] = static_cast<float>(pose[i][4] * viewScale);
        zs[idx] = static_cast<float>(pose[i][5] * viewScale);
    }
    xs.push_back(static_cast<float>(pose[size-1][3] * viewScale));
    ys.push_back(static_cast<float>(pose[size-1][4] * viewScale));
    zs.push_back(static_cast<float>(pose[size-1][5] * viewScale));

    readyToDraw = true;
}

void Viewer::drawPosition() {
    std::lock_guard<std::mutex> lockData(dataMutex);

    if (!readyToDraw) return;

    glPointSize(pointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (int i = 0; i < xs.size(); i++)
        glVertex3f(xs[i], ys[i], zs[i]);
    glEnd();

    // glLineWidth(2);
    // glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
    // glBegin(GL_LINES);
    // for (int i = 0; i < xs.size(); i++)
    //     glVertex3f(xs[i], ys[i], zs[i]);
    // glEnd();
}

void Viewer::pushRawParameters(double* pose_i) {
    // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]
    std::lock_guard<std::mutex> lockRawData(rawDataMutex);

    xsRaw.push_back(static_cast<float>(pose_i[3] * viewScale));
    ysRaw.push_back(static_cast<float>(pose_i[4] * viewScale));
    zsRaw.push_back(static_cast<float>(pose_i[5] * viewScale));

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