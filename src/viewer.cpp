#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"

namespace cfsd {

Viewer::Viewer() : readyToDraw(false), readyToDrawRaw(false) {
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
    // ...
    pangolin::Var<bool> menuReset("menu.Reset", false, false);
    pangolin::Var<bool> menuExit("menu.Exit", false, false);

    // Define Camera Render Object (for view / sceno browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
        pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisNegY)
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

        if (menuShowRawPosition)
            drawRawPosition();

        if (menuShowPosition)
            drawPosition();
        
        if (pangolin::Pushed(menuReset)) {
            std::lock_guard<std::mutex> lockData(dataMutex);
            std::lock_guard<std::mutex> lockRawData(rawDataMutex);
            xs.clear(); ys.clear(); zs.clear();
            xsRaw.clear(); ysRaw.clear(); zsRaw.clear();
            readyToDraw = false; readyToDrawRaw = false;
        }

        // Swap frames and Precess Events
        pangolin::FinishFrame();

        if (pangolin::Pushed(menuExit)) break;
    }
}

void Viewer::setParameters(double* rvp_i, double* rvp_j) {
    // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]
    std::lock_guard<std::mutex> lockData(dataMutex);

    if (xs.empty()) {
        xs.push_back(static_cast<float>(rvp_i[6] * viewScale));
        ys.push_back(static_cast<float>(rvp_i[7] * viewScale));
        zs.push_back(static_cast<float>(rvp_i[8] * viewScale));
    }
    else {
        xs.back() = static_cast<float>(rvp_i[6] * viewScale);
        ys.back() = static_cast<float>(rvp_i[7] * viewScale);
        zs.back() = static_cast<float>(rvp_i[8] * viewScale);
    }

    #ifdef DEBUG_IMU
    std::cout << "scaled optimized position i (scale=" << viewScale << "): " << xs.back() << ", " << ys.back() << ", " << zs.back() << std::endl;
    #endif

    xs.push_back(static_cast<float>(rvp_j[6] * viewScale));
    ys.push_back(static_cast<float>(rvp_j[7] * viewScale));
    zs.push_back(static_cast<float>(rvp_j[8] * viewScale));

    #ifdef DEBUG_IMU
    std::cout << "scaled optimized position j (scale=" << viewScale << "): " << xs.back() << ", " << ys.back() << ", " << zs.back() << std::endl;
    #endif

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

void Viewer::setRawParameters(double* rvp_i, double* rvp_j) {
    // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]
    std::lock_guard<std::mutex> lockRawData(rawDataMutex);

    if (xs.empty()) {
        xsRaw.push_back(static_cast<float>(rvp_i[6] * viewScale));
        ysRaw.push_back(static_cast<float>(rvp_i[7] * viewScale));
        zsRaw.push_back(static_cast<float>(rvp_i[8] * viewScale));
    }
    else {
        xsRaw.back() = static_cast<float>(rvp_i[6] * viewScale);
        ysRaw.back() = static_cast<float>(rvp_i[7] * viewScale);
        zsRaw.back() = static_cast<float>(rvp_i[8] * viewScale);
    }

    #ifdef DEBUG_IMU
    std::cout << "scaled raw position i (scale=" << viewScale << "): " << xsRaw.back() << ", " << ysRaw.back() << ", " << zsRaw.back() << std::endl;
    #endif

    xsRaw.push_back(static_cast<float>(rvp_j[6] * viewScale));
    ysRaw.push_back(static_cast<float>(rvp_j[7] * viewScale));
    zsRaw.push_back(static_cast<float>(rvp_j[8] * viewScale));

    #ifdef DEBUG_IMU
    std::cout << "scaled raw position j (scale=" << viewScale << "): " << xsRaw.back() << ", " << ysRaw.back() << ", " << zsRaw.back() << std::endl;
    #endif

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

} // namespace cfsd

#endif // USE_VIEWER