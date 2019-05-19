#ifdef USE_VIEWER
#include "cfsd/viewer.hpp"

namespace cfsd {

Viewer::Viewer() {
    viewScale = Config::get<int>("viewScale");
    pointSize = Config::get<float>("pointSize");
    landmarkSize = Config::get<float>("landmarkSize");
    lineWidth = Config::get<float>("lineWidth");
    cameraSize = Config::get<float>("cameraSize");
    cameraLineWidth = Config::get<float>("cameraLineWidth");
    viewpointX = Config::get<float>("viewpointX");
    viewpointY = Config::get<float>("viewpointY");
    viewpointZ = Config::get<float>("viewpointZ");
    viewpointF = Config::get<float>("viewpointF");
    axisDirection = Config::get<int>("axisDirection");
    background = Config::get<float>("background");
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
    pangolin::Var<bool> menuFollowBody("menu.Follow Body", true, true);
    pangolin::Var<bool> menuShowCoordinate("menu.Show Coordinate", true, true);
    pangolin::Var<bool> menuShowRawPosition("menu.Show Raw Position", false, true);
    pangolin::Var<bool> menuShowPosition("menu.Show Position", true, true);
    pangolin::Var<bool> menuShowPose("menu.Show Pose", true, true);
    pangolin::Var<bool> menuShowLandmark("menu.Show Landmark", true, true);
    pangolin::Var<bool> menuShowLoopConnection("menu.Show Loop Connection", true, true);
    // ...
    pangolin::Var<bool> menuReset("menu.Reset", false, false);
    pangolin::Var<bool> menuExit("menu.Exit", false, false);

    // Define Camera Render Object (for view / sceno browsing)
    pangolin::OpenGlRenderState s_cam;
    switch (axisDirection) {
        case 0:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisNone)
            );
            break;
        case 1:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisNegX)
            );
            break;
        case 2:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisX)
            );
            break;
        case 3:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisNegY)
            );
            break;
        case 4:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisY)
            );
            break;
        case 5:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisNegZ)
            );
            break;
        case 6:
            s_cam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024,768, viewpointF,viewpointF, 512,389, 0.1,1000),
                pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0,0,0, pangolin::AxisZ)
            );
    }

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    T_WB.SetIdentity();

    // Defalut hooks for existing (Esc) and fullscreen (tab)
    while (!pangolin::ShouldQuit()) {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Make the background white (default black)
        if (background) glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // Activate efficiently by object
        d_cam.Activate(s_cam);

        if (pangolin::Pushed(menuSaveWindow))
            pangolin::SaveWindowOnRender("window");

        if (pangolin::Pushed(menuSaveObject))
            d_cam.SaveOnRender("object");

        if (menuFollowBody)
            followBody(s_cam);

        if (menuShowCoordinate)
            drawCoordinate();

        if (menuShowRawPosition)
            drawRawPosition();

        if (menuShowPosition)
            drawPosition();

        if (menuShowPose)
            drawPose(T_WB);

        if (menuShowLandmark)
            drawLandmark();

        if (menuShowLoopConnection)
            drawLoopConnection();
        
        if (pangolin::Pushed(menuReset)) {
            std::lock_guard<std::mutex> lockPosition(positionMutex);
            std::lock_guard<std::mutex> lockRawPosition(rawPositionMutex);
            xs.clear(); ys.clear(); zs.clear();
            xsRaw.clear(); ysRaw.clear(); zsRaw.clear();
            frameAndPoints.clear();
            readyToDrawPosition = false; readyToDrawRawPosition = false;
        }

        // Swap frames and Precess Events
        pangolin::FinishFrame();

        if (pangolin::Pushed(menuExit)) break;
    }
}

void Viewer::drawCoordinate() {
    float len = 1.0f;
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

void Viewer::genOpenGlMatrix(const Eigen::Matrix3f& R, const float& x, const float& y, const float& z, pangolin::OpenGlMatrix& M) {
    M.m[0] = R(0,0);
    M.m[1] = R(1,0);
    M.m[2] = R(2,0);
    M.m[3] = 0.0;
    M.m[4] = R(0,1);
    M.m[5] = R(1,1);
    M.m[6] = R(2,1);
    M.m[7] = 0.0;
    M.m[8] = R(0,2);
    M.m[9] = R(1,2);
    M.m[10] = R(2,2);
    M.m[11] = 0.0;
    M.m[12] = x;
    M.m[13] = y;
    M.m[14] = z;
    M.m[15] = 1.0;
}

void Viewer::followBody(pangolin::OpenGlRenderState& s_cam) {
    if (readyToDrawPose && readyToDrawPosition) {
        std::lock_guard<std::mutex> lockPose(poseMutex);
        std::lock_guard<std::mutex> lockPosition(positionMutex);
        genOpenGlMatrix(pose, xs.back(), ys.back(), zs.back(), T_WB);
    }
    s_cam.Follow(T_WB);
}

void Viewer::pushRawPosition(const Eigen::Vector3d& p, const int& offset) {
    std::lock_guard<std::mutex> lockRawPosition(rawPositionMutex);

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

    readyToDrawRawPosition = true;
}

void Viewer::pushPosition(const Eigen::Vector3d& p, const int& offset) {
    // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]
    std::lock_guard<std::mutex> lockPosition(positionMutex);

    int i = idx + offset;
    if (xs.size() <= i) {
        xs.push_back(static_cast<float>(p(0) * viewScale));
        ys.push_back(static_cast<float>(p(1) * viewScale));
        zs.push_back(static_cast<float>(p(2) * viewScale));
    }
    else {
        xs[i] = static_cast<float>(p(0) * viewScale);
        ys[i] = static_cast<float>(p(1) * viewScale);
        zs[i] = static_cast<float>(p(2) * viewScale);
    }
    if (xs.size() >= WINDOWSIZE && offset == WINDOWSIZE - 1) idx++;

    readyToDrawPosition = true;
}

void Viewer::pushPose(const Eigen::Matrix3d& R) {
    std::lock_guard<std::mutex> lockPose(poseMutex);
    
    pose = R.cast<float>();

    readyToDrawPose = true;
}

void Viewer::pushLandmark(const int& frameID, const Eigen::Vector3d& point) {
    std::lock_guard<std::mutex> lockLandmark(landmarkMutex);

    if (frameAndPoints.size() < frameID+1)
        frameAndPoints.push_back(std::vector<Eigen::Vector3d>());
    frameAndPoints[frameID].push_back(point);

    readyToDrawLandmark = true;
}

void Viewer::pushLoopConnection(const int& refFrameID, const int& curFrameID) {
    std::lock_guard<std::mutex> lockLoop(loopMutex);

    loopConnection.push_back(std::make_pair(refFrameID, curFrameID));

    readyToDrawLoop = true;
}

void Viewer::drawRawPosition() {
    std::lock_guard<std::mutex> lockRawPosition(rawPositionMutex);

    if (!readyToDrawRawPosition) return;

    glColor3f(0.6f, 0.2f, 0.2f);
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for (int i = 0; i < xsRaw.size(); i++)
        glVertex3f(xsRaw[i], ysRaw[i], zsRaw[i]);
    glEnd();

    glLineWidth(lineWidth);
    glBegin(GL_LINES);
    glVertex3f(xsRaw[0], ysRaw[0], zsRaw[0]);
    for (int i = 0; i < xsRaw.size(); i++) {
        glVertex3f(xsRaw[i], ysRaw[i], zsRaw[i]);
        glVertex3f(xsRaw[i], ysRaw[i], zsRaw[i]);
    }
    glVertex3f(xsRaw.back(), ysRaw.back(), zsRaw.back());
    glEnd();
}

void Viewer::drawPosition() {
    std::lock_guard<std::mutex> lockPosition(positionMutex);

    if (!readyToDrawPosition) return;

    int n = xs.size()-WINDOWSIZE;
    if (n < 0) n = xs.size();

    glPointSize(pointSize);
    glColor3f(0.2f, 0.6f, 0.2f);
    glBegin(GL_POINTS);
    for (int i = 0; i < n; i++)
        glVertex3f(xs[i], ys[i], zs[i]);
    glEnd();

    glPointSize(pointSize+4);
    glColor3f(0.8f, 0.1f, 0.1f);
    glBegin(GL_POINTS);
    for (int i = n; i < xs.size(); i++)
        glVertex3f(xs[i], ys[i], zs[i]);
    glEnd();

    glLineWidth(lineWidth);
    glColor3f(0.2f, 0.6f, 0.2f);
    glBegin(GL_LINES);
    glVertex3f(xs[0], ys[0], zs[0]);
    for (int i = 1; i < xsRaw.size() - 1; i++) {
        glVertex3f(xs[i], ys[i], zs[i]);
        glVertex3f(xs[i], ys[i], zs[i]);
    }
    glVertex3f(xs.back(), ys.back(), zs.back());
    glEnd();
}

void Viewer::drawPose(pangolin::OpenGlMatrix &M) {
    std::lock_guard<std::mutex> lockPose(poseMutex);

    if (!readyToDrawPose) return;

    const float &w = cameraSize;
    const float h = w*0.75;
    const float z = w*0.5;

    glPushMatrix();

    #ifdef HAVE_GLES
    glMultMatrixf(M.m);
    #else
    glMultMatrixd(M.m);
    #endif

    glLineWidth(cameraLineWidth);
    glColor3f(0.6f,0.2f,0.2f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void Viewer::drawLandmark() {
    std::lock_guard<std::mutex> lockLandmark(landmarkMutex);

    if (!readyToDrawLandmark) return;

    int n = frameAndPoints.size()-WINDOWSIZE;
    if (n < 0) n = frameAndPoints.size();

    glPointSize(landmarkSize);
    glBegin(GL_POINTS);
    glColor3f(0.2f, 0.2f, 0.6f);
    for (int i = 0; i < n; i++) {
        std::vector<Eigen::Vector3d>& points = frameAndPoints[i];
        for (int j = 0; j < points.size(); j++)
            glVertex3f(points[j].x(), points[j].y(), points[j].z());
    }
    glEnd();

    glPointSize(landmarkSize+2);
    glBegin(GL_POINTS);
    glColor3f(0.8f, 0.1f, 0.1f);
    for (int i = n; i < frameAndPoints.size(); i++){
        std::vector<Eigen::Vector3d>& points = frameAndPoints[i];
        for (int j = 0; j < points.size(); j++)
            glVertex3f(points[j].x(), points[j].y(), points[j].z());
    }
    glEnd();
}

void Viewer::drawLoopConnection() {
    std::lock_guard<std::mutex> lockLoop(loopMutex);

    if (!readyToDrawLoop) return;

    glLineWidth(lineWidth);
    glBegin(GL_LINES);
    glColor3f(0.2f, 0.4f, 0.4f);
    for (int i = 0; i < loopConnection.size(); i++) {
        int idx1 = loopConnection[i].first;
        int idx2 = loopConnection[i].second;
        glVertex3f(xs[idx1], ys[idx1], zs[idx1]);
        glVertex3f(xs[idx2], ys[idx2], zs[idx2]);
    }
    glEnd();
}

void Viewer::resetIdx() {
    idx = 0;
}

} // namespace cfsd

#endif // USE_VIEWER