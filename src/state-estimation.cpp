#include "cfsd/visual-inertial-odometry.hpp"
#include "cfsd/config.hpp"

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#ifdef USE_VIEWER
#include <opencv2/viz.hpp>
#endif

#include <chrono>

int main(int argc, char** argv) {
    std::map<std::string, std::string> commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ) {
        std::cerr << argv[0] << " is the SLAM module for CFSD19." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OpenDaVINCI session> --name=<name of shared memory> --file=<path of parameter file> [--verbose] [--debug]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --file:   path of the parameter file" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=253 --name=img.argb" << std::endl;
        return 1;
    }
    const bool VERBOSE{commandlineArguments.count("verbose") != 0};
    const bool DEBUG{commandlineArguments.count("debug") != 0};

    const std::string sharedMemoryName{commandlineArguments["name"]};
    std::unique_ptr<cluon::SharedMemory> sharedMemory(new cluon::SharedMemory{sharedMemoryName});
    if (sharedMemory && sharedMemory->valid()) {
        std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;
            
        // Interface to a running OpenDaVINCI session; here, you can send and receive messages.
        cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

        // auto onDistance = [&distancesMutex, &front, &rear, &left, &right](cluon::data::Envelope &&env){
        //     auto senderStamp = env.senderStamp();
        //     // Now, we unpack the cluon::data::Envelope to get the desired DistanceReading.
        //     opendlv::proxy::DistanceReading dr = cluon::extractMessage<opendlv::proxy::DistanceReading>(std::move(env));

        //     // Store distance readings.
        //     std::lock_guard<std::mutex> lck(distancesMutex);
        //     switch (senderStamp) {
        //         case 0: front = dr.distance(); break;
        //         case 2: rear = dr.distance(); break;
        //         case 1: left = dr.distance(); break;
        //         case 3: right = dr.distance(); break;
        //     }
        // };
        // Finally, we register our lambda for the message identifier for opendlv::proxy::DistanceReading.
        // od4.dataTrigger(opendlv::proxy::DistanceReading::ID(), onDistance);

        // Read configuration from parameter file
        const std::string parameterFile{commandlineArguments["file"]};
        cfsd::Config::setParameterFile(parameterFile);
        int HEIGHT = cfsd::Config::get<int>("height");
        int WIDTH = cfsd::Config::get<int>("width");
        if (DEBUG) {
            std::cout << "Image resolution: " << WIDTH << "x" << HEIGHT << std::endl;
        }

        // Interface to VIO
        cfsd::VisualInertialOdometry::Ptr vio = cfsd::VisualInertialOdometry::create(VERBOSE, DEBUG);

        #ifdef USE_VIEWER
            // visualization
            cv::viz::Viz3d viewer("VIO");
            cv::viz::WCoordinateSystem worldCoor(1.0), camCoor(0.5); // 1.0 and 0.5 are scale that determine the size of axes
            cv::Point3d viewerPosition(-1.0,-3.0,-3.0), viewerFocalPoint(0,0,0), viewerYDirection(-1.0,1.0,-1.0);
            cv::Affine3d viewerPose = cv::viz::makeCameraPose(viewerPosition, viewerFocalPoint, viewerYDirection);
            viewer.setViewerPose(viewerPose); // set pose of the viewer
            
            worldCoor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
            camCoor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
            viewer.showWidget("World", worldCoor);
            viewer.showWidget("Camera", camCoor);
        #endif

        // Endless loop; end the program by pressing Ctrl-C.
        while (od4.isRunning()) {
            cv::Mat img;
            cluon::data::TimeStamp TS;
            long timestamp;
            // Wait for a notification of a new frame.
            sharedMemory->wait();
            // Lock the shared memory.
            sharedMemory->lock();
            {
                // Copy image into cvMat structure.
                // Be aware of that any code between lock/unlock is blocking
                // the camera to provide the next frame. Thus, any computationally
                // heavy algorithms should be placed outside lock/unlock.
                cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
                img = wrapped.clone();

                TS = cluon::time::now();
                timestamp = cluon::time::toMicroseconds(TS);
            }
            sharedMemory->unlock();

            // Display image.
            if (DEBUG) {
                cv::imshow(sharedMemory->name().c_str(), img);
                cv::waitKey(1);
            }

            // TODO: Do something with the frame.
            // Example: Draw a red rectangle and display image.
            // cv::rectangle(img, cv::Point(50, 50), cv::Point(100, 100), cv::Scalar(0,0,255));
            auto start = std::chrono::steady_clock::now();
            vio->processFrame(timestamp, img);
            auto end = std::chrono::steady_clock::now();
            if (VERBOSE) { 
                std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
            }

            #ifdef USE_VIEWER
                SophusSE3Type Tcw = vio->getLatestCamPose().inverse();
                // show the map and the camera pose 
                cv::Affine3d M(
                    cv::Affine3d::Mat3( 
                        Tcw.so3().matrix()(0,0), Tcw.so3().matrix()(0,1), Tcw.so3().matrix()(0,2),
                        Tcw.so3().matrix()(1,0), Tcw.so3().matrix()(1,1), Tcw.so3().matrix()(1,2),
                        Tcw.so3().matrix()(2,0), Tcw.so3().matrix()(2,1), Tcw.so3().matrix()(2,2)
                    ), 
                    cv::Affine3d::Vec3(
                        Tcw.translation()(0,0), Tcw.translation()(1,0), Tcw.translation()(2,0)
                    )
                );
                viewer.setWidgetPose("Camera", M);
                viewer.spinOnce(1, true);
            #endif
        }
        return 0;
    }
    else {
        std::cerr << "Failed to attach to shared memory." << std::endl;
        return 1;
    }
}
