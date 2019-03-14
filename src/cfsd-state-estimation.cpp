#include "cfsd/config.hpp"
#include "cfsd/image-reader.hpp"
#include "cfsd/visual-inertial-slam.hpp"

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#ifdef USE_VIEWER
#include <opencv2/viz.hpp>
#endif

int main(int argc, char** argv) {
    int retCode{0};
    std::map<std::string, std::string> commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid"))  ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("config")) ) {
        std::cerr << argv[0] << " is the SLAM module for CFSD19." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OpenDaVINCI session> --name=<name of shared memory> --config=<path of configuration file> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --config: path of the configuration file" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=253 --name=img.argb --config=../config/parameters.yml --verbose" << std::endl;
        return 1;
    }

    // Interface to a running OpenDaVINCI session; here, you can send and receive messages.
    cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

    // Output verbose messages.
    const bool verbose{commandlineArguments.count("verbose") != 0};

    // Read configuration from parameter file.
    const std::string configFilePath{commandlineArguments["config"]};
    cfsd::Config::setParameterFile(configFilePath);

    // cfsd::ImageReader class reads data from shared memory.
    const std::string sharedMemoryName{commandlineArguments["name"]};
    cfsd::Ptr<cfsd::ImageReader> pImgReader{new cfsd::ImageReader(od4, sharedMemoryName, verbose)}; // OK
    // cfsd::Ptr<cfsd::ImageReader> pImgReader = new cfsd::ImageReader(od4, sharedMemoryName, verbose); // failed to compile, conversion from 'ImageReader*' to 'std::shared_ptr<ImageReader>' is invalid
    // cfsd::Ptr<cfsd::ImageReader> pImgReader = std::make_shared<cfsd::ImageReader>(od4, sharedMemoryName, verbose); // OK
    // cfsd::Ptr<cfsd::ImageReader> pImgReader{std::make_shared<cfsd::ImageReader>(od4, sharedMemoryName, verbose)}; // OK

    if (pImgReader->isSharedMemoryValid(retCode)) {
        // Interface to VI-SLAM.
        cfsd::Ptr<cfsd::VisualInertialSLAM> pVISLAM{new cfsd::VisualInertialSLAM(verbose)};

        // Sender stamp of microservice opendlv-proxy-ellipse2n
        const int ellipseID = cfsd::Config::get<int>("ellipseID");
        
        // Function to call on newly arriving Envelopes which contain accelerometer data.
        auto accRecived{
            [&pVISLAM, &ellipseID] (cluon::data::Envelope &&envelope) {
                if (envelope.senderStamp() == ellipseID) {
                    opendlv::proxy::AccelerationReading accData = cluon::extractMessage<opendlv::proxy::AccelerationReading>(std::move(envelope));
                    cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
                    long timestamp = cluon::time::toMicroseconds(ts);
                    float accX = accData.accelerationX();
                    float accY = accData.accelerationY();
                    float accZ = accData.accelerationZ();
                    pVISLAM->processImu(cfsd::SensorType::ACCELEROMETER, timestamp, accX, accY, accZ);
                }
            }
        };
    
        // Function to call on newly arriving Envelopes which contain gyroscope data.
        auto gyrRecived{
            [&pVISLAM, &ellipseID] (cluon::data::Envelope &&envelope) {
                if (envelope.senderStamp() == ellipseID) {
                    opendlv::proxy::AngularVelocityReading gyrData = cluon::extractMessage<opendlv::proxy::AngularVelocityReading>(std::move(envelope));
                    cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
                    long timestamp = cluon::time::toMicroseconds(ts);
                    float gyrX = gyrData.angularVelocityX();
                    float gyrY = gyrData.angularVelocityY();
                    float gyrZ = gyrData.angularVelocityZ();
                    pVISLAM->processImu(cfsd::SensorType::GYROSCOPE, timestamp, gyrX, gyrY, gyrZ);
                }
            }
        };

        // Set a delegate to be called data-triggered on arrival of a new Envelope for a given message identifier.
        od4.dataTrigger(opendlv::proxy::AccelerationReading::ID(), accRecived);
        od4.dataTrigger(opendlv::proxy::AngularVelocityReading::ID(), gyrRecived);

        // An independent thread for continuously reading image data from shared memory,
        // s.t. the timestamp will not depend on the running time of our algorithms.   
        std::thread imgReaderThread(&cfsd::ImageReader::run, pImgReader);
        imgReaderThread.detach(); // permits the thread to execute independently from the thread handle

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
            if (pImgReader->getQueueSize() == 0) {
                if (verbose) std::cout << "No image coming in yet. Wait..." << std::endl;
                continue;
            }
            cv::Mat img;
            long timestamp;
            pImgReader->grabData(img, timestamp);

            cv::Mat gray;
            cv::cvtColor(img, gray, CV_BGR2GRAY);

            // #ifdef DEBUG_IMG
            // cv::imshow("Gray", gray);
            // cv::waitKey(0);
            // #endif

            auto start = std::chrono::steady_clock::now();
            pVISLAM->processImage(timestamp, gray);
            auto end = std::chrono::steady_clock::now();
            std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

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
    }
    return retCode;
}
