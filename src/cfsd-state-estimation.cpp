#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

int main(int argc, char** argv) {
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

    // If true then output verbose messages.
    const bool verbose{commandlineArguments.count("verbose") != 0};

    // Read configuration from parameter file.
    const std::string configFilePath{commandlineArguments["config"]};
    cfsd::Config::setParameterFile(configFilePath);

    // Resolution of the image read from shared memory.
    const int height = cfsd::Config::get<int>("imageHeight");
    const int width = cfsd::Config::get<int>("imageWidth");

    // Resolution that is used in post-processing, i.e. if image from shared memory is of different size, it will be resized.
    cv::Size imgSize(1344, 376);

    // Interface to VI-SLAM.
    cfsd::Ptr<cfsd::VisualInertialSLAM> pVISLAM{new cfsd::VisualInertialSLAM(verbose)};

    // Sender stamp of microservice opendlv-proxy-ellipse2n
    const int ellipseID = cfsd::Config::get<int>("ellipseID");

    // Function to call on newly arriving Envelopes which contain accelerometer data.
    auto accRecived{
        [&pVISLAM, &ellipseID] (cluon::data::Envelope &&envelope) {
            if (envelope.senderStamp() == ellipseID) {
                // std::cout << "acc" << std::endl;
                opendlv::proxy::AccelerationReading accData = cluon::extractMessage<opendlv::proxy::AccelerationReading>(std::move(envelope));
                cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
                long timestamp = cluon::time::toMicroseconds(ts);
                float accX = accData.accelerationX();
                float accY = accData.accelerationY();
                float accZ = accData.accelerationZ();
                pVISLAM->collectImuData(cfsd::SensorType::ACCELEROMETER, timestamp, accX, accY, accZ);
                // pVISLAM->processImu();

                // std::cout << "acc timestamp: " << timestamp << std::endl;
            }
        }
    };

    // Function to call on newly arriving Envelopes which contain gyroscope data.
    auto gyrRecived{
        [&pVISLAM, &ellipseID] (cluon::data::Envelope &&envelope) {
            if (envelope.senderStamp() == ellipseID) {
                // std::cout << "gyr" 
                        //   << "(elapsed: " << Ts - lastTs << "us)" << std::endl;
                opendlv::proxy::AngularVelocityReading gyrData = cluon::extractMessage<opendlv::proxy::AngularVelocityReading>(std::move(envelope));
                cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
                long timestamp = cluon::time::toMicroseconds(ts);
                float gyrX = gyrData.angularVelocityX();
                float gyrY = gyrData.angularVelocityY();
                float gyrZ = gyrData.angularVelocityZ();
                pVISLAM->collectImuData(cfsd::SensorType::GYROSCOPE, timestamp, gyrX, gyrY, gyrZ);
                // pVISLAM->processImu();

                // std::cout << "gyr timestamp: " << timestamp << std::endl;
            }
        }
    };

    // Function to call on newly arriving Envelopes which contain imageReading.
    auto imgRecived{
        [&pVISLAM] (cluon::data::Envelope &&envelope) {
            opendlv::proxy::ImageReading imgReading = cluon::extractMessage<opendlv::proxy::ImageReading>(std::move(envelope));
            cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
            long timestamp = cluon::time::toMicroseconds(ts);
            pVISLAM->setImgTimestamp(timestamp);

            std::cout << "img timestamp: " << timestamp << std::endl;
        }
    };

    // Set a delegate to be called data-triggered on arrival of a new Envelope for a given message identifier.
    od4.dataTrigger(opendlv::proxy::AccelerationReading::ID(), accRecived);
    od4.dataTrigger(opendlv::proxy::AngularVelocityReading::ID(), gyrRecived);
    od4.dataTrigger(opendlv::proxy::ImageReading::ID(), imgRecived);

    // Attach to shared memory.
    const std::string sharedMemoryName{commandlineArguments["name"]};
    std::unique_ptr<cluon::SharedMemory> pSharedMemory(new cluon::SharedMemory{sharedMemoryName});
    if (pSharedMemory && pSharedMemory->valid()) {
    // if (1) {
        std::clog << argv[0] << "attached to shared memory: '" << pSharedMemory->name() << " (" << pSharedMemory->size() << " bytes)." << std::endl;

        #ifdef USE_VIEWER
        // A thread for visulizing.
        cfsd::Ptr<cfsd::Viewer> pViewer{new cfsd::Viewer()};
        pVISLAM->setViewer(pViewer);
        std::thread viewerThread(&cfsd::Viewer::run, pViewer); // no need to detach, since there is a while loop in Viewer::run()
        #endif

        // Endless loop; end the program by pressing Ctrl-C.
        while (od4.isRunning()) {
            cv::Mat img;

            // Wait for a notification of a new frame.
            pSharedMemory->wait();
            // Lock the shared memory.
            pSharedMemory->lock();
            {
                // Copy image into cvMat structure. Be aware of that any code between lock/unlock is blocking the camera to 
                // provide the next frame. Thus, any computationally heavy algorithms should be placed outside lock/unlock.
                cv::Mat wrapped(height, width, CV_8UC4, pSharedMemory->data());
                
                // If image from shared memory has different size with the pre-defined one, resize it.
                cv::resize(wrapped, img, imgSize);

                std::cout << "shared memory timestamp: " << cluon::time::toMicroseconds(pSharedMemory->getTimeStamp().second) << std::endl;
            }
            pSharedMemory->unlock();

            cv::Mat gray;
            cv::cvtColor(img, gray, CV_BGR2GRAY);

            // pVISLAM->processImage(imgTimestamp, gray);

            #ifdef USE_VIEWER
            // ...
            #endif
        }
    }
    else {
        std::cerr << "Failed to attach to shared memory." << std::endl;
        return 1;
    }
    
    return 0;
}
