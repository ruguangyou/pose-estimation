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
    const int height = cfsd::Config::get<int>("readHeight");
    const int width = cfsd::Config::get<int>("readWidth");

    // Resolution that is used in post-processing, i.e. if image from shared memory is of different size, it will be resized.
    const int resizeHeight = cfsd::Config::get<int>("imageHeight");
    const int resizeWidth = cfsd::Config::get<int>("imageWidth") * 2;
    cv::Size imgSize(resizeWidth, resizeHeight);

    // Time interval in microseconds between two image frames.
    // const long imgDeltaTus = (long)1000000 / cfsd::Config::get<int>("cameraFrequency");

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
                pVISLAM->collectImuData(cfsd::SensorType::ACCELEROMETER, timestamp, accX, accY, accZ);
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
                pVISLAM->collectImuData(cfsd::SensorType::GYROSCOPE, timestamp, gyrX, gyrY, gyrZ);
            }
        }
    };

    // Sleep for .. seconds, wait for sensor initialization (e.g. camera adjusts its optical parameters).
    // using namespace std::chrono_literals;
    // auto start = std::chrono::steady_clock::now();
    // std::this_thread::sleep_for(2s);
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "Wait " << std::chrono::duration<double, std::milli>(end-start).count() << " ms" << std::endl;

    // Set a delegate to be called data-triggered on arrival of a new Envelope for a given message identifier.
    od4.dataTrigger(opendlv::proxy::AngularVelocityReading::ID(), gyrRecived);
    od4.dataTrigger(opendlv::proxy::AccelerationReading::ID(), accRecived);

    // Attach to shared memory.
    const std::string sharedMemoryName{commandlineArguments["name"]};
    std::unique_ptr<cluon::SharedMemory> pSharedMemory(new cluon::SharedMemory{sharedMemoryName});
    if (pSharedMemory && pSharedMemory->valid()) {
        std::clog << argv[0] << " attached to shared memory: '" << pSharedMemory->name() << " (" << pSharedMemory->size() << " bytes)." << std::endl << std::endl;

        // Endless loop; end the program by pressing Ctrl-C.
        long imgTimestamp = 0;
        while (od4.isRunning()) {
            cv::Mat img;

            // Wait for a notification of a new frame.
            pSharedMemory->wait();
            // Lock the shared memory.
            pSharedMemory->lock();
            {
                imgTimestamp = cluon::time::toMicroseconds(pSharedMemory->getTimeStamp().second);

                // Copy image into cvMat structure. Be aware of that any code between lock/unlock is blocking the camera to 
                // provide the next frame. Thus, any computationally heavy algorithms should be placed outside lock/unlock.
                cv::Mat wrapped(height, width, CV_8UC4, pSharedMemory->data());
                // If image from shared memory has different size with the pre-defined one, resize it.
                cv::resize(wrapped, img, imgSize);
            }
            pSharedMemory->unlock();

            // Split image into left and right.
            cv::Mat grayL, grayR;
            if (img.channels() == 3) {
                cv::cvtColor(img(cv::Rect(0, 0, img.cols/2, img.rows)), grayL, CV_BGR2GRAY);
                cv::cvtColor(img(cv::Rect(img.cols/2, 0, img.cols/2, img.rows)), grayR, CV_BGR2GRAY);
            }
            else if (img.channels() == 4) {
                cv::cvtColor(img(cv::Rect(0, 0, img.cols/2, img.rows)), grayL, CV_BGRA2GRAY);
                cv::cvtColor(img(cv::Rect(img.cols/2, 0, img.cols/2, img.rows)), grayR, CV_BGRA2GRAY);
            }
                        
            if (!pVISLAM->process(grayL, grayR, imgTimestamp)) {
                std::cerr << "Error occurs in processing!" << std::endl;
                return 1;
            }
        }
    }
    else {
        std::cerr << "Failed to attach to shared memory." << std::endl;
        return 1;
    }
    return 0;
}