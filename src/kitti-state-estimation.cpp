#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

#include <opencv2/imgcodecs.hpp>

#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./kitti-state-estimation [config file path]" << std::endl;
        return 1;
    }

    std::string configFilePath{argv[1]};
    cfsd::Config::setParameterFile(configFilePath);
    
    std::string dataPath = cfsd::Config::get<std::string>("dataset");
    std::string imuTimestampFile = dataPath + "oxts/processed/timestamps.txt";
    std::string imuDataPath = dataPath + "oxts/processed/";
    
    std::string imgTimestampFile = dataPath + "image_00/processed_timestamps.txt";
    std::string imgLeftDataPath = dataPath + "image_00/data/";
    std::string imgRightDataPath = dataPath + "image_01/data/";

    std::ifstream f_imuTimestamp(imuTimestampFile);
    if (!f_imuTimestamp.is_open()) {
        std::cerr << "Failed to open imu timestamps.txt" << std::endl;
        return 1;
    }

    std::ifstream f_imgTimestamp(imgTimestampFile);
    if (!f_imgTimestamp.is_open()) {
        std::cerr << "Failed to open image timestamps.txt" << std::endl;
        return 1;
    }

    cfsd::Ptr<cfsd::VisualInertialSLAM> pVISLAM{new cfsd::VisualInertialSLAM(false)};
    
    double ax, ay, az, wx, wy, wz;
    long imuTimestamp, imgTimestamp;

    int maxNumImu = cfsd::Config::get<int>("maxNumImu");
    int maxNumImage = cfsd::Config::get<int>("maxNumImage");
    int rate = cfsd::Config::get<int>("samplingRate") / cfsd::Config::get<int>("cameraFrequency");

    int numImu = 0, numImage = 0;
    while (numImu < maxNumImu && numImage < maxNumImage) {
        std::string count;

        // Read imu measurements.
        for (int i = 0; i < rate+1; i++) {
            if (numImu < 10) count = "0000" + std::to_string(numImu);
            else if (numImu < 100) count = "000" + std::to_string(numImu);
            else if (numImu < 1000) count = "00" + std::to_string(numImu);
            else if (numImu < 10000) count = "0" + std::to_string(numImu);
            else count = std::to_string(numImu);
            std::ifstream f_imudata(imuDataPath + "00000" + count + ".txt");
            if (!f_imuTimestamp.is_open()) {
                if (numImu >= maxNumImu) {
                    std::cout << "All imu measurements have been read" << std::endl;
                    return 0;
                }
                else {
                    std::cerr << "Failed to open imu data file" << std::endl;
                    return 1;
                }
            }
            f_imudata >> ax >> ay >> az >> wx >> wy >> wz;
            f_imuTimestamp >> imuTimestamp;
            pVISLAM->collectImuData(cfsd::SensorType::ACCELEROMETER, imuTimestamp, ax, ay, az);
            pVISLAM->collectImuData(cfsd::SensorType::GYROSCOPE, imuTimestamp, wx, wy, wz);
            numImu++;
        }

        // Read images.
        if (numImage < 10) count = "000" + std::to_string(numImage);
        else if (numImage < 100) count = "00" + std::to_string(numImage);
        else if (numImage < 1000) count = "0" + std::to_string(numImage);
        else count = std::to_string(numImage);
        cv::Mat grayL = cv::imread(imgLeftDataPath + "000000" + count + ".png");
        cv::Mat grayR = cv::imread(imgRightDataPath + "000000" + count + ".png");
        if (grayL.channels() == 3) {
            cv::cvtColor(grayL, grayL, CV_BGR2GRAY);
            cv::cvtColor(grayR, grayR, CV_BGR2GRAY);
        }
        else if (grayL.channels() == 4) {
            cv::cvtColor(grayL, grayL, CV_BGRA2GRAY);
            cv::cvtColor(grayR, grayR, CV_BGRA2GRAY);
        }
        f_imgTimestamp >> imgTimestamp;
        numImage++;

        // Process.
        if (!pVISLAM->process(grayL, grayR, imgTimestamp)) {
            std::cerr << "Error occurs in processing!" << std::endl;
            return 1;
        }
    }
    
    std::cout << "Done!" << std::endl;
    viewerThread.join();

    return 0;
}