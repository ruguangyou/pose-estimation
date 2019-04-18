#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

#include <opencv2/imgcodecs.hpp>

// #include <iostream>
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
    std::string imgLeftDataPath = dataPath + "image_02/data/";
    std::string imgRightDataPath = dataPath + "image_00/data/";

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
    
    #ifdef USE_VIEWER
    // A thread for visulizing.
    cfsd::Ptr<cfsd::Viewer> pViewer{new cfsd::Viewer()};
    pVISLAM->setViewer(pViewer);
    std::thread viewerThread(&cfsd::Viewer::run, pViewer); // no need to detach, since there is a while loop in Viewer::run()
    #endif
    
    double ax, ay, az, wx, wy, wz;
    long timestamp;
    for (int num_imu = 0; num_imu < 1000; num_imu++) {
        std::string count;
        if (num_imu < 10) count = "000" + std::to_string(num_imu);
        else if (num_imu < 100) count = "00" + std::to_string(num_imu);
        else if (num_imu < 1000) count = "0" + std::to_string(num_imu);
        else count = std::to_string(num_imu);
        std::ifstream f_imudata(imuDataPath + "000000" + count + ".txt");
        if (!f_imuTimestamp.is_open()) {
            std::cerr << "Failed to open imu data file" << std::endl;
            return 1;
        }
        f_imudata >> ax >> ay >> az >> wx >> wy >> wz;
        f_imuTimestamp >> timestamp;
        pVISLAM->collectImuData(cfsd::SensorType::ACCELEROMETER, timestamp, ax, ay, az);
        pVISLAM->collectImuData(cfsd::SensorType::GYROSCOPE, timestamp, wx, wy, wz);
    }

    long imgTimestamp;
    for (int num_img = 0; num_img < 90; num_img++) {
        std::string count;
        if (num_img < 10) count = "000" + std::to_string(num_img);
        else if (num_img < 100) count = "00" + std::to_string(num_img);
        else if (num_img < 1000) count = "0" + std::to_string(num_img);
        else count = std::to_string(num_img);
        cv::Mat grayL = cv::imread(imgLeftDataPath + "000000" + count + ".png");
        cv::Mat grayR = cv::imread(imgRightDataPath + "000000" + count + ".png");
        f_imgTimestamp >> imgTimestamp;

        if (!pVISLAM->process(grayL, grayR, imgTimestamp)) {
            std::cerr << "Error occurs in processing!" << std::endl;
            return 1;
        }
        std::cout << "number of image: " << num_img << std::endl;
        if (num_img == 99)
            num_img = 99;
    }
    std::cout << "Done!" << std::endl;
    viewerThread.join();

    return 0;
}