#include "cfsd/config.hpp"
#include "cfsd/visual-inertial-slam.hpp"

#include <opencv2/imgcodecs.hpp>

#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./euroc-state-estimation [config file path]" << std::endl;
        return 1;
    }

    std::string configFilePath{argv[1]};
    cfsd::Config::setParameterFile(configFilePath);
    
    std::string dataPath = cfsd::Config::get<std::string>("dataset");
    std::string imuData = dataPath + "imu0/data.csv";
    std::string imgData = dataPath + "cam0/data.csv"; // cam0/data.csv is the same as cam1/data.csv
    std::string imgLeftDataPath = dataPath + "cam0/data/";
    std::string imgRightDataPath = dataPath + "cam1/data/";

    std::string imgName;
    
    std::ifstream f_imu(imuData);
    if (!f_imu.is_open()) {
        std::cerr << "Failed to open imu data file" << std::endl;
        return 1;
    }
    std::getline(f_imu, imgName); // Remove the header in csv file first line.

    std::ifstream f_img(imgData);
    if (!f_img.is_open()) {
        std::cerr << "Failed to open image data file" << std::endl;
        return 1;
    }
    std::getline(f_img,imgName); // Remove the header in csv file first line.

    cfsd::Ptr<cfsd::VisualInertialSLAM> pVISLAM{new cfsd::VisualInertialSLAM(false)};
    
    double wx, wy, wz, ax, ay, az;
    long imuTimestamp, imgTimestamp;

    int rate = cfsd::Config::get<int>("samplingRate") / cfsd::Config::get<int>("cameraFrequency");
    int speedUp = cfsd::Config::get<int>("speedUp");
    while (!f_imu.eof() && !f_img.eof()) {
        for (int i = 0; i < speedUp*rate + 1; i++) {
            f_imu >> imuTimestamp;
            f_imu.ignore(1, ',');
            f_imu >> wx;
            f_imu.ignore(1, ',');
            f_imu >> wy;
            f_imu.ignore(1, ',');
            f_imu >> wz;
            f_imu.ignore(1, ',');
            f_imu >> ax;
            f_imu.ignore(1, ',');
            f_imu >> ay;
            f_imu.ignore(1, ',');
            f_imu >> az;

            pVISLAM->collectImuData(cfsd::SensorType::ACCELEROMETER, imuTimestamp, ax, ay, az);
            pVISLAM->collectImuData(cfsd::SensorType::GYROSCOPE, imuTimestamp, wx, wy, wz);
        }

        for (int i = 0; i < speedUp; i++) {
            f_img >> imgTimestamp;
            f_img.ignore(1, ',');
            f_img >> imgName;
        }
        cv::Mat grayL = cv::imread(imgLeftDataPath + imgName);
        cv::Mat grayR = cv::imread(imgRightDataPath + imgName);

        if (grayL.channels() == 3) {
            cv::cvtColor(grayL, grayL, CV_BGR2GRAY);
            cv::cvtColor(grayR, grayR, CV_BGR2GRAY);
        }
        else if (grayL.channels() == 4) {
            cv::cvtColor(grayL, grayL, CV_BGRA2GRAY);
            cv::cvtColor(grayR, grayR, CV_BGRA2GRAY);
        }

        if (!pVISLAM->process(grayL, grayR, imgTimestamp)) {
            std::cerr << "Error occurs in processing!" << std::endl;
            return 1;
        }
    }
    f_imu.close();
    f_img.close();
    pVISLAM->saveResults();

    std::cout << "Done!" << std::endl;

    return 0;
}