#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: ./cvRectify [camera parameters yml] [image0] [image2]" << std::endl;
        return 1;
    }

    // cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    // cv::Mat K1, D1, R1, P1, K2, D2, R2, P2, rvec, R, T, Q;

    // kitti
    // fs["K_00"] >> K2;
    // fs["D_00"] >> D2;
    // // fs["R_rect_00"] >> R1;
    // // fs["P_rect_00"] >> P1;
    // fs["K_02"] >> K1;
    // fs["D_02"] >> D1;
    // // fs["R_rect_02"] >> R2;
    // // fs["P_rect_02"] >> P2;
    // fs["R_02"] >> R;
    // fs["T_02"] >> T;

    // cfsd
    // fs["camLeft"] >> K1;
    // fs["distLeft"] >> D1;
    // fs["camRight"] >> K2;
    // fs["distRight"] >> D2;
    // fs["rotationLeftToRight"] >> rvec;
    // fs["translationLeftToRight"] >> T;
    // fs.release();

    // cv::Rodrigues(rvec, R);

    // std::cout << "R:\n" << R << std::endl;
    
    cv::Size s(672, 376);
    // cv::Mat color1 = cv::imread(argv[2]);
    // cv::Mat color2 = cv::imread(argv[3]);
    // cv::Mat img1, img2;
    // cv::cvtColor(color1, img1, CV_BGR2GRAY);
    // cv::cvtColor(color2, img2, CV_BGR2GRAY);
    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    cv::resize(img1, img1, s);
    cv::resize(img2, img2, s);
    cv::Size imageSize = img1.size();

    // cv::Rect validRoi[2];
    // // cv::transpose(R, R);
    // cv::stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);

    // cv::Mat roi1 = img1(validRoi[0]);
    // std::cout << "roi size: " << roi1.size() << std::endl;
    // cv::imshow("roi", roi1);

    // std::cout << "K1:\n" << K1 << std::endl
    //           << "K2:\n" << K2 << std::endl
    //           << "R1:\n" << R1 << std::endl
    //           << "R2:\n" << R2 << std::endl
    //           << "P1:\n" << P1 << std::endl
    //           << "P2:\n" << P2 << std::endl
    //           << "Q:\n" << Q << std::endl;

    cv::Mat rmap[2][2];
    // cv::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    // cv::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    cv::Mat P1, P2;
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    fs["rmap00"] >> rmap[0][0];
    fs["rmap01"] >> rmap[0][1];
    fs["rmap10"] >> rmap[1][0];
    fs["rmap11"] >> rmap[1][1];
    fs["P1"] >> P1;
    fs["P2"] >> P2;

    // cv::FileStorage fw("cfsdRectified.yml", cv::FileStorage::WRITE);
    // fw << "K1" << K1;
    // fw << "D1" << D1;
    // fw << "R1" << R1;
    // fw << "P1" << P1;
    // fw << "K2" << K2;
    // fw << "D2" << D2;
    // fw << "R2" << R2;
    // fw << "P2" << P2;
    // fw << "rmap00" << rmap[0][0];
    // fw << "rmap01" << rmap[0][1];
    // fw << "rmap10" << rmap[1][0];
    // fw << "rmap11" << rmap[1][1];
    // fw.release();

    int rowT = 190, rowB = 300;

    cv::Mat img;
    cv::hconcat(img1, img2, img);
    cv::imshow("Original image", img);
    cv::waitKey(0);

    auto start = std::chrono::steady_clock::now();
    cv::Mat rimg1, rimg2;
    // cur-left and cur-right
    // cv::remap(img1, rimg1, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
    // cv::remap(img2, rimg2, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
    // hist-left and cur-left
    cv::remap(img1, rimg1, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
    cv::remap(img2, rimg2, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
    auto end = std::chrono::steady_clock::now();
    std::cout << "rimg size: " << rimg1.size() << std::endl;
    std::cout << "remap elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    cv::Mat rimg;
    cv::hconcat(rimg1, rimg2, rimg);
    cv::imshow("Undistort and rectify", rimg);
    cv::waitKey(0);

    // start = std::chrono::steady_clock::now();
    // cv::Mat uimg1, uimg2;
    // cv::undistort(img1, uimg1, K1, D1);
    // cv::undistort(img2, uimg2, K2, D2);
    // end = std::chrono::steady_clock::now();
    // std::cout << "udistort elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    cv::line(rimg1, cv::Point(0,rowT), cv::Point(672,rowT), cv::Scalar(0,0,0), 2);
    cv::line(rimg1, cv::Point(0,rowB), cv::Point(672,rowB), cv::Scalar(0,0,0), 2);
    // cv::imshow("rectify", rimg1);
    cv::line(rimg2, cv::Point(0,rowT), cv::Point(672,rowT), cv::Scalar(0,0,0), 2);
    cv::line(rimg2, cv::Point(0,rowB), cv::Point(672,rowB), cv::Scalar(0,0,0), 2);
    // cv::imshow("rectify", rimg2);

    // try ORB detection and triangulation.
    cv::Mat mask = cv::Mat::zeros(rimg1.size(), CV_8U);
    for (int i = rowT; i < rowB; i++)
        for (int j = 0; j < rimg1.cols; j++)
            mask.at<char>(i, j) = 255;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(200);

    start = std::chrono::steady_clock::now();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(rimg1, mask, keypoints1, descriptors1);
    orb->detectAndCompute(rimg2, mask, keypoints2, descriptors2);
    end = std::chrono::steady_clock::now();
    std::cout << "orb detection elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    start = std::chrono::steady_clock::now();
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    end = std::chrono::steady_clock::now();
    std::cout << "BF match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    // Draw keypoints.
    cv::Mat key1, key2, key;
    cv::drawKeypoints(rimg1, keypoints1, key1);
    cv::drawKeypoints(rimg2, keypoints2, key2);
    cv::hconcat(key1, key2, key);
    cv::imwrite("keypoints.png", key);
    cv::imshow("Keypoints", key);
    cv::waitKey(0);

    // Draw matches.
    cv::Mat img_matches;
    cv::drawMatches(rimg1, keypoints1, rimg2, keypoints2, matches, img_matches);
    cv::imwrite("matches.png", img_matches);
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);

    float maxDist = 0; float minDist = 10000;
    for (int i = 0; i < keypoints1.size(); ++i) {
        float dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }
    std::cout << "maxDist: " << maxDist << ", minDist: " << minDist << std::endl;

    start = std::chrono::steady_clock::now();
    std::vector<cv::Point2d> pixels1, pixels2;
    std::vector<cv::DMatch> good_matches;
    // Only keep good matches (i.e. whose distance is less than matchRatio * minDist, or a small arbitary value (e.g. 30.0f) in case min_dist is very small.
    for (auto& m : matches) {
        if (m.distance < std::max(2.0f * minDist, 30.0f)) {
        // if (m.distance < std::max(3.0f * minDist, 30.0f) && std::abs(keypoints1[m.queryIdx].pt.y - keypoints2[m.trainIdx].pt.y) < 0.1) {
            good_matches.push_back(m);
            pixels1.push_back(keypoints1[m.queryIdx].pt);
            pixels2.push_back(keypoints2[m.trainIdx].pt);
        }
    }
    end = std::chrono::steady_clock::now();
    std::cout << "find good matches using distance elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    start = std::chrono::steady_clock::now();
    cv::Mat ransacMask;
    cv::findFundamentalMat(pixels1, pixels2, ransacMask);
    std::vector<cv::Point2d> ransac_pixels1, ransac_pixels2;
    int good_count = 0;
    for (int i = 0; i < ransacMask.rows; i++) {
        if (ransacMask.at<bool>(i)) {
            // std::cout << "left pixel: " << pixels1[i] << std::endl;
            // std::cout << "right pixel: " << pixels2[i] << std::endl;
            ransac_pixels1.push_back(pixels1[i]);
            ransac_pixels2.push_back(pixels2[i]);

            std::cout << good_count << ": left pixel: " << pixels1[i] << std::endl;
            std::cout << good_count << ": right pixel: " << pixels2[i] << std::endl;
            good_count++;
        }
        else {
            std::cout << "outlier: " << pixels1[i] << ", " << pixels2[i] << std::endl;
        }
    }
    end = std::chrono::steady_clock::now();
    std::cout << "find good matches using ransac elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
    std::cout << "number of pixels after RANSAC: " << ransac_pixels1.size() << ", " << ransac_pixels2.size() << std::endl;

    // Draw only good matches.
    cv::Mat img_good_matches;
    cv::drawMatches(rimg1, keypoints1, rimg2, keypoints2, good_matches, img_good_matches);
    cv::imwrite("good_matches.png", img_good_matches);
    cv::imshow("Good Matches", img_good_matches);
    std::cout << "Left-Right matches: " << good_matches.size() << std::endl;
    std::cout << "Left-Right matches after RANSAC: " << good_count << std::endl;
    cv::waitKey(0);

    start = std::chrono::steady_clock::now();
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, ransac_pixels1, ransac_pixels2, points4D);
    end = std::chrono::steady_clock::now();
    std::cout << "triangulatePoints elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    start = std::chrono::steady_clock::now();
    std::vector<cv::Point3d> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        points3D.push_back(cv::Point3d(points4D.at<double>(0,i) / points4D.at<double>(3,i),
                                       points4D.at<double>(1,i) / points4D.at<double>(3,i),
                                       points4D.at<double>(2,i) / points4D.at<double>(3,i)));
        std::cout << i << ": " << points3D[i] << std::endl;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "convert from homogeneous elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    return 0;
}