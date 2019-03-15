#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


cv::Mat camL = (cv::Mat_<float>(3,3) << 342.791722, 0.000000, 334.706838, 0.000000, 344.124357, 196.550958, 0.000000, 0.000000, 1.000000);
cv::Mat camR = (cv::Mat_<float>(3,3) << 341.251536, 0.000000, 349.920843, 0.000000, 342.367160, 198.743062, 0.000000, 0.000000, 1.000000);
cv::Mat L2R = (cv::Mat_<float>(3,4) << 0.999980, 0.000561, -0.006365, -0.113323,
                                      -0.000626, 0.999948, -0.010216,  0.000475,
                                       0.006359, 0.010220,  0.999928, -0.003484);


void triangulate(cv::Mat& imgL,
                 cv::Mat& imgR,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors,
                 std::vector<cv::Point3f>& points3D) {
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    orb->detectAndCompute(imgL, cv::noArray(), keypointsL, descriptorsL);
    orb->detectAndCompute(imgR, cv::noArray(), keypointsR, descriptorsR);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsL, descriptorsR, matches);
    float min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2){ return m1.distance < m2.distance; })->distance;
    std::cout << "[LEFT-RIGHT] min_dist: " << min_dist << std::endl;
    std::vector<cv::Point2f> goodPointsL, goodPointsR;
    for (cv::DMatch& m : matches) {
        if (m.distance < std::max(min_dist*2, 30.0f)) {
            keypoints.push_back(keypointsL[m.queryIdx]);
            if (descriptors.rows == 0) { descriptors = descriptorsL.row(m.queryIdx); }
            else { cv::vconcat(descriptors, descriptorsL.row(m.queryIdx), descriptors); }
            goodPointsL.push_back(keypointsL[m.queryIdx].pt);
            goodPointsR.push_back(keypointsR[m.trainIdx].pt);
        }
    }
    std::cout << "[LEFT-RIGHT] number of matched points: " << goodPointsL.size() << std::endl;

    cv::Mat projL = camL * (cv::Mat_<float>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
    cv::Mat projR = camR * L2R;
    // std::cout << "projL:\n" << projL << "\nprojR:\n" << projR << std::endl;
    // cv::Mat p = (cv::Mat_<float>(4,1) << 1,1,1,1);
    // std::cout << "point (1,1,1,1) projected to left image: " << projL*p << std::endl;
    // std::cout << "point (1,1,1,1) projected to right image: " << projR*p << std::endl;

    cv::Mat homogeneous4D; // 4xN array
    cv::triangulatePoints(projL, projR, goodPointsL, goodPointsR, homogeneous4D);
    for (int i = 0; i < homogeneous4D.cols; ++i) {
        // std::cout << "left image points: " << goodPointsL[i] << "\nright image points: " << goodPointsR[i] << std::endl;
        // std::cout << homogeneous4D.col(i) << std::endl;
        float sign = (homogeneous4D.at<float>(2,i) / homogeneous4D.at<float>(3,i)) > 0 ? 1.0 : -1.0;
        points3D.push_back(sign * cv::Point3f(homogeneous4D.at<float>(0,i) / homogeneous4D.at<float>(3,i),
                                              homogeneous4D.at<float>(1,i) / homogeneous4D.at<float>(3,i),
                                              homogeneous4D.at<float>(2,i) / homogeneous4D.at<float>(3,i)));
        // std::cout << points3D[i] << std::endl;
    }
    std::cout << "[LEFT-RIGHT] number of triangulated 3D points: " << points3D.size() << std::endl;
}

void computePose(cv::Mat& imgPrev,
                 cv::Mat& imgNext,
                 std::vector<cv::KeyPoint>& keypointsPrev,
                 cv::Mat& descriptorsPrev,
                 std::vector<cv::Point3f>& points3DPrev) {
    std::vector<cv::KeyPoint> keypointsNext;
    cv::Mat descriptorsNext;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    
    orb->detectAndCompute(imgNext, cv::noArray(), keypointsNext, descriptorsNext);
    
    std::vector<cv::DMatch> matches, good_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsPrev, descriptorsNext, matches);
    float min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2){ return m1.distance < m2.distance; })->distance;
    std::cout << "[PREV-NEXT] min_dist: " << min_dist << std::endl;
    for (cv::DMatch& m : matches) {
        if (m.distance < std::max(min_dist*1.5f, 30.0f)) {
            good_matches.push_back(m);
        }
    }

    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;
    for (cv::DMatch& m : good_matches) {
        points3D.push_back(points3DPrev[m.queryIdx]);
        points2D.push_back(keypointsNext[m.trainIdx].pt);
    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(points3D, points2D, camL, cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    std::cout << "[PREV-NEXT] number of matched points: " << points2D.size() << std::endl;
    std::cout << "[PREV-NEXT] number of inliers: " << inliers.rows << std::endl;
    std::cout << "rvec: " << std::endl << rvec << std::endl << "tvec: " << std::endl << tvec << std::endl << std::endl;

    // visulization matching results
    // cv::Mat img_match, img_goodmatch;
    // cv::drawMatches(imgPrev, keypointsPrev, imgNext, keypointsNext, matches, img_match);
    // cv::drawMatches(imgPrev, keypointsPrev, imgNext, keypointsNext, good_matches, img_goodmatch);
    // cv::imshow("All matches", img_match);
    // cv::imshow("Good matches", img_goodmatch);
    // cv::waitKey(0);
}

int main (int argc, char **argv) {
    if (argc != 5) {
        std::cout << "usage: feature_extraction img_1 img_2 img_11 num_iteration" << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "2_left.jpg")) {
        std::cout << "ok" << std::endl;
    }
    else {
        std::cout << "no" << std::endl;
    }

    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_11 = cv::imread(argv[3], CV_LOAD_IMAGE_COLOR);

    img_1 = img_1(cv::Range(img_1.rows/2,img_1.rows/9*8), cv::Range(0,img_1.cols));
    img_2 = img_2(cv::Range(img_2.rows/2,img_2.rows/9*8), cv::Range(0,img_2.cols));
    img_11 = img_11(cv::Range(img_11.rows/2,img_11.rows/9*8), cv::Range(0,img_11.cols));

    int n = std::stoi(argv[4]);
    double runtime = 0;
    for (int i = 0; i < n; ++i) {
        auto start = std::chrono::steady_clock::now();
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<cv::Point3f> points3D;
        triangulate(img_1, img_2, keypoints, descriptors, points3D);
        computePose(img_1, img_11, keypoints, descriptors, points3D);
        
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        runtime += std::chrono::duration<double, std::milli>(diff).count();
    }

    std::cout << "Average runtime (" << n << " iters): " << runtime / n << "ms" << std::endl;
    return 0;
}