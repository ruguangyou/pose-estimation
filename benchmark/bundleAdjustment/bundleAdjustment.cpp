#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ceres/ceres.h>

struct PoseParameterization : public ceres::LocalParameterization {
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // input: pose [r,p]
        Eigen::Map<const Eigen::Vector3d> r(x);
        Eigen::Map<const Eigen::Vector3d> p(x+3);

        Eigen::Map<const Eigen::Vector3d> dr(delta);
        Eigen::Map<const Eigen::Vector3d> dp(delta+3);

        Eigen::Map<Eigen::Vector3d> rr(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> pp(x_plus_delta+3);

        rr = Sophus::SO3d::exp(r+dr).log();
        pp = p + dp;

        return true;
    }

    virtual bool ComputeJacobian (const double* x, double* jacobian) const {
        Eigen::Map<Eigen::Matrix<double,6,6>> J(jacobian);
        J.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 6; }

    virtual int LocalSize() const { return 6; }
};

struct ReprojectCostFunction {
    ReprojectCostFunction(const cv::Point3d& point3D, const cv::Point2d& pixel, const cv::Mat& P1) { 
    
        point(0) = point3D.x;
        point(1) = point3D.y;
        point(3) = point3D.z;
        u = pixel.x;
        v = pixel.y;
        cv::cv2eigen(P1, P);
    }

    template<typename T>
    bool operator() (const T* const pose_j, T* residuals) const {
        Eigen::Matrix<T,3,1> r_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Matrix<T,3,1> p_j(pose_j[3], pose_j[4], pose_j[5]);
        // T_WB = (r_j, p_j), is transformation from body to world frame.
        // T_BW = T_WB.inverse()
        Sophus::SE3<T> T_BW = Sophus::SE3<T>(Sophus::SO3<T>::exp(r_j), p_j).inverse();

        Eigen::Matrix<T,4,1> point_homo;
        point_homo << T(point(0)), T(point(1)), T(point(2)), T(1);
        Eigen::Matrix<T,3,1> pixel_homo;
        pixel_homo = P * point_homo;

        residuals[0] = pixel_homo(0)/pixel_homo(2) - T(u);
        residuals[1] = pixel_homo(1)/pixel_homo(2) - T(u);
        
        return true;
    }

  private:
    // 3D landmark point is w.r.t world frame.
    Eigen::Vector3d point;
    
    // Pixel coordinates.
    double u, v;
    
    Eigen::Matrix<double,3,4> P;
};

void motionOnlyBA(const std::vector<cv::Point3d>& points, const std::vector<cv::Point2d>& pixels, const cv::Mat& P1) {
    double pose_j[6] = {0,0,0,0,0,0};

    // Build the problem.
    ceres::Problem problem;

    // Add parameter block with parameterization.
    ceres::LocalParameterization* poseParameterization = new PoseParameterization();
    problem.AddParameterBlock(pose_j, 6, poseParameterization);

    // Set up reprojection cost function (a.k.a. residuals).
    for (int i = 0; i < points.size(); i++) {
        // Reproject to left image.
        ceres::CostFunction* reprojectCost = new ceres::AutoDiffCostFunction<ReprojectCostFunction, 2, 6>(new ReprojectCostFunction(points[i], pixels[i], P1));
        problem.AddResidualBlock(reprojectCost, nullptr, pose_j);
    }

    // Set the solver.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    // Run the solver.
    ceres::Solve(options, &problem, &summary);
    
    // if (_verbose) {
        // Show the report.
        std::cout << summary.FullReport() << std::endl;
    // }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cout << "Usage: ./cvRectify [number of iteration] [camera parameters yml] [image-left0] [image-right0] [image-left1]" << std::endl;
        return 1;
    }

    double remapT = 0;
    double orbT = 0;
    double matchT = 0;
    double distT = 0;
    double ransacT = 0;
    double triT = 0;
    double homoT = 0;
    double baT = 0;

    cv::FileStorage fs(argv[2], cv::FileStorage::READ);
    cv::Mat K1, D1, R1, P1, K2, D2, R2, P2, rvec, R, T, Q;
    fs["camLeft"] >> K1;
    fs["distLeft"] >> D1;
    fs["camRight"] >> K2;
    fs["distRight"] >> D2;
    fs["rotationLeftToRight"] >> rvec;
    fs["translationLeftToRight"] >> T;
    fs.release();

    cv::Rodrigues(rvec, R);

    cv::Size s(672, 376);
    cv::Mat img1 = cv::imread(argv[3]);
    cv::Mat img2 = cv::imread(argv[4]);
    cv::Mat img3 = cv::imread(argv[5]);
    cv::resize(img1, img1, s);
    cv::resize(img2, img2, s);
    cv::resize(img3, img3, s);
    cv::Size imageSize = img1.size();

    cv::Rect validRoi[2];
    // cv::transpose(R, R);
    cv::stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);

    std::cout << P1 << std::endl;
    std::cout << P2 << std::endl;

    cv::Mat rmap[2][2];
    cv::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    int rowT = 210, rowB = 300;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);

    int iters = std::atoi(argv[1]);
    for (int i = 0; i < iters; i++) {
        start = std::chrono::steady_clock::now();
        cv::Mat rimg1, rimg2;
        cv::remap(img1, rimg1, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
        cv::remap(img2, rimg2, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
        end = std::chrono::steady_clock::now();
        remapT += std::chrono::duration<double, std::milli>(end-start).count();

        // cv::line(rimg1, cv::Point(0,rowT), cv::Point(672,rowT), cv::Scalar(0,0,255), 2);
        // cv::line(rimg1, cv::Point(0,rowB), cv::Point(672,rowB), cv::Scalar(0,0,255), 2);
        // cv::imshow("rectify", rimg1);

        // try ORB detection and triangulation.
        cv::Mat mask = cv::Mat::zeros(rimg1.size(), CV_8U);
        for (int i = rowT; i < rowB; i++)
            for (int j = 0; j < rimg1.cols; j++)
                mask.at<char>(i, j) = 255;
        start = std::chrono::steady_clock::now();
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        orb->detectAndCompute(rimg1, mask, keypoints1, descriptors1);
        orb->detectAndCompute(rimg2, mask, keypoints2, descriptors2);
        end = std::chrono::steady_clock::now();
        orbT += std::chrono::duration<double, std::milli>(end-start).count();

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        start = std::chrono::steady_clock::now();
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);
        end = std::chrono::steady_clock::now();
        matchT += std::chrono::duration<double, std::milli>(end-start).count();
        // std::cout << "Descriptor size: " << descriptors1.size() << std::endl;

        float maxDist = 0; float minDist = 10000;
        for (int i = 0; i < keypoints1.size(); ++i) {
            float dist = matches[i].distance;
            if (dist < minDist) minDist = dist;
            if (dist > maxDist) maxDist = dist;
        }
        // std::cout << "maxDist: " << maxDist << ", minDist: " << minDist << std::endl;

        start = std::chrono::steady_clock::now();
        std::vector<cv::Point2d> pixels1, pixels2;
        std::vector<cv::DMatch> good_matches;
        cv::Mat descriptors11;
        // Only keep good matches (i.e. whose distance is less than matchRatio * minDist, or a small arbitary value (e.g. 30.0f) in case min_dist is very small.
        for (auto& m : matches) {
            // if (m.distance < std::max(3.0f * minDist, 30.0f)) {
            if (m.distance < std::max(3.0f * minDist, 30.0f) && std::abs(keypoints1[m.queryIdx].pt.y - keypoints2[m.trainIdx].pt.y) < 0.05) {
                good_matches.push_back(m);
                pixels1.push_back(keypoints1[m.queryIdx].pt);
                pixels2.push_back(keypoints2[m.trainIdx].pt);
                descriptors11.push_back(descriptors1.row(m.queryIdx));
            }
        }
        end = std::chrono::steady_clock::now();
        distT += std::chrono::duration<double, std::milli>(end-start).count();

        start = std::chrono::steady_clock::now();
        cv::Mat ransacMask;
        cv::findFundamentalMat(pixels1, pixels2, ransacMask);
        std::vector<cv::Point2d> ransac_pixels1, ransac_pixels2;
        cv::Mat ransac_descriptors1;
        int good_count = 0;
        for (int i = 0; i < ransacMask.rows; i++) {
            if (ransacMask.at<bool>(i)) {
                // std::cout << "left pixel: " << pixels1[i] << std::endl;
                // std::cout << "right pixel: " << pixels2[i] << std::endl;
                ransac_pixels1.push_back(pixels1[i]);
                ransac_pixels2.push_back(pixels2[i]);
                ransac_descriptors1.push_back(descriptors11.row(i));
                // std::cout << good_count << ": left pixel: " << pixels1[i] << std::endl;
                // std::cout << good_count << ": right pixel: " << pixels2[i] << std::endl;
                good_count++;
            }
            else {
                // std::cout << "outlier: " << pixels1[i] << ", " << pixels2[i] << std::endl;
            }
        }
        end = std::chrono::steady_clock::now();
        ransacT += std::chrono::duration<double, std::milli>(end-start).count();
        // std::cout << "number of pixels after RANSAC: " << ransac_pixels1.size() << ", " << ransac_pixels2.size() << std::endl;

        // Draw only good matches.
        // cv::Mat img_matches;
        // cv::drawMatches(rimg1, keypoints1, rimg2, keypoints2, good_matches, img_matches);
        // cv::imshow("Left-Right Good Matches", img_matches);
        // std::cout << "Left-Right matches: " << good_matches.size() << std::endl;
        // std::cout << "Left-Right matches after RANSAC: " << good_count << std::endl;
        // cv::waitKey(0);

        start = std::chrono::steady_clock::now();
        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, ransac_pixels1, ransac_pixels2, points4D);
        end = std::chrono::steady_clock::now();
        triT += std::chrono::duration<double, std::milli>(end-start).count();

        start = std::chrono::steady_clock::now();
        std::vector<cv::Point3d> points3D;
        for (int i = 0; i < points4D.cols; i++) {
            points3D.push_back(cv::Point3d(points4D.at<double>(0,i) / points4D.at<double>(3,i),
                                        points4D.at<double>(1,i) / points4D.at<double>(3,i),
                                        points4D.at<double>(2,i) / points4D.at<double>(3,i)));
            // std::cout << i << ": " << points3D[i] << std::endl;
        }
        end = std::chrono::steady_clock::now();
        homoT += std::chrono::duration<double, std::milli>(end-start).count();


        // cv::Mat rimg3;
        // cv::remap(img3, rimg3, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
        // std::vector<cv::KeyPoint> keypoints3;
        // cv::Mat descriptors3;
        // orb->detectAndCompute(rimg3, mask, keypoints3, descriptors3);
        // std::vector<cv::DMatch> matches13;
        // matcher.match(descriptors3, ransac_descriptors1, matches13);

        // std::vector<cv::Point2d> pixels;
        // std::vector<cv::Point3d> points;
        // for (auto& m : matches13) {
        //     if (m.distance < std::max(3.0f * minDist, 30.0f)) {
        //         pixels.push_back(keypoints3[m.queryIdx].pt);
        //         points.push_back(points3D[m.trainIdx]);
        //     }
        // }

        // todo..................
        // start = std::chrono::steady_clock::now();
        // motionOnlyBA(points, pixels, P1);
        // end = std::chrono::steady_clock::now();
        // baT += std::chrono::duration<double, std::milli>(end-start).count();
    }

    std::cout << std::endl << "number of iterations: " << iters << std::endl << std::endl;
    std::cout << "average elapsed time: " << std::endl << std::endl;
    std::cout << "  remap (undistort and rectiry): " << remapT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "  orb detection: " << orbT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "  BF match: " << matchT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "  find good matches using matching distance: " << distT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "  find good matches using ransac: " << ransacT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "  triangulatePoints: " << triT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "  convert from homogeneous4D to point3D: " << homoT / (double)iters << "ms" << std::endl << std::endl;
    std::cout << "sum: " << (remapT+orbT+matchT+distT+ransacT+triT+homoT) / (double)iters << "ms, FPS: " << (double)iters / (remapT+orbT+matchT+distT+ransacT+triT+homoT) * 1000 << std::endl << std::endl;
    std::cout << "  motion-only BA elapsed time: " << baT / (double)iters << "ms" << std::endl << std::endl;

    return 0;
}