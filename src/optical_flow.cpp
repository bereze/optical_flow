#include "optical_flow.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

/**
 * @brief Calculates the optical flow of keypoints in range
 *
 * @param range keypoints range
 */
void myslam::OpticalFLowTracker::calcOpticalFlowLK(const Range& range) {
  int iterations = 10;

  for (size_t i = range.start; i < range.end; ++i) {
    KeyPoint kp = kpts1_[i];
    double dx = 0, dy = 0;  // dx, dy need to be estimated
    if (has_initial_) {
      dx = kpts2_[i].pt.x - kp.pt.x;
      dy = kpts2_[i].pt.y - kp.pt.y;
    }

    double cost = 0., last_cost = 0.;
    bool success = true;  // indicates if the keypoint is tracked

    // Gauss-Newtion iteration
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();  // Hessian = J * Jt
    Eigen::Vector2d b = Eigen::Vector2d::Zero();  // bias = - J * R
    Eigen::Vector2d J;                            // Jacobian, J = [f_x, f_y]
    for (int iter = 0; iter < iterations; ++iter) {
      H = Eigen::Matrix2d::Zero();
      b = Eigen::Vector2d::Zero();
      cost = 0;

      // compute cost and Jacobian
      for (int x = -half_patch_size_; x < half_patch_size_; ++x) {
        for (int y = -half_patch_size_; y < half_patch_size_; ++y) {
          double error =
              getPixelValue(img2_, kp.pt.x + dx + x, kp.pt.y + dy + y) -
              getPixelValue(img1_, kp.pt.x + x, kp.pt.y + y);

          // Jacobian
          J = Eigen::Vector2d(0.5 * (getPixelValue(img2_, kp.pt.x + dx + x + 1,
                                                   kp.pt.y + dy + y) -
                                     getPixelValue(img2_, kp.pt.x + dx + x - 1,
                                                   kp.pt.y + dy + y)),
                              0.5 * (getPixelValue(img2_, kp.pt.x + dx + x,
                                                   kp.pt.y + dy + y + 1) -
                                     getPixelValue(img2_, kp.pt.x + dx + x,
                                                   kp.pt.y + dy + y - 1)));

          // H and b, and set cost:
          H += J * J.transpose();
          cost += error * error;
          b += -J * error;
        }
      }

      // update dx, dy
      Eigen::Vector2d update = H.ldlt().solve(b);

      if (isnan(update.x())) {
        // sometimes occurred when we have a black or white patch and H is
        // irreversible
        success = false;
        break;
      }

      if (iter > 0 && cost > last_cost) {
        break;
      }

      dx += update.x();
      dy += update.y();
      last_cost = cost;
      success = true;

      if (update.norm() < 0.01) {
        // converge
        break;
      }
    }

    status_[i] = success;

    // set kp2
    kpts2_[i].pt = kp.pt + Point2f(dx, dy);
  }
}

/**
 * @brief Calculates the optical flow from img1 to img2
 *
 * @param img1 the previous image
 * @param img2 the current image
 * @param kpts1 the keypoints in img1
 * @param kpts2 the tracked keypoints in img2 to output
 * @param status true if a keypoint is tracked successfully
 * @param half_patch_size the half patch size to calculate LK optical flow
 */
void myslam::calcOpticalFlowPyrLK(const Mat& img1, const Mat& img2,
                                  const vector<KeyPoint>& kpts1,
                                  vector<KeyPoint>& kpts2, vector<bool>& status,
                                  int half_patch_size) {
  // create pyramids
  int pyramids = 4;
  double pyramid_scale = 0.5;
  double scales[] = {1.0, 0.5, 0.25, 0.125};

  vector<Mat> pyr1, pyr2;
  for (int i = 0; i < pyramids; ++i) {
    if (i == 0) {
      pyr1.push_back(img1);
      pyr2.push_back(img2);
    } else {
      Mat img1_pyr, img2_pyr;
      cv::resize(pyr1[i - 1], img1_pyr,
                 cv::Size(pyr1[i - 1].cols * pyramid_scale,
                          pyr1[i - 1].rows * pyramid_scale));
      cv::resize(pyr2[i - 1], img2_pyr,
                 cv::Size(pyr1[i - 1].cols * pyramid_scale,
                          pyr2[i - 1].rows * pyramid_scale));
      pyr1.push_back(img1_pyr);
      pyr2.push_back(img2_pyr);
    }
  }

  // coarse-to-fine LK tracking in pyramids
  for (int level = pyramids - 1; level >= 0; level--) {
    vector<KeyPoint> kp1_pyr;
    for (int i = 0; i < kpts1.size(); ++i) {
      KeyPoint pt_pyr = kpts1[i];
      pt_pyr.pt = pt_pyr.pt * scales[level];
      kp1_pyr.push_back(pt_pyr);
    }

    if (level == pyramids - 1)
      calcOpticalFLowSingle(pyr1[level], pyr2[level], kp1_pyr, kpts2, status,
                            false);
    else
      calcOpticalFLowSingle(pyr1[level], pyr2[level], kp1_pyr, kpts2, status,
                            true);

    if (level > 0) {
      for (int i = 0; i < kpts2.size(); ++i) {
        kpts2[i].pt = kpts2[i].pt * 2;
      }
    }

    // Mat img2_self;
    // cv::cvtColor(pyr2[level], img2_self, CV_GRAY2BGR);
    // for (int i = 0; i < kpts2.size(); ++i) {
    //   // cout << status_single[i] << endl;
    //   if (status[i]) {
    //     cv::circle(img2_self, kpts2[i].pt, 2, cv::Scalar(0, 255, 0), 2);
    //     cv::line(img2_self, kp1_pyr[i].pt, kpts2[i].pt, cv::Scalar(0, 255,
    //     0));
    //   }
    // }
    // namedWindow("level", 0);
    // imshow("level", img2_self);
    // waitKey(0);
  }
}

void myslam::calcOpticalFLowSingle(const Mat& img1, const Mat& img2,
                                   const vector<KeyPoint>& kpts1,
                                   vector<KeyPoint>& kpts2,
                                   vector<bool>& status, bool has_initial,
                                   int half_patch_size) {
  kpts2.resize(kpts1.size());
  status.resize(kpts1.size());
  OpticalFLowTracker tracker(img1, img2, kpts1, kpts2, status, half_patch_size,
                             has_initial);
  // tracker.calcOpticalFlowLK(Range(0, kpts1.size()));
  cv::parallel_for_(Range(0, kpts1.size()),
                    std::bind(&OpticalFLowTracker::calcOpticalFlowLK, &tracker,
                              placeholders::_1));
}
