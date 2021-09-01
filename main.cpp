#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "src/optical_flow.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: ./optical_flow first.png second.png" << endl;
    return -1;
  }

  string file1 = argv[1];
  string file2 = argv[2];

  Mat img1 = imread(file1, IMREAD_GRAYSCALE);
  Mat img2 = imread(file2, IMREAD_GRAYSCALE);

  // Keypoint, using GFTT
  vector<KeyPoint> kp1;
  Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
  detector->detect(img1, kp1);

  // single optical flow
  vector<KeyPoint> kp2_single;
  vector<bool> status_single;
  chrono::steady_clock::time_point start = chrono::steady_clock::now();
  myslam::calcOpticalFLowSingle(img1, img2, kp1, kp2_single, status_single,
                                false);
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  auto time_used = chrono::duration_cast<chrono::duration<double>>(end - start);
  cout << "optical flow single: " << time_used.count() << endl;

  // track in the second image
  vector<KeyPoint> kp2_self;
  vector<bool> status_self;
  start = chrono::steady_clock::now();
  myslam::calcOpticalFlowPyrLK(img1, img2, kp1, kp2_self, status_self);
  end = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(end - start);
  cout << "optical flow by gauss-newton: " << time_used.count() << endl;

  // opencv's optical flow
  vector<Point2f> pt1_cv, pt2_cv;
  for (auto& kp : kp1) pt1_cv.push_back(kp.pt);
  vector<uchar> status_cv;
  vector<float> error;

  start = chrono::steady_clock::now();
  cv::calcOpticalFlowPyrLK(img1, img2, pt1_cv, pt2_cv, status_cv, error);
  end = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(end - start);
  cout << "optical flow by opencv: " << time_used.count() << endl;

  // plot
  Mat img2_single;
  cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
  for (int i = 0; i < kp2_single.size(); ++i) {
    // cout << status_single[i] << endl;
    if (status_single[i]) {
      cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 255, 0), 2);
      cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 255, 0));
    }
  }

  Mat img2_self;
  cv::cvtColor(img2, img2_self, CV_GRAY2BGR);
  for (int i = 0; i < kp2_self.size(); ++i) {
    // cout << status_self[i] << endl;
    if (status_self[i]) {
      cv::circle(img2_self, kp2_self[i].pt, 2, cv::Scalar(0, 255, 0), 2);
      cv::line(img2_self, kp1[i].pt, kp2_self[i].pt, cv::Scalar(0, 255, 0));
    }
  }

  Mat img2_cv;
  cv::cvtColor(img2, img2_cv, CV_GRAY2BGR);
  for (int i = 0; i < pt2_cv.size(); ++i) {
    if (status_cv[i]) {
      cv::circle(img2_cv, pt2_cv[i], 2, cv::Scalar(0, 255, 0), 2);
      cv::line(img2_cv, pt1_cv[i], pt2_cv[i], cv::Scalar(0, 255, 0));
    }
  }

  cv::imshow("tracked single", img2_single);
  cv::imshow("tracked self", img2_self);
  cv::imshow("tracked by opencv", img2_cv);
  cv::waitKey(0);
  return 0;
}
