#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace myslam {
class OpticalFLowTracker {
 public:
  OpticalFLowTracker(const Mat& img1, const Mat& img2,
                     const vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2,
                     vector<bool>& status, int half_patch_size,
                     bool has_initial)
      : img1_(img1),
        img2_(img2),
        kpts1_(kpts1),
        kpts2_(kpts2),
        status_(status),
        half_patch_size_(half_patch_size),
        has_initial_(has_initial) {}

  void calcOpticalFlowLK(const Range& range);

 private:
  const Mat& img1_;
  const Mat& img2_;
  const vector<KeyPoint>& kpts1_;
  vector<KeyPoint>& kpts2_;
  vector<bool>& status_;
  int half_patch_size_;
  bool has_initial_;
};

void calcOpticalFlowPyrLK(const Mat& img1, const Mat& img2,
                          const vector<KeyPoint>& kpts1,
                          vector<KeyPoint>& kpts2, vector<bool>& status,
                          int half_patch_size = 4);

void calcOpticalFLowSingle(const Mat& img1, const Mat& img2,
                           const vector<KeyPoint>& kpts1,
                           vector<KeyPoint>& kpts2, vector<bool>& status,
                           bool has_initial = false, int half_patch_size = 4);

inline float getPixelValue(const Mat& img, float x, float y) {
  // boundary check
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= img.cols) x = img.cols - 1;
  if (y >= img.rows) y = img.rows - 1;

  uchar* data = &img.data[(int)y * img.step + (int)x];
  float xx = x - floor(x);
  float yy = y - floor(y);

  // p(xx, yy) = p_11 * (x2-xx) * (y2-yy) + p_21 * (xx-x1) * (y2-yy) +
  //             p_12 * (x2-xx) * (yy-y1) + p_22 * (xx-x1) * (xx-x1)
  return float(data[0] * (1 - xx) * (1 - yy) + data[1] * xx * (1 - yy) +
               data[img.step] * (1 - xx) * yy + data[img.step + 1] * xx * yy);
}
}  // namespace myslam

#endif