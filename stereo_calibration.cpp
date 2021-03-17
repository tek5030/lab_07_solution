#include "stereo_calibration.h"
#include "opencv2/opencv.hpp"

StereoCalibration::StereoCalibration(const std::string& intrinsic_filename,
                                     const std::string& extrinsic_filename,
                                     const cv::Size& img_size)
    : img_size_{img_size}
{
  cv::FileStorage fs(intrinsic_filename, cv::FileStorage::READ);
  if (!fs.isOpened())
  {
    std::stringstream message;
    message << "Failed to open file " << intrinsic_filename;
    throw std::runtime_error{message.str()};
  }

  fs["M1"] >> K_left_;
  fs["D1"] >> distortion_left_;
  fs["M2"] >> K_right_;
  fs["D2"] >> distortion_right_;

  fs.open(extrinsic_filename, cv::FileStorage::READ);
  if (!fs.isOpened())
  {
    std::stringstream message;
    message << "Failed to open file " << extrinsic_filename;
    throw std::runtime_error{message.str()};
  }

  fs["R"] >> R_;
  fs["T"] >> t_;

  computeRectificationMapping();
}

StereoCalibration::StereoCalibration(const tek5030::RealSense::StereoCamera& stereo_camera)
{
  using stream = tek5030::RealSense::StereoCamera::CameraStream;
  cv::Mat(stereo_camera.K(stream::LEFT)).convertTo(K_left_, CV_64F);
  cv::Mat(stereo_camera.K(stream::RIGHT)).convertTo(K_right_, CV_64F);
  cv::Mat(stereo_camera.distortion(stream::LEFT)).convertTo(distortion_left_, CV_64F);
  cv::Mat(stereo_camera.distortion(stream::RIGHT)).convertTo(distortion_right_, CV_64F);
  cv::Mat(stereo_camera.pose().rotation()).convertTo(R_, CV_64F);
  cv::Mat(stereo_camera.pose().translation()).convertTo(t_, CV_64F);

  img_size_ = stereo_camera.getResolution(stream::LEFT);

  computeRectificationMapping();
}

void StereoCalibration::computeRectificationMapping()
{
  cv::Mat R_left;
  cv::Mat R_right;
  cv::Mat P_left;
  cv::Mat P_right;

  cv::stereoRectify(K_left_, distortion_left_,
                    K_right_, distortion_right_,
                    img_size_, R_, t_,
                    R_left, R_right, P_left, P_right,
                    Q_, cv::CALIB_ZERO_DISPARITY, -1, img_size_);

  cv::initUndistortRectifyMap(K_left_, distortion_left_, R_left, P_left,
                              img_size_, CV_16SC2, map_left_x_, map_left_y_);
  cv::initUndistortRectifyMap(K_right_, distortion_right_, R_right, P_right,
                              img_size_, CV_16SC2, map_right_x_, map_right_y_);
}

tek5030::StereoPair StereoCalibration::rectify(const tek5030::StereoPair& raw_stereo_pair) const
{
  tek5030::StereoPair rectified;

  if (raw_stereo_pair.left.type() == CV_16UC1)
  {
    raw_stereo_pair.left.convertTo(rectified.left, CV_8UC1, 255.0/65535.0);
    raw_stereo_pair.right.convertTo(rectified.right, CV_8UC1, 255.0/65535.0);
  }
  else
  {
    rectified = raw_stereo_pair;
  }

  remap(rectified.left, rectified.left, map_left_x_, map_left_y_, cv::INTER_LINEAR);
  remap(rectified.right, rectified.right, map_right_x_, map_right_y_, cv::INTER_LINEAR);
  return rectified;
}
