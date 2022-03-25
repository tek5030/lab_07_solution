#include "lab_7.h"
#include "cv_stereo_matcher_wrap.h"
#include "sparse_stereo_matcher.h"
#include "stereo_calibration.h"
#include "viewer_3d.h"
#include "visualization.h"
#include "tek5030/realsense_stereo.h"

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stereo.hpp"

#include <iostream>

using namespace tek5030;
using namespace tek5030::RealSense;

void lab7()
{
  StereoCamera camera(StereoCamera::CaptureMode::RECTIFIED);
  camera.setLaserMode(StereoCamera::LaserMode::OFF);

  cv::Ptr<cv::Feature2D> detector = cv::FastFeatureDetector::create();
  cv::Ptr<cv::Feature2D> desc_extractor = cv::ORB::create(2000, 1.2, 1);

  SparseStereoMatcher stereo_matcher{detector, desc_extractor};

  const StereoCalibration calibration(camera);

  const std::string matching_win{"Stereo matching"};
  const std::string depth_win{"Stereo depth"};
  const std::string dense_win{"Dense disparity"};
  cv::namedWindow(matching_win, cv::WINDOW_NORMAL);
  cv::namedWindow(depth_win, cv::WINDOW_NORMAL);
  cv::namedWindow(dense_win, cv::WINDOW_NORMAL);

  Viewer3D viewer_3d;
  for (;;)
  {
    // Grab raw images
    const StereoPair stereo_raw = camera.getStereoPair();

    // Rectify images.
    const StereoPair stereo_rectified = calibration.rectify(stereo_raw);

    // Add frustums to the 3D visualization
    viewer_3d.addCameraFrustum("Camera-frustum-left",  calibration.K_left(),  stereo_rectified.left);
    viewer_3d.addCameraFrustum("Camera-frustum-right", calibration.K_right(), stereo_rectified.right, calibration.pose());

    // Perform sparse matching.
    stereo_matcher.match(stereo_rectified);

    // Visualize matched point correspondences
    const cv::Mat match_image = visualizeMatches(stereo_rectified, stereo_matcher);
    cv::imshow(matching_win, match_image);

    // Get points with disparities.
    if (!stereo_matcher.point_disparities().empty())
    {
      // Compute depth in meters for each point.
      const double fu = calibration.f();
      const double bx = calibration.baseline();

      cv::Mat visualized_depths;
      cv::cvtColor(stereo_rectified.left, visualized_depths, cv::COLOR_GRAY2BGR);

      std::vector<uint8_t> point_colors;
      for (const auto& d_vec : stereo_matcher.point_disparities())
      {
        const cv::Point pixel_pos{
            static_cast<int>(std::round(d_vec[0])),
            static_cast<int>(std::round(d_vec[1]))
        };

        const auto disparity = d_vec[2];
        const auto depth = fu * bx / disparity;

        addDepthPoint(visualized_depths, pixel_pos, depth);

        point_colors.push_back(stereo_rectified.left.at<uint8_t>(pixel_pos));
      }
      cv::imshow(depth_win, visualized_depths);

      std::vector<cv::Vec3d> world_points(point_colors.size());
      cv::perspectiveTransform(stereo_matcher.point_disparities(), world_points, calibration.Q());

      viewer_3d.addPointCloud(world_points, point_colors);
    }

   {  // Dense stereo matching using OpenCV
      cv::Ptr<cv::stereo::StereoMatcher> ptr = cv::stereo::StereoBinarySGBM::create(0, 192,5);

      CvStereoMatcherWrap dense_matcher(ptr);
      const auto dense_disparity = dense_matcher.compute(stereo_rectified);
      if (!dense_disparity.empty())
      {
        cv::Mat dense_depth = static_cast<float>(calibration.f() * calibration.baseline()) / dense_disparity;
        constexpr float max_depth = 5.f;
        dense_depth.setTo(0, (dense_disparity < 0) | (dense_depth > max_depth));

        dense_depth /= max_depth;

        cv::cvtColor(dense_depth, dense_depth, cv::COLOR_GRAY2BGR);
        dense_depth.convertTo(dense_depth, CV_8U, 255);
        cv::applyColorMap(dense_depth, dense_depth, cv::COLORMAP_JET);

        cv::imshow(dense_win, dense_depth);
      }
    }
    viewer_3d.spinOnce();

    if (cv::waitKey(1) >= 0)
    { break; }
  }
}


