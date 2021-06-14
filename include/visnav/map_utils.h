/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <fstream>

#include <tbb/task_scheduler_init.h>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // Create bearing vectors for central relative adapter
  opengv::bearingVectors_t bv0;
  opengv::bearingVectors_t bv1;

  // Cycle through shared track ids and create new landmarks
  for (TrackId st_id : shared_track_ids) {
    // Check if track id is in landmarks
    if (landmarks.find(st_id) == landmarks.end()) {
      // Add new track id
      new_track_ids.push_back(st_id);

      const std::shared_ptr<AbstractCamera<double>>& cam1 =
          calib_cam.intrinsics[fcid0.cam_id];
      const std::shared_ptr<AbstractCamera<double>>& cam2 =
          calib_cam.intrinsics[fcid1.cam_id];

      // Extract features for frame fcid0 and fcid1
      FeatureId f0_id = feature_tracks.at(st_id).at(fcid0);
      FeatureId f1_id = feature_tracks.at(st_id).at(fcid1);

      // Extract 2d points for unprojection
      Eigen::Vector2d p0_2d = feature_corners.at(fcid0).corners[f0_id];
      Eigen::Vector2d p1_2d = feature_corners.at(fcid1).corners[f1_id];

      // Unproject 2d points and push to bearing vector
      bv0.push_back(cam1->unproject(p0_2d));
      bv1.push_back(cam2->unproject(p1_2d));
    }
  }

  // Relative pose
  Sophus::SE3d relative_pose =
      cameras.at(fcid0).T_w_c.inverse() * cameras.at(fcid1).T_w_c;

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bv0, bv1, relative_pose.translation(), relative_pose.rotationMatrix());

  // Cycle through track ids and add calculated 3d point to landmark
  for (size_t i = 0; i < new_track_ids.size(); ++i) {
    TrackId tid = new_track_ids[i];
    //     3d position in world coordinates
    Eigen::Vector3d landmark_3d_position =
        cameras.at(fcid0).T_w_c *
        opengv::triangulation::triangulate(adapter, i);

    // Update landmark
    landmarks[tid].p = landmark_3d_position;

    // Add all observations
    for (const auto& feature_track : feature_tracks.at(tid))
      if (cameras.find(feature_track.first) != cameras.end())
        landmarks[tid].obs.emplace(feature_track);
  }

  return new_track_ids.size();
}
// Initialize the scene from a stereo pair, using the known transformation
// from camera calibration. This adds the inital two cameras and
// triangulates shared landmarks. Note: in principle we could also
// initialize a map from another images pair using the transformation from
// the pairwise matching with the 5-point algorithm. However, using a stereo
// pair has the advantage that the map is initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  cameras[fcid0].T_w_c = calib_cam.T_i_c[fcid0.cam_id];
  cameras[fcid1].T_w_c = calib_cam.T_i_c[fcid1.cam_id];

  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We
// use pnp and ransac to localize the camera in the presence of outlier
// tracks. After finding an inlier set with pnp, we do non-linear refinement
// using all inliers and also update the set of inliers using the refined
// pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv
// documentation on how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // Create bearing vector for absolute adapter
  opengv::bearingVectors_t bv;

  // Store landmark points
  opengv::points_t points;

  const std::shared_ptr<AbstractCamera<double>>& cam =
      calib_cam.intrinsics[fcid.cam_id];

  // Cycle through shared track ids
  for (TrackId st_id : shared_track_ids) {
    // Add landmark point to points
    points.push_back(landmarks.at(st_id).p);

    // Extract features for frame fcid
    FeatureId f_id = feature_tracks.at(st_id).at(fcid);

    // Extract 2d points for unprojection
    Eigen::Vector2d p_2d = feature_corners.at(fcid).corners[f_id];

    // Unproject 2d points and push to bearing vector
    bv.push_back(cam->unproject(p_2d).normalized());
  }

  // Followed example for absolute pose from
  // https://laurentkneip.github.io/opengv/page_how_to_use.html
  // Create the central absolute adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bv, points);

  // Create a RANSAC object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;

  // Create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  // Run ransac
  ransac.sac_model_ = absposeproblem_ptr;

  // Calcualte threshold
  ransac.threshold_ =
      1.0 - cos((reprojection_error_pnp_inlier_threshold_pixel / 500.0));

  ransac.computeModel();

  // Set translantion and rotation of ransac to adapter
  adapter.sett(ransac.model_coefficients_.topRightCorner(3, 1));
  adapter.setR(ransac.model_coefficients_.topLeftCorner(3, 3));

  // Perform nonlinear optimization with all ransac inliers
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  absposeproblem_ptr->selectWithinDistance(nonlinear_transformation,
                                           ransac.threshold_, ransac.inliers_);

  // Push back trackids of inliers
  for (const auto inlier : ransac.inliers_) {
    inlier_track_ids.push_back(shared_track_ids.at(inlier));
  }

  // Store translation and rotation
  T_w_c = Sophus::SE3d(nonlinear_transformation.topLeftCorner(3, 3),
                       nonlinear_transformation.topRightCorner(3, 1));
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally
// intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<FrameCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // Add cameras to problem
  for (auto& camera : cameras) {
    problem.AddParameterBlock(camera.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    // Check if camera is fixed
    if (fixed_cameras.find(camera.first) != fixed_cameras.end()) {
      problem.SetParameterBlockConstant(camera.second.T_w_c.data());
    }
  }

  // Cycle through landmarks and observations
  for (auto& landmark : landmarks) {
    for (auto& observation : landmark.second.obs) {
      // Extract p_2d from feature corners with observation information
      const auto& p_2d =
          feature_corners.at(observation.first).corners.at(observation.second);

      // Add landmark 3d point to problem
      problem.AddParameterBlock(landmark.second.p.data(), 3);

      // Create ceres cost function
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(
          new BundleAdjustmentReprojectionCostFunctor(
              p_2d, calib_cam.intrinsics[observation.first.cam_id]->name()));

      if (options.use_huber) {
        problem.AddResidualBlock(
            cost_function, new ceres::HuberLoss(options.huber_parameter),
            cameras[observation.first].T_w_c.data(), landmark.second.p.data(),
            calib_cam.intrinsics[observation.first.cam_id]->data());
      } else {
        problem.AddResidualBlock(
            cost_function, nullptr, cameras[observation.first].T_w_c.data(),
            landmark.second.p.data(),
            calib_cam.intrinsics[observation.first.cam_id]->data());
      }
    }
  }

  // Add camera intrinsics to problem
  problem.AddParameterBlock(calib_cam.intrinsics[0]->data(), 8);
  problem.AddParameterBlock(calib_cam.intrinsics[1]->data(), 8);
  if (!options.optimize_intrinsics) {
    problem.SetParameterBlockConstant(calib_cam.intrinsics[0]->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics[1]->data());
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = tbb::task_scheduler_init::default_num_threads();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
