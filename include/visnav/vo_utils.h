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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // Cycle through landmarks
  for (const auto& landmark : landmarks) {
    // Convert landmark world coordinate to camera coordinate
    Eigen::Vector3d p_3d = current_pose.inverse() * landmark.second.p;
    // Check whether pointi behind camera
    if (p_3d.z() >= cam_z_threshold) {
      // Project to image plane
      Eigen::Vector2d p_2d = cam->project(p_3d);
      // Check if point is projected outside of the image
      if (p_2d.x() >= 0 && p_2d.x() < cam->width() && p_2d.y() >= 0 &&
          p_2d.y() < cam->height()) {
        // Save projection and landmark trackids
        projected_points.push_back(p_2d);
        projected_track_ids.push_back(landmark.first);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // Cycle through keypoint corners
  for (size_t i = 0; i < kdl.corners.size(); i++) {
    // Store first and second best match
    int first = std::numeric_limits<int>::max();
    int second = std::numeric_limits<int>::max();
    int corner_descriptors_index = -1;

    // Extract descriptor of current keypoint corner
    std::bitset<256> corner_descriptor = kdl.corner_descriptors.at(i);

    // Cycle through projected points
    for (size_t j = 0; j < projected_points.size(); ++j) {
      // Calculate norm of kdl.corners - projected_points
      double norm = (kdl.corners[i] - projected_points[j]).norm();
      // If norm smaller than match_max_dist_2d search for matches
      if (norm <= match_max_dist_2d) {
        int first_obs = std::numeric_limits<int>::max();
        // Cycle through observations and check for best hamming distance
        for (const auto& obs : landmarks.at(projected_track_ids.at(j)).obs) {
          int hamming_distance =
              (corner_descriptor ^
               feature_corners.at(obs.first).corner_descriptors[obs.second])
                  .count();

          if (hamming_distance < first_obs) {
            first_obs = hamming_distance;
          }
        }

        // Check if distance has to be updated
        if (first_obs <= first) {
          second = first;
          first = first_obs;
          corner_descriptors_index = j;
          continue;
        }

        if (first_obs < second) {
          second = first_obs;
          continue;
        }
      }
    }

    // Check if threshold is reached or distance to second best match is
    // smaller than smallest distance multiplied by feature_match_dist_2_best
    if (first < feature_match_threshold &&
        !(second < first * feature_match_dist_2_best)) {
      md.matches.emplace_back(i,
                              projected_track_ids.at(corner_descriptors_index));
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // Create bearing vector for absolute adapter
  opengv::bearingVectors_t bv;

  // Store landmark points
  opengv::points_t points;

  // Cycle through matches
  for (const auto& match : md.matches) {
    // Extract landmark 3d point
    opengv::point_t p_3d = landmarks.at(match.second).p;
    points.push_back(p_3d);

    // Extract 2d points for unprojection
    Eigen::Vector2d p_2d = kdl.corners.at(match.first);

    // Unproject 2d points and push to bearing vector
    bv.push_back(cam->unproject(p_2d));
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

  // Store translation and rotation
  md.T_w_c = Sophus::SE3d(nonlinear_transformation.topLeftCorner(3, 3),
                          nonlinear_transformation.topRightCorner(3, 1));

  // Push back inliers
  for (const auto inlier : ransac.inliers_) {
    md.inliers.push_back(md.matches.at(inlier));
  }
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // Add all left inliers to landmarks
  std::map<FeatureId, TrackId> lc_inliers;
  for (const auto& inlier : md.inliers) {
    landmarks[inlier.second].obs[fcidl] = inlier.first;
    lc_inliers[inlier.first] = inlier.second;
  }

  // Cycle through stereo inliers
  // Add observation to existing landmarks if left cameras feature appears in
  // inliers Otherwise store not added
  std::vector<std::pair<std::pair<FeatureId, FeatureId>, size_t>>
      not_added_inliers;
  for (size_t i = 0; i < md_stereo.inliers.size(); ++i) {
    if (lc_inliers.find(md_stereo.inliers[i].first) != lc_inliers.end())
      landmarks[lc_inliers[md_stereo.inliers[i].first]].obs[fcidr] =
          md_stereo.inliers[i].second;
    else
      not_added_inliers.push_back(std::make_pair(md_stereo.inliers[i], i));
  }

  opengv::bearingVectors_t bv0;
  opengv::bearingVectors_t bv1;

  // Create bearing vectors
  for (const auto& inlier : not_added_inliers) {
    bv0.push_back(calib_cam.intrinsics[fcidl.cam_id]->unproject(
        kdl.corners[inlier.first.first]));
    bv1.push_back(calib_cam.intrinsics[fcidr.cam_id]->unproject(
        kdr.corners[inlier.first.second]));
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(bv0, bv1, t_0_1, R_0_1);

  // Perform triangulation and increase next_landmard_id
  for (size_t i = 0; i < not_added_inliers.size(); ++i) {
    next_landmark_id++;
    landmarks[next_landmark_id].p =
        md.T_w_c * opengv::triangulation::triangulate(adapter, i);

    landmarks[next_landmark_id].obs.emplace(
        fcidl, md_stereo.inliers[not_added_inliers[i].second].first);
    landmarks[next_landmark_id].obs.emplace(
        fcidr, md_stereo.inliers.at(not_added_inliers[i].second).second);
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // Erase keyframes and cameras as long as size bigger than max_num_kfs
  auto it = kf_frames.begin();
  while (it != kf_frames.end() && kf_frames.size() > max_num_kfs) {
    FrameCamId fci_0 = FrameCamId(*it, 0);
    FrameCamId fci_1 = FrameCamId(*it, 1);

    // Erase cameras
    cameras.erase(fci_0);
    cameras.erase(fci_1);

    // Erase keyframe
    it = kf_frames.erase(it);

    // Cycle through landmarks
    auto it_l = landmarks.begin();
    while (it_l != landmarks.end()) {
      // Erase observation
      it_l->second.obs.erase(fci_0);
      it_l->second.obs.erase(fci_1);

      // Erase landmark
      if (it_l->second.obs.size() == 0) {
        old_landmarks.insert(*it_l);
        it_l = landmarks.erase(it_l);
      } else
        it_l++;
    }
  }
}
}  // namespace visnav
