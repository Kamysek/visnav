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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // The essential matrix is the normalized skew symmetric matrix of t_0_1 x
  // R_0_1

  // Normalize t_0_1
  Eigen::Vector3d norm_t_0_1 = t_0_1.normalized();

  // Create skew symmetric matrix
  Eigen::Matrix<double, 3, 3> T_x;
  T_x << 0, -norm_t_0_1(2), norm_t_0_1(1), norm_t_0_1(2), 0, -norm_t_0_1(0),
      -norm_t_0_1(1), norm_t_0_1(0), 0;

  // Calculate essentail matrix
  E = T_x * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // Retrieve 3D points from their respective 2D image plane location
    Eigen::Vector3d p = cam1->unproject(p0_2d);
    Eigen::Vector3d p_hat = cam2->unproject(p1_2d);

    // Calcualte the epipolar constraint p^T * E * p' = 0
    double constraint = p.transpose() * E * p_hat;

    // Check if constraint is smaller than threshold
    if (-epipolar_error_threshold < constraint &&
        constraint < epipolar_error_threshold) {
      md.inliers.push_back(
          std::make_pair(md.matches[j].first, md.matches[j].second));
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // Create opengv bearing vectors
  opengv::bearingVectors_t bv1;
  opengv::bearingVectors_t bv2;

  for (size_t j = 0; j < md.matches.size(); j++) {
    // Extract 2d image plane locations
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // Retrieve 3D points from their respective 2D image plane location
    Eigen::Vector3d p = cam1->unproject(p0_2d);
    Eigen::Vector3d p_hat = cam2->unproject(p1_2d);

    // Normalize
    Eigen::Vector3d norm_p = p.normalized();
    Eigen::Vector3d norm_p_hat = p_hat.normalized();

    // Add to bearing vectors
    bv1.push_back(norm_p);
    bv2.push_back(norm_p_hat);
  }

  // Followed example for central relative pose from
  // https://laurentkneip.github.io/opengv/page_how_to_use.html
  // Create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(bv1, bv2);

  // Create a RANSAC object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;

  // Create a CentralRelativePoseSacProblem
  // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));
  // Run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();

  // Get the result
  // 3x4 transformation matrix containing rotation R and t
  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  // Extract translation and rotation from transformation and add translation
  // and rotation to adapter
  adapter.sett12(best_transformation.topRightCorner(3, 1));
  adapter.setR12(best_transformation.topLeftCorner(3, 3));

  // Perform nonlinear optimization with all ransac inliers
  opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  // Select inlier samples
  ransac.sac_model_->selectWithinDistance(ransac.model_coefficients_,
                                          ransac.threshold_, ransac.inliers_);

  // Count inlier samples
  int inliers_counter = ransac.sac_model_->countWithinDistance(
      nonlinear_transformation, ransac.threshold_);

  if (inliers_counter >= ransac_min_inliers) {
    for (auto const il : ransac.inliers_) {
      md.inliers.push_back(md.matches[il]);
    }

    md.T_i_j = Sophus::SE3d(
        nonlinear_transformation.topLeftCorner(3, 3),
        nonlinear_transformation.topRightCorner(3, 1).normalized());
  }
}
}  // namespace visnav
