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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // Calculate theta
  T theta = sqrt(xi.transpose() * xi);

  // If theta empty return identity matrix
  if (theta == T(0)) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }

  Eigen::Vector3d w = xi / theta;

  // Calculate skew symmetric matrix w_hat
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  // Calculate exp SO(3)
  return Eigen::Matrix<T, 3, 3>::Identity() + sin(theta) * w_hat +
         (1 - cos(theta)) * (w_hat * w_hat);
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // Calculate theta
  T theta = acos((mat.trace() - T(1)) / T(2));

  // If theta empty return identity matrix
  if (theta == T(0)) {
    return Eigen::Matrix<T, 3, 1>::Zero();
  }

  // Calculate w
  Eigen::Matrix<T, 3, 1> r;
  r << (mat(2, 1) - mat(1, 2)), (mat(0, 2) - mat(2, 0)),
      (mat(1, 0) - mat(0, 1));

  Eigen::Matrix<T, 3, 1> w;
  w = (T(1) / (T(2) * sin(theta))) * r;

  // Calculate log SO(3)
  return theta * w;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // Extract v and w from xi
  Eigen::Matrix<T, 3, 1> v, w;
  v << xi(0), xi(1), xi(2);
  w << xi(3), xi(4), xi(5);

  // Calculate theta
  T theta = sqrt(w.transpose() * w);

  // If theta empty return identity matrix
  if (theta == T(0)) {
    Eigen::Matrix<T, 4, 4> mat = Eigen::Matrix<T, 4, 4>::Identity();
    mat.topRightCorner(3, 1) = v;
    return mat;
  }

  // Get exp SO3
  Eigen::Matrix<T, 3, 3> so3_expmap_w = user_implemented_expmap(w);

  Eigen::Vector3d rw = w / theta;

  // Calculate skew symmetric matrix w_hat
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -rw(2), rw(1), rw(2), 0, -rw(0), -rw(1), rw(0), 0;

  // Calculate Jacobian
  Eigen::Matrix<T, 3, 3> J = Eigen::Matrix<T, 3, 3>::Identity() +
                             ((T(1) - cos(theta)) / (theta)) * w_hat +
                             ((theta - sin(theta)) / (theta)) * (w_hat * w_hat);

  // Calculate exp SE(3)
  Eigen::Matrix<T, 4, 4> mat = Eigen::Matrix<T, 4, 4>::Identity();
  mat.topLeftCorner(3, 3) = so3_expmap_w;
  mat.topRightCorner(3, 1) = (J * v);

  return mat;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // Extract R and t from mat
  Eigen::Matrix<T, 3, 3> R = mat.topLeftCorner(3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.topRightCorner(3, 1);

  // Calculate theta
  T theta = acos((R.trace() - T(1)) / T(2));

  // If theta empty return identity matrix
  if (theta == T(0)) {
    // Create output
    Eigen::Matrix<T, 6, 1> out;
    out << t, Eigen::Matrix<T, 3, 1>::Zero();
    return out;
  }

  // Calculate w
  Eigen::Matrix<T, 3, 1> w = user_implemented_logmap(R);
  Eigen::Matrix<T, 3, 1> w_div_theta = w / theta;

  // Calculate skew symmetric matrix w_hat
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -w_div_theta(2), w_div_theta(1), w_div_theta(2), 0,
      -w_div_theta(0), -w_div_theta(1), w_div_theta(0), 0;

  // Calculate Jacobian
  Eigen::Matrix<T, 3, 3> J = Eigen::Matrix<T, 3, 3>::Identity() +
                             ((T(1) - cos(theta)) / (theta)) * w_hat +
                             ((theta - sin(theta)) / (theta)) * (w_hat * w_hat);

  // Calculate v
  Eigen::Matrix<T, 3, 1> v = (J.inverse() * t);

  // Create output
  Eigen::Matrix<T, 6, 1> out;
  out << v, w;
  return out;
}

}  // namespace visnav
