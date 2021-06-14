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

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

namespace visnav {

class BowDatabase {
 public:
  BowDatabase() {}

  inline void insert(const FrameCamId& fcid, const BowVector& bow_vector) {
    // Cycle through the bow vector and add the fcid
    // and wordValue for each wordId wordId
    for (const auto& bow_pair : bow_vector) {
      inverted_index[bow_pair.first].push_back(
          std::make_pair(fcid, bow_pair.second));
    }
  }

  static bool sort_function(std::pair<FrameCamId, WordValue> i,
                            std::pair<FrameCamId, WordValue> j) {
    if (i.second < j.second) {
      return true;
    }
    return false;
  }

  inline void query(const BowVector& bow_vector, size_t num_results,
                    BowQueryResult& results) const {
    // Contains WordId and WordValue for each FrameCamId
    std::unordered_map<FrameCamId, std::vector<std::pair<WordId, WordValue>>>
        frame_value_pairs;

    // Cylce through bow vector and store pairs
    // of wordId and wordValue for a frameId
    for (const auto& bow_pair : bow_vector) {
      // Check if wordId is in inverted_index
      if (inverted_index.find(bow_pair.first) != inverted_index.end())
        // Cycle through all frameId and wordValue pairs
        for (const auto& frame_value_pair : inverted_index.at(bow_pair.first)) {
          // Add wordId and wordValue for frameId to accumulated_scores
          frame_value_pairs[frame_value_pair.first].push_back(
              std::make_pair(bow_pair.first, frame_value_pair.second));
        }
    }

    std::vector<std::pair<FrameCamId, WordValue>> bow_distances;
    // Calculate efficient bow scoring
    for (const auto& frame_value_pair : frame_value_pairs) {
      // Store wordId and wordValue in map
      std::map<WordId, WordValue> words_with_value;
      for (size_t i = 0; i < frame_value_pair.second.size(); ++i) {
        // Store values of words in map and add together if index is the same
        words_with_value[frame_value_pair.second[i].first] +=
            frame_value_pair.second[i].second;
      }

      // Calculate distance
      double distance = 2;
      for (const auto& bow_pair : bow_vector) {
        // Check if wordId is in map, if so calculate distance metric
        if (words_with_value.find(bow_pair.first) != words_with_value.end())
          distance += abs(bow_pair.second - words_with_value[bow_pair.first]) -
                      abs(bow_pair.second) -
                      abs(words_with_value[bow_pair.first]);
      }

      // Store frameId and distance
      bow_distances.push_back(std::make_pair(frame_value_pair.first, distance));
    }

    // Make sure the num_results size is not bigger than possible results
    num_results = std::min(bow_distances.size(), num_results);

    // Perform partial sort of the lenght num_results with custom sort_function
    std::partial_sort(bow_distances.begin(),
                      bow_distances.begin() + num_results, bow_distances.end(),
                      sort_function);

    // Extract certain length of the bow distances vector
    std::vector<std::pair<FrameCamId, WordValue>> res(
        bow_distances.begin(), bow_distances.begin() + num_results);
    results = res;
  }

  void clear() { inverted_index.clear(); }

  void save(const std::string& out_path) {
    BowDBInverseIndex state;
    for (const auto& kv : inverted_index) {
      for (const auto& a : kv.second) {
        state[kv.first].emplace_back(a);
      }
    }
    std::ofstream os;
    os.open(out_path, std::ios::binary);
    cereal::JSONOutputArchive archive(os);
    archive(state);
  }

  void load(const std::string& in_path) {
    BowDBInverseIndex inverseIndexLoaded;
    {
      std::ifstream os(in_path, std::ios::binary);
      cereal::JSONInputArchive archive(os);
      archive(inverseIndexLoaded);
    }
    for (const auto& kv : inverseIndexLoaded) {
      for (const auto& a : kv.second) {
        inverted_index[kv.first].emplace_back(a);
      }
    }
  }

  const BowDBInverseIndexConcurrent& getInvertedIndex() {
    return inverted_index;
  }

 protected:
  BowDBInverseIndexConcurrent inverted_index;
};

}  // namespace visnav
