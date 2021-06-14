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
#include <cstdlib>
#include <fstream>
#include <unordered_map>

#include <cereal/archives/binary.hpp>

#include <visnav/common_types.h>

namespace cereal {
class access;
}

namespace visnav {

class BowVocabulary {
 public:
  using NodeId = unsigned int;
  using TDescriptor = std::bitset<256>;

  BowVocabulary(const std::string& filename) { load(filename); }

  inline void transformFeatureToWord(const TDescriptor& feature,
                                     WordId& word_id, WordValue& weight) const {
    std::pair<WordValue, WordId> result = recPropagateFeature(feature, 0);

    weight = std::get<0>(result);
    word_id = std::get<1>(result);
  }

  inline std::pair<WordValue, WordId> recPropagateFeature(
      const TDescriptor& feature, size_t current_node) const {
    // Check if current node is a leaf,
    // if so return pair with weight and word_id
    if (m_nodes[current_node].isLeaf()) {
      return std::make_pair(m_nodes[current_node].weight,
                            m_nodes[current_node].word_id);
    }

    // Search shortest distance between children descriptor and feature
    NodeId id = 0;
    size_t distanceFD = std::numeric_limits<size_t>::max();
    for (const auto child_index : m_nodes[current_node].children) {
      size_t distance = (feature ^ m_nodes[child_index].descriptor).count();
      // Check if current distance is shorter than stored distance
      // otherwise update values
      if (distance < distanceFD) {
        distanceFD = distance;
        id = child_index;
      }
    }

    return recPropagateFeature(feature, id);
  }

  inline void transform(const std::vector<TDescriptor>& features,
                        BowVector& v) const {
    v.clear();

    if (m_nodes.empty()) {
      return;
    }

    // Transform features to a word
    std::vector<WordId> wordIds(features.size(), 0);
    std::vector<WordValue> wordValues(features.size(), 0);
    for (size_t i = 0; i < features.size(); ++i) {
      transformFeatureToWord(features[i], wordIds[i], wordValues[i]);
    }

    // Add same ids together
    std::map<WordId, WordValue> words_with_value;
    for (size_t i = 0; i < wordIds.size(); ++i) {
      // Ignore too small values
      if (wordValues[i] < 1e-10) {
        continue;
      }

      // Store values of words in map and add together if index is the same
      words_with_value[wordIds[i]] += wordValues[i];
    }

    // Calculate norm value to norm the wordvalues with
    WordValue norm = 0;
    for (const auto& word : words_with_value) {
      norm += word.second;
    }

    // Normalize WordValues and push to BowVector
    for (const auto& word : words_with_value) {
      std::pair<WordId, WordValue> word_value_pair =
          std::make_pair(word.first, word.second / norm);
      v.push_back(word_value_pair);
    }
  }
  void save(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);

      archive(*this);

    } else {
      std::cout << "Failed to save vocabulary as " << filename << std::endl;
    }
  }

  void load(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);

      archive(*this);

      std::cout << "Loaded vocabulary from " << filename << " with "
                << m_words.size() << " words." << std::endl;

    } else {
      std::cout << "Failed to load vocabulary " << filename << std::endl;
      std::abort();
    }
  }

 protected:
  /// Tree node
  struct Node {
    /// Node id
    NodeId id;
    /// Weight if the node is a word; may be positive or zero
    WordValue weight;
    /// Children
    std::vector<NodeId> children;
    /// Parent node (undefined in case of root)
    NodeId parent;
    /// Node descriptor
    TDescriptor descriptor;

    /// Word id if the node is a word
    WordId word_id;

    /**
     * Empty constructor
     */
    Node() : id(0), weight(0), parent(0), word_id(0) {}

    /**
     * Constructor
     * @param _id node id
     */
    Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

    /**
     * Returns whether the node is a leaf node
     * @return true iff the node is a leaf
     */
    inline bool isLeaf() const { return children.empty(); }

    template <class Archive>
    void serialize(Archive& ar) {
      ar(id, weight, children, parent, descriptor, word_id);
    }
  };

  template <class Archive>
  void save(Archive& ar) const {
    ar(CEREAL_NVP(this->m_k));
    ar(CEREAL_NVP(this->m_L));
    ar(CEREAL_NVP(this->m_nodes));
  }

  template <class Archive>
  void load(Archive& ar) {
    ar(CEREAL_NVP(this->m_k));
    ar(CEREAL_NVP(this->m_L));
    ar(CEREAL_NVP(this->m_nodes));

    createWords();
  }

  void createWords() {
    m_words.clear();

    if (!m_nodes.empty()) {
      m_words.reserve((int)pow((double)m_k, (double)m_L));

      for (Node& n : m_nodes) {
        if (n.isLeaf()) {
          n.word_id = m_words.size();
          m_words.push_back(&n);
        }
      }
    }
  }

  friend class cereal::access;

  /// Branching factor
  int m_k;

  /// Depth levels
  int m_L;

  /// Tree nodes
  std::vector<Node> m_nodes;

  /// Words of the vocabulary (tree leaves)
  /// this condition holds: m_words[wid]->word_id == wid
  std::vector<Node*> m_words;
};

}  // namespace visnav
