// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCrystal/CompositeCluster.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {
/**
 * Helper comparitor for finding IClusters based on an input label.
 */
class Comparitor {
private:
  size_t m_label;

public:
  explicit Comparitor(const size_t &label) : m_label(label) {}
  bool operator()(const std::shared_ptr<Mantid::Crystal::ICluster> &pCluster) const {
    return pCluster->containsLabel(m_label);
  }
};
} // namespace

namespace Mantid::Crystal {

/**
 * Integrate the composite cluster.
 * @param ws : Workspace to integrate
 * @return Integrated signal and error sq values.
 */
ICluster::ClusterIntegratedValues
CompositeCluster::integrate(std::shared_ptr<const Mantid::API::IMDHistoWorkspace> ws) const {

  double errorIntSQ = 0;
  double sigInt = 0;
  // Integrate owned clusters and add those results too.
  for (const auto &ownedCluster : m_ownedClusters) {
    auto integratedValues = ownedCluster->integrate(ws);
    sigInt += integratedValues.get<0>();
    errorIntSQ += integratedValues.get<1>();
  }
  return ClusterIntegratedValues(sigInt, errorIntSQ);
}

/**
 * Write to an output histo workspace.
 * @param ws
 */
void CompositeCluster::writeTo(std::shared_ptr<Mantid::API::IMDHistoWorkspace> ws) const {
  for (const auto &ownedCluster : m_ownedClusters) {
    ownedCluster->writeTo(ws);
  }
}

/**
 * Get the label.
 * @return Current label.
 */
size_t CompositeCluster::getLabel() const {
  findMinimum();
  if (!m_label.has_value()) {
    throw std::runtime_error("No child IClusters. CompositeCluster::getLabel() is not supported.");
  } else {
    return m_label.value(); // Assumes all are uniform.
  }
}

/**
 * Get the original label. Doesn't make sense for composites, so wired through
 * to getLabel()
 * @return getLabel()
 */
size_t CompositeCluster::getOriginalLabel() const { return getLabel(); }

/**
 * Get the size of the composite. This is the total size of all owned clusters.
 * @return total size.
 */
size_t CompositeCluster::size() const {
  return std::accumulate(m_ownedClusters.cbegin(), m_ownedClusters.cend(), static_cast<size_t>(0),
                         [](auto sum, const auto &cluster) { return sum + cluster->size(); });
}

/// Add an index. This method does not apply to composite clusters.
void CompositeCluster::addIndex(const size_t & /*index*/) {
  throw std::runtime_error("addIndex not implemented on CompositeCluster");
}

/**
 * Find the minimum label in the composite
 */
void CompositeCluster::findMinimum() const {
  if (!m_ownedClusters.empty()) {
    ICluster const *minCluster = m_ownedClusters.front().get();
    size_t minLabel = minCluster->getLabel();
    for (size_t i = 1; i < m_ownedClusters.size(); ++i) {
      size_t temp = m_ownedClusters[i]->getLabel();
      if (temp < minLabel) {
        minLabel = temp;
      }
    }
    m_label = minLabel;
  }
}

/**
 * Convert the disjointSet to a uniform minimum value
 * @param disjointSet : DisjointSets to adapt.
 */
void CompositeCluster::toUniformMinimum(std::vector<DisjointElement> &disjointSet) {
  if (!m_ownedClusters.empty()) {
    ICluster const *minCluster = m_ownedClusters.front().get();
    size_t minLabel = minCluster->getLabel();
    for (size_t i = 1; i < m_ownedClusters.size(); ++i) {
      size_t temp = m_ownedClusters[i]->getLabel();
      if (temp < minLabel) {
        minLabel = temp;
        minCluster = m_ownedClusters[i].get();
      }
    }
    m_label = minLabel;

    for (auto &ownedCluster : m_ownedClusters) {
      ownedCluster->setRootCluster(minCluster);
      ownedCluster->toUniformMinimum(disjointSet);
    }
  }
}

/**
 * Get any representative index from this cluster
 * @return : Representative index.
 */
size_t CompositeCluster::getRepresentitiveIndex() const {
  return this->m_ownedClusters.front()->getRepresentitiveIndex();
}

/**
 * Set the root cluster
 * @param root : Root cluster to use
 */
void CompositeCluster::setRootCluster(ICluster const *root) {
  for (auto &ownedCluster : m_ownedClusters) {
    ownedCluster->setRootCluster(root);
  }
}

/**
 * Add other IClusters to own.
 * @param toOwn : Item to own
 */
void CompositeCluster::add(std::shared_ptr<ICluster> &toOwn) {
  if (toOwn->size() > 0) // Do not add empty clusters.
  {
    m_ownedClusters.emplace_back(toOwn);
  }
}

/**
 * Does this cluster contain the label of the argument.
 * @param label : Label id to find
 * @return True only if that label is contained.
 */
bool CompositeCluster::containsLabel(const size_t &label) const {
  Comparitor comparitor(label);
  return m_ownedClusters.end() != std::find_if(m_ownedClusters.begin(), m_ownedClusters.end(), comparitor);
}

} // namespace Mantid::Crystal
