// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "PreviewInstrumentDisplay.h"
#include "MantidQtWidgets/InstrumentView/InstrumentActor.h"
#include "MantidQtWidgets/InstrumentView/ProjectionSurface.h"
#include "MantidQtWidgets/InstrumentView/UnwrappedCylinder.h"

#include <QVBoxLayout>

using MantidQt::MantidWidgets::ProjectionSurface;

namespace MantidQt::CustomInterfaces::ISISReflectometry {

PreviewInstrumentDisplay::PreviewInstrumentDisplay(QWidget *placeholder, std::function<void()> onShapeChanged,
                                                   std::unique_ptr<IInstViewModel> instViewModel)
    : m_placeholder(placeholder), m_instViewModel(std::move(instViewModel)),
      m_onShapeChanged(std::move(onShapeChanged)) {
  resetInstView();
}

PreviewInstrumentDisplay::~PreviewInstrumentDisplay() { disconnectSurfaceSignals(); }

void PreviewInstrumentDisplay::updateWorkspace(Mantid::API::MatrixWorkspace_sptr &workspace) {
  m_instViewModel->updateWorkspace(workspace);
}

void PreviewInstrumentDisplay::resetInstView() {
  disconnectSurfaceSignals();
  m_instDisplay = nullptr;
  m_instDisplay = std::make_unique<MantidWidgets::InstrumentDisplay>(m_placeholder);
}

void PreviewInstrumentDisplay::plotInstView() {
  auto *actor = m_instViewModel->getInstrumentViewActor();
  if (!actor)
    return;
  if (!m_instDisplay)
    return;
  disconnectSurfaceSignals();
  auto widgetSize = m_instDisplay->currentWidget()->size();
  m_instDisplay->setSurface(std::make_shared<MantidWidgets::UnwrappedCylinder>(
      actor, m_instViewModel->getSamplePos(), m_instViewModel->getAxis(), widgetSize, false));
  connectSurfaceSignals();
}

QLayout *PreviewInstrumentDisplay::getInstViewLayout() {
  disconnectSurfaceSignals();
  m_instDisplay.reset();
  if (!m_placeholder->layout()) {
    new QVBoxLayout(m_placeholder);
  }
  return m_placeholder->layout();
}

void PreviewInstrumentDisplay::setInstViewZoomMode() {
  if (m_instDisplay)
    if (auto surface = m_instDisplay->getSurface())
      surface->setInteractionMode(ProjectionSurface::MoveMode);
}

void PreviewInstrumentDisplay::setInstViewEditMode() {
  if (m_instDisplay)
    if (auto surface = m_instDisplay->getSurface())
      surface->setInteractionMode(ProjectionSurface::EditShapeMode);
}

void PreviewInstrumentDisplay::setInstViewSelectRectMode() {
  if (m_instDisplay)
    if (auto surface = m_instDisplay->getSurface()) {
      surface->setInteractionMode(ProjectionSurface::EditShapeMode);
      surface->startCreatingShape2D("rectangle", Qt::green, QColor(255, 255, 255, 80));
    }
}

std::vector<size_t> PreviewInstrumentDisplay::getSelectedDetectors() const {
  std::vector<size_t> result;
  if (m_instDisplay)
    if (auto surface = m_instDisplay->getSurface())
      surface->getMaskedDetectors(result);
  return result;
}

std::vector<Mantid::detid_t> PreviewInstrumentDisplay::detIndicesToDetIDs(std::vector<size_t> const &detIndices) const {
  return m_instViewModel->detIndicesToDetIDs(detIndices);
}

void PreviewInstrumentDisplay::disconnectSurfaceSignals() {
  for (auto &conn : m_surfaceConnections) {
    QObject::disconnect(conn);
  }
  m_surfaceConnections.clear();
}

void PreviewInstrumentDisplay::connectSurfaceSignals() {
  if (!m_instDisplay)
    return;
  auto surface = m_instDisplay->getSurface();
  if (!surface)
    return;
  auto callback = [this]() {
    if (m_onShapeChanged)
      m_onShapeChanged();
  };
  m_surfaceConnections.push_back(
      QObject::connect(surface.get(), &ProjectionSurface::shapeChangeFinished, m_placeholder, callback));
  m_surfaceConnections.push_back(
      QObject::connect(surface.get(), &ProjectionSurface::shapesRemoved, m_placeholder, callback));
  m_surfaceConnections.push_back(
      QObject::connect(surface.get(), &ProjectionSurface::shapesCleared, m_placeholder, callback));
}

} // namespace MantidQt::CustomInterfaces::ISISReflectometry
