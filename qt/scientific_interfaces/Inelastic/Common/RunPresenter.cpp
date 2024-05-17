// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2024 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "RunPresenter.h"

#include "IRunSubscriber.h"
#include "RunView.h"

namespace MantidQt {
namespace CustomInterfaces {

RunPresenter::RunPresenter(IRunSubscriber *subscriber, IRunView *view) : m_subscriber(subscriber), m_view(view) {
  m_view->subscribePresenter(this);
}

void RunPresenter::handleRunClicked() { m_subscriber->handleRunClicked(); }

} // namespace CustomInterfaces
} // namespace MantidQt