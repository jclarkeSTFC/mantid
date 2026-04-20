// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "Common/DllConfig.h"
#include "IPreviewInstrumentDisplay.h"

#include <QLayout>

namespace MantidQt::CustomInterfaces::ISISReflectometry {

class MANTIDQT_ISISREFLECTOMETRY_DLL IPreviewPythonInstrumentView : public IPreviewInstrumentDisplay {
public:
  ~IPreviewPythonInstrumentView() override = default;
  virtual void show() = 0;
  virtual void close() = 0;
  virtual void setLayout(QLayout *layout) = 0;
};

} // namespace MantidQt::CustomInterfaces::ISISReflectometry
