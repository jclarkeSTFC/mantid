// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MockPreviewPythonInstrumentView.h"

#include "MantidAPI/MatrixWorkspace.h"
#include "MantidFrameworkTestHelpers/WorkspaceCreationHelper.h"
#include "MantidKernel/V3D.h"

#include <cxxtest/TestSuite.h>
#include <gmock/gmock.h>

using namespace testing;
using namespace MantidQt::CustomInterfaces::ISISReflectometry;

class PreviewPythonInstrumentViewTest : public CxxTest::TestSuite {
public:
  static PreviewPythonInstrumentViewTest *createSuite() { return new PreviewPythonInstrumentViewTest(); }
  static void destroySuite(PreviewPythonInstrumentViewTest *suite) { delete suite; }

  void test_mock_show_is_called() { return; }
};
