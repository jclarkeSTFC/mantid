# Copyright &copy; 2024 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from mantid.simpleapi import CreateSampleWorkspace
from instrumentview.FullInstrumentViewModel import FullInstrumentViewModel
import unittest


class TestFullInstrumentViewModel(unittest.TestCase):
    def setUp(self):
        self._ws = CreateSampleWorkspace(OutputWorkspace="TestFullInstrumentViewModel")
        self._model = FullInstrumentViewModel(self._ws, False)

    def tearDown(self):
        self._ws.delete()

    def test_union_with_current_bin_min_max(self):
        current_min = self._model._bin_min
        current_max = self._model._bin_max
        self._model._union_with_current_bin_min_max(current_min - 1)
        self.assertEqual(self._model._bin_min, current_min - 1)
        self._model._union_with_current_bin_min_max(current_min)
        self.assertEqual(self._model._bin_min, current_min - 1)
        self.assertEqual(self._model._bin_max, current_max)
        self._model._union_with_current_bin_min_max(current_max + 1)
        self.assertEqual(self._model._bin_max, current_max + 1)
