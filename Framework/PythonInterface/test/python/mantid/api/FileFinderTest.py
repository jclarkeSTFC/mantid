# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
from __future__ import (absolute_import, division, print_function)

import os
import unittest

from mantid.api import FileFinder


class FileFinderTest(unittest.TestCase):

    def test_full_path_returns_an_absolute_path_and_the_files_exists(self):
        path = FileFinder.getFullPath("CNCS_7860_event.nxs")
        self.assertTrue(len(path) > 0)
        # We can't be sure what the full path is in general but it should certainly exist!
        self.assertTrue(os.path.exists(path))

    def test_find_runs_returns_absolute_paths_of_given_runs(self):
        runs = FileFinder.findRuns("CNCS7860")
        self.assertTrue(len(runs) == 1)
        # We can't be sure what the full path is in general but it should certainly exist!
        self.assertTrue(os.path.exists(runs[0]))

    def test_that_find_runs_accepts_a_list_of_strings(self):
        try:
            runs = FileFinder.findRuns("CNCS7860", [".nxs", ".txt"])
        except Exception as e:
            self.assertFalse(True, "Expected findRuns to accept list of strings as input. {} error was raised "
                                    "with message {}".format(type(e).__name__, str(e)))
        else:
            # Confirm that it works as above
            self.assertTrue(len(runs) == 1)
            self.assertTrue(os.path.exists(runs[0]))


if __name__ == '__main__':
    unittest.main()
