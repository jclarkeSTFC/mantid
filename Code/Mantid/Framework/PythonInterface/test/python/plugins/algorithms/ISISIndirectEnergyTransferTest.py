#pylint: disable=too-many-public-methods,invalid-name

import unittest
from mantid.simpleapi import *
from mantid.api import *


class ISISIndirectEnergyTransferTest(unittest.TestCase):

    def test_basic_reduction_completes(self):
        """
        Sanity test to ensure the most basic reduction actually completes.
        """

        wks = ISISIndirectEnergyTransfer(InputFiles=['IRS26176.RAW'],
                                         Instrument='IRIS',
                                         Analyser='graphite',
                                         Reflection='002',
                                         SpectraRange=[3, 53])

        self.assertTrue(isinstance(wks, WorkspaceGroup), 'Result workspace should be a workspace group.')
        self.assertEqual(wks.getNames()[0], 'IRS26176_graphite002_red')


    def test_reduction_with_detailed_balance_completes(self):
        """
        Sanity test to ensure a reduction using detailed balance option
        completes.
        """

        wks = ISISIndirectEnergyTransfer(InputFiles=['IRS26176.RAW'],
                                         Instrument='IRIS',
                                         Analyser='graphite',
                                         Reflection='002',
                                         SpectraRange=[3, 53],
                                         DetailedBalance='300')

        self.assertTrue(isinstance(wks, WorkspaceGroup), 'Result workspace should be a workspace group.')
        self.assertEqual(wks.getNames()[0], 'IRS26176_graphite002_red')


    def test_reduction_with_map_file_completes(self):
        """
        Sanity test to ensure a reduction using a mapping/grouping file
        completes.
        """

        wks = ISISIndirectEnergyTransfer(InputFiles=['OSI97919.raw'],
                                         Instrument='OSIRIS',
                                         Analyser='graphite',
                                         Reflection='002',
                                         SpectraRange=[963, 1004])

        self.assertTrue(isinstance(wks, WorkspaceGroup), 'Result workspace should be a workspace group.')
        self.assertEqual(wks.getNames()[0], 'OSI97919_graphite002_red')


    def test_instrument_validation_failure(self):
        """
        Tests that an invalid instrument configuration causes the validation to
        fail.
        """

        self.assertRaises(RuntimeError,
                          ISISIndirectEnergyTransfer,
                          OutputWorkspace='__ISISIndirectEnergyTransferTest_ws',
                          InputFiles=['IRS26176.RAW'],
                          Instrument='IRIS',
                          Analyser='graphite',
                          Reflection='006',
                          SpectraRange=[3, 53])


    def test_group_workspace_validation_failure(self):
        """
        Tests that validation fails when Workspace is selected as the
        GroupingMethod but no workspace is provided.
        """

        self.assertRaises(RuntimeError,
                          ISISIndirectEnergyTransfer,
                          OutputWorkspace='__ISISIndirectEnergyTransferTest_ws',
                          InputFiles=['IRS26176.RAW'],
                          Instrument='IRIS',
                          Analyser='graphite',
                          Reflection='002',
                          SpectraRange=[3, 53],
                          GroupingMethod='Workspace')


if __name__ == '__main__':
    unittest.main()
