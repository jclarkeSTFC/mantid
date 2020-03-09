# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
class Presenter(object):

    def __init__(self, view, colours):
        self.view = view
        self.view.setColours(colours)

    def getPlotInfo(self):
        return str(self.view.getColour()), self.view.getFreq(), self.view.getPhase()

    def getGridLines(self):
        return self.view.getGridLines()
