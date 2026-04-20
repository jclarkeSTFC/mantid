# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
import pyvista as pv

from pyvistaqt import BackgroundPlotter

from qtpy.QtWidgets import QHBoxLayout, QWidget

from mantidqt.utils.qt.qappthreadcall import run_on_qapp_thread


@run_on_qapp_thread()
class ReflectometryInstrumentViewView(QWidget):
    """A minimal instrument view for ISISReflectometry.

    Contains only a pyvista BackgroundPlotter for 3D instrument rendering.
    The BackgroundPlotter is created lazily via ``initialise()`` to avoid
    OpenGL context errors when VTK tries to render before the widget is
    embedded in its final layout.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        pv.global_theme.background = "black"
        pv.global_theme.font.color = "white"

        self._initialised = False
        self.main_plotter = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def initialise(self):
        """Create the BackgroundPlotter.

        Must be called after this widget is embedded in its final layout.
        """
        if self._initialised:
            return
        self._initialised = True

        self.main_plotter = BackgroundPlotter(show=False, menu_bar=False, toolbar=False, off_screen=False)
        self.layout().addWidget(self.main_plotter.app_window)

    def closeEvent(self, event):
        super().closeEvent(event)
        if self.main_plotter is not None:
            self.main_plotter.close()
