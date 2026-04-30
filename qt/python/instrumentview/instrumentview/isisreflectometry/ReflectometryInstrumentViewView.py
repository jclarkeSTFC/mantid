# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
import pyvista as pv

from pyvistaqt import BackgroundPlotter

from qtpy.QtWidgets import QHBoxLayout, QWidget
from qtpy.QtCore import QTimer

from mantidqt.utils.qt.qappthreadcall import run_on_qapp_thread
from instrumentview.ShapeOverlayManager import ShapeOverlayManager
from instrumentview.ShapeWidgets import RectangleSelectionShape


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
        self._shape_overlay_manager: ShapeOverlayManager | None = None
        self._on_resize_callback = None

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

    def set_on_resize_callback(self, callback) -> None:
        """Set a callable invoked (deferred) on every resize, or None to clear."""
        self._on_resize_callback = callback

    def resizeEvent(self, event):
        """Re-fit the camera (and fill transform) when the widget is resized.

        Qt propagates the resize to child widgets synchronously inside
        super().resizeEvent(), so QVTKRenderWindowInteractor.resizeEvent runs
        and calls vtkRenderWindow.SetSize(w, h) before we return.  We then
        defer one event-loop cycle (QTimer.singleShot(0, ...)) so that VTK's
        own paint cycle completes before we reset the camera or recompute the
        fill transform against the now-correct render window dimensions.
        """
        super().resizeEvent(event)
        if self.main_plotter is None:
            return
        callback = self._on_resize_callback if self._on_resize_callback is not None else self.main_plotter.reset_camera
        QTimer.singleShot(0, callback)

    @property
    def shape_overlay_manager(self) -> ShapeOverlayManager | None:
        """The active ShapeOverlayManager, or None if no shape is shown."""
        return self._shape_overlay_manager

    def overlay_rectangle(self, on_shape_changed=None) -> None:
        if self.main_plotter is None:
            return
        if self._shape_overlay_manager is None:
            self._shape_overlay_manager = ShapeOverlayManager(self.main_plotter)
        if on_shape_changed is not None:
            self._shape_overlay_manager.set_on_shape_changed(on_shape_changed)
        shape = RectangleSelectionShape(cx=0.5, cy=0.5, half_w=1.0 / 6.0, half_h=1.0 / 6.0)
        self._shape_overlay_manager.add_shape(shape)

    def remove_shape(self) -> None:
        """Remove the current selection shape overlay."""
        if self._shape_overlay_manager is not None:
            self._shape_overlay_manager.remove_shape()
            self._shape_overlay_manager = None

    def closeEvent(self, event):
        self.remove_shape()
        super().closeEvent(event)
        if self.main_plotter is not None:
            self.main_plotter.close()
