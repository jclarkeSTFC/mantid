# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2025 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from contextlib import suppress
from typing import Callable, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from vtkmodules.vtkRenderingCore import vtkPointPicker

from instrumentview.renderers.base_renderer import InstrumentRenderer


class PointCloudRenderer(InstrumentRenderer):
    """Renders all detectors as a point cloud with spherical point sprites.

    This is the original (fast) rendering mode. Each detector is a single
    point; VTK renders it as a screen-space sphere of constant pixel size.
    """

    _DETECTOR_POINT_SIZE = 15
    _PICKABLE_POINT_SIZE = 30
    _MASKED_COLOUR = (0.25, 0.25, 0.25)

    def __init__(self):
        self._hover_observer_style = None
        self._hover_observer_tag = None

    # ------------------------------------------------------------------ build
    def build_detector_mesh(self, positions: np.ndarray, flip_z: bool, model=None) -> pv.PolyData:
        return pv.PolyData(positions)

    def build_pickable_mesh(self, positions: np.ndarray, flip_z: bool) -> pv.PolyData:
        return pv.PolyData(positions)

    def build_masked_mesh(self, positions: np.ndarray, flip_z: bool, model=None) -> pv.PolyData:
        return pv.PolyData(positions)

    # ------------------------------------------------------------ add to plot
    def add_detector_mesh_to_plotter(
        self, plotter: BackgroundPlotter, mesh: pv.PolyData, is_projection: bool, scalars: Optional[str] = None
    ) -> None:
        scalar_bar_args = dict(interactive=True, vertical=False, title_font_size=15, label_font_size=12) if scalars is not None else None
        plotter.add_mesh(
            mesh,
            pickable=False,
            scalars=scalars,
            render_points_as_spheres=True,
            point_size=self._DETECTOR_POINT_SIZE,
            scalar_bar_args=scalar_bar_args,
        )

        if plotter.off_screen:
            return

    def add_pickable_mesh_to_plotter(self, plotter: BackgroundPlotter, mesh: pv.PolyData, scalars) -> None:
        plotter.add_mesh(
            mesh,
            scalars=scalars,
            opacity=[0.0, 0.3],
            clim=[0, 1],
            show_scalar_bar=False,
            pickable=True,
            cmap="Oranges",
            point_size=self._PICKABLE_POINT_SIZE,
            render_points_as_spheres=True,
        )

    def add_masked_mesh_to_plotter(self, plotter: BackgroundPlotter, mesh: pv.PolyData) -> None:
        if mesh.number_of_points == 0:
            return
        plotter.add_mesh(
            mesh,
            color=self._MASKED_COLOUR,
            pickable=False,
            render_points_as_spheres=True,
            point_size=self._DETECTOR_POINT_SIZE,
        )

    # --------------------------------------------------------------- picking
    def enable_picking(self, plotter: BackgroundPlotter, callback: Callable[[int], None]) -> None:
        """Set up left-click point picking.  *callback* receives ``(detector_index: int)``."""
        plotter.disable_picking()

        if plotter.off_screen:
            return

        picking_tolerance = 0.01
        picker = vtkPointPicker()
        picker.SetTolerance(picking_tolerance)
        interactor = plotter.iren

        def _on_left_button_press(obj, event):
            """Handle left mouse button press for picking."""
            # Get the current mouse position from the interactor
            x, y = interactor.get_event_position()
            # Perform the pick operation
            pick_result = picker.Pick(x, y, 0, plotter.renderer)
            if pick_result > 0:
                # Get the picked point ID
                point_id = picker.GetPointId()
                if point_id >= 0:
                    callback(point_id)

        # Register callback for left button press
        plotter.iren.style.AddObserver("LeftButtonPressEvent", _on_left_button_press)

    def enable_hover_picking(self, plotter: BackgroundPlotter, callback: Callable[[int], None]) -> None:
        """Register a mouse-move observer that fires *callback* with the local detector index under the cursor."""
        self.disable_hover_picking(plotter)

        if plotter.off_screen:
            return

        picking_tolerance = 0.01
        picker = vtkPointPicker()
        picker.SetTolerance(picking_tolerance)
        interactor = plotter.iren

        def _on_mouse_move(obj, event):
            x, y = interactor.get_event_position()
            if picker.Pick(x, y, 0, plotter.renderer) > 0:
                point_id = picker.GetPointId()
                if point_id >= 0:
                    callback(point_id)

        style = plotter.iren.style
        self._hover_observer_style = style
        self._hover_observer_tag = style.AddObserver("MouseMoveEvent", _on_mouse_move)

    def disable_hover_picking(self, plotter: BackgroundPlotter) -> None:
        """Remove any registered hover-pick mouse-move observer."""
        if self._hover_observer_style is not None and self._hover_observer_tag is not None:
            with suppress(Exception):
                self._hover_observer_style.RemoveObserver(self._hover_observer_tag)
        self._hover_observer_style = None
        self._hover_observer_tag = None

    # -------------------------------------------------------------- scalars
    def set_detector_scalars(self, mesh: pv.PolyData, counts: np.ndarray, label: str) -> None:
        mesh.point_data[label] = counts

    def set_pickable_scalars(self, mesh: pv.PolyData, visibility: np.ndarray, label: str) -> None:
        mesh.point_data[label] = visibility
