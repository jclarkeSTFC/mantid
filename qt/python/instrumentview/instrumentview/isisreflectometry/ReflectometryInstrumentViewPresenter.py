# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +

from instrumentview.FullInstrumentViewModel import FullInstrumentViewModel
from instrumentview.isisreflectometry.ReflectometryInstrumentViewView import ReflectometryInstrumentViewView
from instrumentview.renderers.shape_renderer import ShapeRenderer
from instrumentview.Projections.ProjectionType import ProjectionType
from qtpy.QtCore import QObject, QMetaObject, Qt
import numpy as np
from vtk import vtkCoordinate
from typing import Optional


class ReflectometryInstrumentViewPresenter:
    """Presenter that wraps a pyvista-based instrument view for the
    ISISReflectometry preview tab.

    Renders the instrument directly using FullInstrumentViewModel and
    PointCloudRenderer, without the full FullInstrumentViewPresenter
    (no peaks, masking, grouping, component tree, etc.).
    """

    _COUNTS_LABEL = "Integrated Counts"
    _VISIBLE_LABEL = "Visible Picked"

    def __init__(self, view: Optional[ReflectometryInstrumentViewView] = None):
        self.view = view or ReflectometryInstrumentViewView()
        self._model: Optional[FullInstrumentViewModel] = None
        self._transform: Optional[np.ndarray] = None
        self._rect_selected_detector_ids: list[int] = []

    def update_workspace(self, workspace):
        """Set up the model from the workspace and render the instrument."""
        self.view.initialise()
        self._model = FullInstrumentViewModel(workspace)
        self._model.setup()
        self._model.projection_type = ProjectionType.CYLINDRICAL_Y
        self._renderer = ShapeRenderer(workspace)
        self._renderer.precompute()
        self._render()

    def reset(self):
        """Clear and re-initialise the view."""
        if self.view.main_plotter is not None:
            self.view.main_plotter.clear()
        self._model = None

    def plot(self):
        """Re-render the current instrument."""
        if self._model is not None:
            self._render()

    def set_zoom_mode(self):
        """Set the plotter interaction to zoom/pan mode."""
        self.view.remove_shape()
        plotter = self.view.main_plotter
        if plotter is not None and self._model is not None:
            self._renderer.set_interactive_style(plotter, self._model.is_2d_projection)

    def set_edit_mode(self):
        """Set the plotter interaction to edit/picking mode."""
        self.view.remove_shape()

    def set_select_rect_mode(self):
        """Set the plotter interaction to rectangle selection mode."""
        plotter = self.view.main_plotter
        if plotter is not None:
            self.view.overlay_rectangle(on_shape_changed=self._on_rect_shape_changed)

    def get_rect_selected_detector_ids(self) -> list[int]:
        """Return the detector IDs currently within the rectangle selection."""
        return list(self._rect_selected_detector_ids)

    def _on_rect_shape_changed(self) -> None:
        """Recompute which detectors fall inside the rectangle and store their IDs."""
        if self._model is None or self._transform is None:
            return
        mgr = self.view.shape_overlay_manager
        if mgr is None:
            return
        positions = self._model.detector_positions
        if len(positions) == 0:
            self._rect_selected_detector_ids = []
            return
        ones = np.ones((len(positions), 1))
        transformed = (self._transform @ np.hstack([positions, ones]).T).T[:, :3]
        mask = mgr.get_shape_mask(transformed)
        all_ids = self._model.all_detector_ids
        self._rect_selected_detector_ids = [int(all_ids[i]) for i in np.where(mask)[0]]

        relay = self.view.findChild(QObject, "ShapeChangedRelay")
        if relay is not None:
            QMetaObject.invokeMethod(relay, "notify", Qt.DirectConnection)

    def selected_detector_ids(self) -> list[int]:
        return self._rect_selected_detector_ids

    def _render(self):
        """Render the instrument into the view's plotter."""
        if self._model is None:
            return

        plotter = self.view.main_plotter
        plotter.clear()

        self._detector_mesh = self._renderer.build_detector_mesh(self._model.detector_positions, self._model.flip_z, self._model)
        self._renderer.set_detector_scalars(self._detector_mesh, self._model.detector_counts, self._COUNTS_LABEL)
        self._renderer.add_detector_mesh_to_plotter(plotter, self._detector_mesh, scalars=self._COUNTS_LABEL, show_scalar_bar=False)

        self._transform = self._transform_mesh_to_fill_window()
        self._detector_mesh.transform(self._transform, inplace=True)

        self._renderer.set_parallel_view(plotter)
        plotter.reset_camera()
        self._renderer.set_interactive_style(plotter, self._model.is_2d_projection)

    def _transform_mesh_to_fill_window(self) -> np.ndarray:
        xmin, xmax, ymin, ymax, zmin, zmax = self._detector_mesh.bounds
        min_point = np.array([xmin, ymin, zmin])
        max_point = np.array([xmax, ymax, zmax])

        # Convert to display coordinates (pixels)
        plotter = self.view.main_plotter
        coordinate = vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        display_coords = []
        for p in (min_point, max_point):
            coordinate.SetValue(*p)
            display_coords.append(coordinate.GetComputedDisplayValue(plotter.renderer))

        mesh_width = display_coords[1][0] - display_coords[0][0]
        mesh_height = display_coords[1][1] - display_coords[0][1]

        window_width, window_height = plotter.window_size

        # Safeguard against division by zero
        mesh_width = mesh_width if mesh_width > 0 else window_width
        mesh_height = mesh_height if mesh_height > 0 else window_height

        return self._scale_matrix_relative_to_centre((min_point + max_point) / 2, window_width / mesh_width, window_height / mesh_height)

    @staticmethod
    def _scale_matrix_relative_to_centre(centre, scale_x=1.0, scale_y=1.0) -> np.ndarray:
        # Translate to centre, scale, translate back
        # The matrix below is the product of those three transformations
        c_x, c_y, _ = centre
        return np.array([[scale_x, 0, 0, c_x * (1 - scale_x)], [0, scale_y, 0, c_y * (1 - scale_y)], [0, 0, 1, 0], [0, 0, 0, 1]])
