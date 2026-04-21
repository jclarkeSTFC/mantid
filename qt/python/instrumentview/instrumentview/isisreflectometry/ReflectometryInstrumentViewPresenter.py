# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2026 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +

from instrumentview.FullInstrumentViewModel import FullInstrumentViewModel
from instrumentview.isisreflectometry.ReflectometryInstrumentViewView import ReflectometryInstrumentViewView
from instrumentview.renderers.shape_renderer import ShapeRenderer


class ReflectometryInstrumentViewPresenter:
    """Presenter that wraps a pyvista-based instrument view for the
    ISISReflectometry preview tab.

    Renders the instrument directly using FullInstrumentViewModel and
    PointCloudRenderer, without the full FullInstrumentViewPresenter
    (no peaks, masking, grouping, component tree, etc.).
    """

    _COUNTS_LABEL = "Integrated Counts"
    _VISIBLE_LABEL = "Visible Picked"

    def __init__(self, view=None):
        self.view = view or ReflectometryInstrumentViewView()
        self._model = None

    def update_workspace(self, workspace):
        """Set up the model from the workspace and render the instrument."""
        self.view.initialise()
        self._model = FullInstrumentViewModel(workspace)
        self._model.setup()
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
        plotter = self.view.main_plotter
        if plotter is not None and self._model is not None:
            self._renderer.set_interactive_style(plotter, self._model.is_2d_projection)

    def set_edit_mode(self):
        """Set the plotter interaction to edit/picking mode."""
        plotter = self.view.main_plotter
        if plotter is not None:
            self._renderer.enable_picking(plotter, callback=self._on_detector_picked)

    def set_select_rect_mode(self):
        """Set the plotter interaction to rectangle selection mode."""
        plotter = self.view.main_plotter
        if plotter is not None:
            self.view.set_shape("rectangle")

    def get_selected_detectors(self):
        """Return a list of picked detector indices."""
        if self._model is None:
            return []
        return self._model.picked_detector_ids.tolist()

    def det_indices_to_det_ids(self, det_indices):
        """Convert detector indices to detector IDs."""
        if self._model is None:
            return []
        all_ids = self._model.all_detector_ids
        return [int(all_ids[i]) for i in det_indices]

    def _on_detector_picked(self, detector_index):
        """Callback for when a detector is picked."""
        if self._model is not None:
            self._model.update_point_picked_detectors(detector_index, expand_to_parent_subtree=False)
            self._render()

    def _render(self):
        """Render the instrument into the view's plotter."""
        plotter = self.view.main_plotter
        plotter.clear()

        det_mesh = self._renderer.build_detector_mesh(self._model.detector_positions, self._model.flip_z, self._model)
        self._renderer.set_detector_scalars(det_mesh, self._model.detector_counts, self._COUNTS_LABEL)
        self._renderer.add_detector_mesh_to_plotter(
            plotter, det_mesh, is_projection=self._model.is_2d_projection, scalars=self._COUNTS_LABEL
        )

        pick_mesh = self._renderer.build_pickable_mesh(self._model.detector_positions, self._model.flip_z)
        self._renderer.set_pickable_scalars(pick_mesh, self._model.picked_visibility, self._VISIBLE_LABEL)
        self._renderer.add_pickable_mesh_to_plotter(plotter, pick_mesh, scalars=self._VISIBLE_LABEL)

        mask_mesh = self._renderer.build_masked_mesh(self._model.masked_positions, self._model.flip_z, self._model)
        self._renderer.add_masked_mesh_to_plotter(plotter, mask_mesh)

        self._renderer.set_parallel_view(plotter)
        plotter.reset_camera()
        self._renderer.set_interactive_style(plotter, self._model.is_2d_projection)
