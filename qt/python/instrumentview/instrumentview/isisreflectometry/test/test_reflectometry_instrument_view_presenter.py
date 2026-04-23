# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy 2026 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter import ReflectometryInstrumentViewPresenter
from instrumentview.Projections.ProjectionType import ProjectionType


def _make_presenter():
    """Return a presenter with a fully mocked view."""
    mock_view = MagicMock()
    mock_view.main_plotter = MagicMock()
    mock_view.shape_overlay_manager = None
    return ReflectometryInstrumentViewPresenter(view=mock_view), mock_view


class TestReflectometryInstrumentViewPresenter(unittest.TestCase):
    def setUp(self):
        self._presenter, self._mock_view = _make_presenter()

    def test_view_is_stored(self):
        self.assertIs(self._presenter.view, self._mock_view)

    def test_model_is_none_initially(self):
        self.assertIsNone(self._presenter._model)

    def test_transform_is_none_initially(self):
        self.assertIsNone(self._presenter._transform)

    def test_rect_selected_detector_ids_is_empty_initially(self):
        self.assertEqual(self._presenter._rect_selected_detector_ids, [])

    def test_default_view_created_when_not_provided(self):
        """Passing no view should create a ReflectometryInstrumentViewView."""
        with mock.patch(
            "instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.ReflectometryInstrumentViewView"
        ) as mock_view_cls:
            presenter = ReflectometryInstrumentViewPresenter()
            mock_view_cls.assert_called_once()
            self.assertIs(presenter.view, mock_view_cls.return_value)

    @mock.patch("instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.ShapeRenderer")
    @mock.patch("instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.FullInstrumentViewModel")
    def test_update_workspace_calls_view_initialise(self, mock_model_cls, mock_renderer_cls):
        mock_ws = MagicMock()
        mock_model_cls.return_value.detector_positions = np.zeros((10, 3))
        mock_model_cls.return_value.flip_z = False
        mock_model_cls.return_value.detector_counts = np.zeros(10)
        mock_model_cls.return_value.is_2d_projection = True
        mock_renderer_cls.return_value.build_detector_mesh.return_value = MagicMock(bounds=(0, 1, 0, 1, 0, 1), transform=MagicMock())
        self._presenter._transform_mesh_to_fill_window = MagicMock(return_value=np.eye(4))
        self._presenter.update_workspace(mock_ws)
        self._mock_view.initialise.assert_called_once()

    @mock.patch("instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.ShapeRenderer")
    @mock.patch("instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.FullInstrumentViewModel")
    def test_update_workspace_sets_model(self, mock_model_cls, mock_renderer_cls):
        mock_ws = MagicMock()
        mock_model_cls.return_value.detector_positions = np.zeros((10, 3))
        mock_model_cls.return_value.flip_z = False
        mock_model_cls.return_value.detector_counts = np.zeros(10)
        mock_model_cls.return_value.is_2d_projection = True
        mock_renderer_cls.return_value.build_detector_mesh.return_value = MagicMock(bounds=(0, 1, 0, 1, 0, 1), transform=MagicMock())
        self._presenter._transform_mesh_to_fill_window = MagicMock(return_value=np.eye(4))
        self._presenter.update_workspace(mock_ws)
        mock_model_cls.assert_called_once_with(mock_ws)
        self.assertIsNotNone(self._presenter._model)

    @mock.patch("instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.ShapeRenderer")
    @mock.patch("instrumentview.isisreflectometry.ReflectometryInstrumentViewPresenter.FullInstrumentViewModel")
    def test_update_workspace_sets_cylindrical_projection(self, mock_model_cls, mock_renderer_cls):
        mock_ws = MagicMock()
        mock_model_cls.return_value.detector_positions = np.zeros((10, 3))
        mock_model_cls.return_value.flip_z = False
        mock_model_cls.return_value.detector_counts = np.zeros(10)
        mock_model_cls.return_value.is_2d_projection = True
        mock_renderer_cls.return_value.build_detector_mesh.return_value = MagicMock(bounds=(0, 1, 0, 1, 0, 1), transform=MagicMock())
        self._presenter._transform_mesh_to_fill_window = MagicMock(return_value=np.eye(4))
        self._presenter.update_workspace(mock_ws)
        self.assertEqual(mock_model_cls.return_value.projection_type, ProjectionType.CYLINDRICAL_Y)

    def test_reset_clears_model(self):
        self._presenter._model = MagicMock()
        self._presenter.reset()
        self.assertIsNone(self._presenter._model)
        self._mock_view.main_plotter.clear.assert_called_once()

    def test_reset_no_plotter_does_not_raise(self):
        self._mock_view.main_plotter = None
        self._presenter.reset()  # should not raise

    def test_plot_calls_render_when_model_set(self):
        self._presenter._model = MagicMock()
        self._presenter._render = MagicMock()
        self._presenter.plot()
        self._presenter._render.assert_called_once()

    def test_plot_no_op_when_no_model(self):
        self._presenter._render = MagicMock()
        self._presenter.plot()
        self._presenter._render.assert_not_called()

    def test_set_zoom_mode_removes_shape(self):
        self._presenter._model = MagicMock()
        self._presenter._model.is_2d_projection = True
        self._presenter._renderer = MagicMock()
        self._presenter.set_zoom_mode()
        self._mock_view.remove_shape.assert_called_once()

    def test_set_zoom_mode_calls_set_interactive_style(self):
        self._presenter._model = MagicMock()
        self._presenter._model.is_2d_projection = True
        self._presenter._renderer = MagicMock()
        self._presenter.set_zoom_mode()
        self._presenter._renderer.set_interactive_style.assert_called_once_with(self._mock_view.main_plotter, True)

    def test_set_zoom_mode_no_op_when_no_plotter(self):
        self._mock_view.main_plotter = None
        self._presenter._model = MagicMock()
        self._presenter._renderer = MagicMock()
        self._presenter.set_zoom_mode()
        self._presenter._renderer.set_interactive_style.assert_not_called()

    def test_set_zoom_mode_no_op_when_no_model(self):
        self._presenter._renderer = MagicMock()
        self._presenter.set_zoom_mode()
        self._presenter._renderer.set_interactive_style.assert_not_called()

    def test_set_select_rect_mode_calls_overlay_rectangle(self):
        self._presenter.set_select_rect_mode()
        self._mock_view.overlay_rectangle.assert_called_once()

    def test_set_select_rect_mode_passes_callback(self):
        self._presenter.set_select_rect_mode()
        call_kwargs = self._mock_view.overlay_rectangle.call_args[1]
        self.assertIn("on_shape_changed", call_kwargs)
        self.assertEqual(call_kwargs["on_shape_changed"], self._presenter._on_rect_shape_changed)

    def test_set_select_rect_mode_no_op_when_no_plotter(self):
        self._mock_view.main_plotter = None
        self._presenter.set_select_rect_mode()
        self._mock_view.overlay_rectangle.assert_not_called()

    def test_get_rect_selected_detector_ids_returns_copy(self):
        self._presenter._rect_selected_detector_ids = [1, 2, 3]
        result = self._presenter.get_rect_selected_detector_ids()
        self.assertEqual(result, [1, 2, 3])
        result.append(4)
        self.assertEqual(self._presenter._rect_selected_detector_ids, [1, 2, 3])

    def test_get_rect_selected_ids_empty_by_default(self):
        self.assertEqual(self._presenter.get_rect_selected_detector_ids(), [])

    def test_selected_detector_ids_returns_stored_ids(self):
        self._presenter._rect_selected_detector_ids = [10, 20]
        self.assertEqual(self._presenter.selected_detector_ids(), [10, 20])

    def test_on_rect_shape_changed_no_op_when_no_model(self):
        self._presenter._transform = np.eye(4)
        self._mock_view.shape_overlay_manager = MagicMock()
        self._presenter._on_rect_shape_changed()
        self.assertEqual(self._presenter._rect_selected_detector_ids, [])

    def test_on_rect_shape_changed_no_op_when_no_transform(self):
        self._presenter._model = MagicMock()
        self._mock_view.shape_overlay_manager = MagicMock()
        self._presenter._on_rect_shape_changed()
        self.assertEqual(self._presenter._rect_selected_detector_ids, [])

    def test_on_rect_shape_changed_no_op_when_no_manager(self):
        self._presenter._model = MagicMock()
        self._presenter._transform = np.eye(4)
        self._mock_view.shape_overlay_manager = None
        self._presenter._on_rect_shape_changed()
        self.assertEqual(self._presenter._rect_selected_detector_ids, [])

    def test_on_rect_shape_changed_empty_positions_gives_no_ids(self):
        mock_model = MagicMock()
        mock_model.detector_positions = np.zeros((0, 3))
        self._presenter._model = mock_model
        self._presenter._transform = np.eye(4)
        mock_manager = MagicMock()
        self._mock_view.shape_overlay_manager = mock_manager
        self._presenter._on_rect_shape_changed()
        self.assertEqual(self._presenter._rect_selected_detector_ids, [])

    def test_on_rect_shape_changed_selects_masked_detectors(self):
        """Detectors where get_shape_mask returns True should appear in the result."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        detector_ids = np.array([101, 102, 103])

        mock_model = MagicMock()
        mock_model.detector_positions = positions
        mock_model.all_detector_ids = detector_ids
        self._presenter._model = mock_model
        self._presenter._transform = np.eye(4)

        mock_manager = MagicMock()
        # Only the first and third detectors are inside the shape
        mock_manager.get_shape_mask.return_value = np.array([True, False, True])
        self._mock_view.shape_overlay_manager = mock_manager
        # No relay child expected
        self._mock_view.findChild.return_value = None

        self._presenter._on_rect_shape_changed()

        self.assertEqual(self._presenter._rect_selected_detector_ids, [101, 103])

    def test_on_rect_shape_changed_applies_transform(self):
        """The identity transform should leave positions unchanged."""
        positions = np.array([[1.0, 2.0, 0.0]])
        mock_model = MagicMock()
        mock_model.detector_positions = positions
        mock_model.all_detector_ids = np.array([55])
        self._presenter._model = mock_model
        self._presenter._transform = np.eye(4)

        mock_manager = MagicMock()
        mock_manager.get_shape_mask.return_value = np.array([True])
        self._mock_view.shape_overlay_manager = mock_manager
        self._mock_view.findChild.return_value = None

        self._presenter._on_rect_shape_changed()

        # Verify the coordinates passed to get_shape_mask are [position, 0] (no z component)
        called_coords = mock_manager.get_shape_mask.call_args[0][0]
        np.testing.assert_allclose(called_coords[0, :3], [1.0, 2.0, 0.0])
        self.assertEqual(self._presenter._rect_selected_detector_ids, [55])

    def test_scale_matrix_identity_when_scale_1(self):
        matrix = ReflectometryInstrumentViewPresenter._scale_matrix_relative_to_centre(np.array([0.0, 0.0, 0.0]), scale_x=1.0, scale_y=1.0)
        np.testing.assert_allclose(matrix, np.eye(4))

    def test_scale_matrix_scales_correctly(self):
        centre = np.array([2.0, 3.0, 0.0])
        matrix = ReflectometryInstrumentViewPresenter._scale_matrix_relative_to_centre(centre, scale_x=2.0, scale_y=3.0)
        # A point at the centre should map to itself
        point = np.append(centre, 1.0)
        result = matrix @ point
        np.testing.assert_allclose(result[:3], centre)

    def test_scale_matrix_scales_origin_point(self):
        """Point at origin should be scaled relative to centre."""
        centre = np.array([1.0, 1.0, 0.0])
        matrix = ReflectometryInstrumentViewPresenter._scale_matrix_relative_to_centre(centre, scale_x=2.0, scale_y=2.0)
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        result = matrix @ origin
        # Origin is 1 unit away from centre; after 2x scale it should be 2 units away
        np.testing.assert_allclose(result[:2], [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
