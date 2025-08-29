# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2025 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from instrumentview.Detectors import DetectorInfo
import instrumentview.Projections.SphericalProjection as iv_spherical
import instrumentview.Projections.CylindricalProjection as iv_cylindrical

from mantid.dataobjects import Workspace2D
from mantid.simpleapi import CreateDetectorTable
import numpy as np
import math


class FullInstrumentViewModel:
    """Model for the Instrument View Window. Will calculate detector positions, indices, and integrated counts that give the colours"""

    _sample_position = np.array([0, 0, 0])
    _source_position = np.array([0, 0, 0])
    _invalid_index = -1
    _data_min = 0.0
    _data_max = 0.0

    def __init__(self, workspace: Workspace2D):
        """For the given workspace, calculate detector positions, the map from detector indices to workspace indices, and integrated
        counts. Optionally will draw detector geometry, e.g. rectangular bank or tube instead of points."""
        self._workspace = workspace

    def setup(self):
        self._detector_info = self._workspace.detectorInfo()
        self._component_info = self._workspace.componentInfo()
        self._sample_position = np.array(self._component_info.samplePosition()) if self._component_info.hasSample() else np.zeros(3)
        has_source = self._workspace.getInstrument().getSource() is not None
        self._source_position = np.array(self._component_info.sourcePosition()) if has_source else np.array([0, 0, 0])

        detector_info_table = CreateDetectorTable(self._workspace, IncludeDetectorPosition=True, StoreInADS=False)

        # Might have comma-separated multiple detectors, choose first one in the string in that case
        first_numbers = np.char.split(detector_info_table.column("Detector ID(s)"), sep=",")
        self._detector_ids = np.array([int(x[0]) for x in first_numbers])

        detector_positions = detector_info_table.column("Position")
        self._spherical_positions = np.array([pos.getSpherical() for pos in detector_positions])
        self._detector_positions = np.array(detector_positions)

        self._counts = np.zeros_like(self._detector_ids)
        self._workspace_indices = np.array(detector_info_table.column("Index")).astype(int)

        self._is_monitor = np.array(detector_info_table.column("Monitor"))
        spectrum_number = np.array(detector_info_table.column("Spectrum No"))
        self._is_valid = (self._is_monitor == "no") & (spectrum_number != -1)
        self._monitor_positions = self._detector_positions[self._is_monitor == "yes"]
        self._detector_projection_positions = np.zeros_like(self._detector_positions)
        self._detector_is_picked = np.full(len(self._detector_ids[self._is_valid]), False)

        data_x = self._workspace.extractX()
        self._bin_min = np.min(data_x[:, 0])
        self._bin_max = np.max(data_x[:, -1])

        self.update_time_of_flight_range(self._bin_min, self._bin_max, True)

    def _union_with_current_bin_min_max(self, bin_edge: float) -> None:
        """Expand current bin limits to include new bin edge"""
        if not math.isinf(bin_edge):
            if bin_edge < self._bin_min:
                self._bin_min = bin_edge
            elif bin_edge > self._bin_max:
                self._bin_max = bin_edge

    def update_time_of_flight_range(self, tof_min: float, tof_max: float, entire_range: bool = False) -> None:
        workspace_indices = self._workspace_indices[self._is_valid]
        new_detector_counts = np.array(
            self._workspace.getIntegratedCountsForWorkspaceIndices(
                workspace_indices, len(workspace_indices), float(tof_min), float(tof_max), entire_range
            ),
            dtype=int,
        )
        self._data_max = max(new_detector_counts)
        self._data_min = min(new_detector_counts)
        self._counts[self._is_valid] = new_detector_counts

    def workspace(self) -> Workspace2D:
        return self._workspace

    def default_projection(self) -> str:
        return self._workspace.getInstrument().getDefaultView()

    def sample_position(self) -> np.ndarray:
        return self._sample_position

    def detector_positions(self) -> np.ndarray:
        return self._detector_positions[self._is_valid]

    def detector_projection_positions(self) -> np.ndarray:
        return self._detector_projection_positions[self._is_valid]

    def negate_picked_visibility(self, indices: list[int] | np.ndarray) -> None:
        self._detector_is_picked[indices] = ~self._detector_is_picked[indices]

    def clear_all_picked_detectors(self) -> None:
        self._detector_is_picked.fill(False)

    def picked_visibility(self) -> np.ndarray:
        return self._detector_is_picked.astype(int)

    def picked_detector_ids(self) -> np.ndarray:
        return self._detector_ids[self._is_valid][self._detector_is_picked]

    def picked_workspace_indices(self) -> np.ndarray:
        return self._workspace_indices[self._is_valid][self._detector_is_picked]

    def detector_counts(self) -> np.ndarray:
        return self._counts[self._is_valid]

    def detector_ids(self) -> np.ndarray:
        return self._detector_ids[self._is_valid]

    def data_limits(self) -> list:
        return [self._data_min, self._data_max]

    def bin_limits(self) -> list:
        return [self._bin_min, self._bin_max]

    def monitor_positions(self) -> np.ndarray:
        return self._monitor_positions
        # return self._detector_positions[self._is_monitor == 'yes']
        # return self._detector_positions[[self._detector_info.isMonitor(i) for i in range(len(self._detector_ids))]]

    def picked_detectors_info_text(self) -> list[DetectorInfo]:
        """For the specified detector, extract info that can be displayed in the View, and wrap it all up in a DetectorInfo class"""

        picked_ws_indices = self._workspace_indices[self._is_valid][self._detector_is_picked]
        picked_ids = self._detector_ids[self._is_valid][self._detector_is_picked]
        picked_xyz_positions = self._detector_positions[self._is_valid][self._detector_is_picked]
        picked_spherical_positions = self._spherical_positions[self._is_valid][self._detector_is_picked]
        picked_counts = self._counts[self._is_valid][self._detector_is_picked]

        picked_info = []
        for i, ws_index in enumerate(picked_ws_indices):
            ws_detector = self._workspace.getDetector(int(ws_index))
            name = ws_detector.getName()
            component_path = ws_detector.getFullName()
            det_info = DetectorInfo(
                name, picked_ids[i], ws_index, picked_xyz_positions[i], picked_spherical_positions[i], component_path, int(picked_counts[i])
            )
            picked_info.append(det_info)
        return picked_info

    def calculate_projection(self, is_spherical: bool, axis: list[int]):
        """Calculate the 2D projection with the specified axis. Can be either cylindrical or spherical."""
        root_position = np.array(self._component_info.position(0))
        projection = (
            iv_spherical.SphericalProjection(self._sample_position, root_position, self._detector_positions, np.array(axis))
            if is_spherical
            else iv_cylindrical.CylindricalProjection(self._sample_position, root_position, self._detector_positions, np.array(axis))
        )
        self._detector_projection_positions[:, :2] = projection.positions()  # Assign only x and y coordinate
        return self._detector_projection_positions
