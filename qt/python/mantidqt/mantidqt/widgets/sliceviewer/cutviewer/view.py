# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2022 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.
#
from qtpy.QtWidgets import QVBoxLayout, QWidget, QTableWidget, QHeaderView, QTableWidgetItem
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from mantid.plots.plotfunctions import create_subplots
from mantidqt.MPLwidgets import FigureCanvas
from mantidqt.plotting.mantid_navigation_toolbar import MantidNavigationToolbar
import matplotlib.text as text
import numpy as np
from mantid.simpleapi import AnalysisDataService as ADS
from mantid.kernel import SpecialCoordinateSystem


class CutViewerView(QWidget):
    """Displays a table view of the PeaksWorkspace along with controls
    to interact with the peaks.
    """
    def __init__(self, presenter, canvas, frame, parent=None):
        """
        :param painter: An object responsible for drawing the representation of the cut
        :param sliceinfo_provider: An object responsible for providing access to current slice information
        :param parent: An optional parent widget
        """
        super().__init__(parent)
        self.presenter = presenter
        self.layout = None
        self.figure_layout = None
        self.table = None
        self.figure = None
        self.cut_rep = None
        self.canvas = canvas
        self.frame = frame
        self._setup_ui()
        self._init_slice_table()
        self.table.cellChanged.connect(self.on_cell_changed)

    def hide(self):
        super().hide()
        if self.cut_rep is not None:
            self.cut_rep.remove()
            self.cut_rep = None

    # signals

    def on_cell_changed(self, irow, icol):
        self.set_bin_params(*self.presenter.validate_bin_params(irow, icol))
        self.presenter.update_cut()

    # getters

    def get_step(self, irow):
        return float(self.table.item(irow, 6).text())

    def get_bin_params(self):
        vectors = np.zeros((3, 3), dtype=float)
        extents = np.zeros((2, 3), dtype=float)
        nbins = np.zeros(3, dtype=int)
        for ivec in range(vectors.shape[0]):
            for icol in range(vectors.shape[0]):
                vectors[ivec, icol] = float(self.table.item(ivec, icol).text())
            extents[0, ivec] = float(self.table.item(ivec, 3).text())  # start
            extents[1, ivec] = float(self.table.item(ivec, 4).text())  # stop
            nbins[ivec] = int(self.table.item(ivec, 5).text())
        return vectors, extents, nbins

    # setters

    def set_vector(self, irow, vector):
        for icol in range(len(vector)):
            self.table.item(irow, icol).setData(Qt.EditRole, float(vector[icol]))

    def set_extent(self, irow, start=None, stop=None):
        if start is not None:
            self.table.item(irow, 3).setData(Qt.EditRole, float(start))
        if stop is not None:
            self.table.item(irow, 4).setData(Qt.EditRole, float(stop))

    def set_step(self, irow, step):
        self.table.item(irow, 6).setData(Qt.EditRole, float(step))

    def update_step(self, irow):
        _, extents, nbins = self.get_bin_params()
        self.set_step(irow, (extents[1, irow]-extents[0, irow])/nbins[irow])

    def set_nbin(self, irow, nbin):
        self.table.item(irow, 5).setData(Qt.EditRole, int(nbin))
        self.update_step(irow)

    def set_bin_params(self, vectors, extents, nbins):
        self.table.blockSignals(True)
        for irow in range(len(nbins)):
            self.set_vector(irow, vectors[irow])
            self.set_extent(irow, *extents[:, irow])
            self.set_nbin(irow, nbins[irow])  # do this last as step automatically updated given extents
        self.table.blockSignals(False)
        self.plot_cut_representation()
        return vectors, extents, nbins

    def set_slicepoint(self, slicept, width):
        self.table.blockSignals(True)
        self.set_extent(2, slicept - width/2, slicept + width/2)
        self.set_step(2, width)
        self.table.blockSignals(False)

    # plotting

    def plot_cut_ws(self, wsname):
        if len(self.figure.axes[0].tracked_workspaces) == 0:
            self.figure.axes[0].errorbar(ADS.retrieve(wsname), wkspIndex=None, marker='o', capsize=2, color='k',
                                         markersize=3)
        self._format_cut_figure()
        self.figure.canvas.draw()

    def plot_cut_representation(self):
        if self.cut_rep is not None:
            self.cut_rep.remove()
        self.cut_rep = CutRepresentation(self.canvas, self.presenter.update_bin_params_from_cut_representation,
                                         *self.presenter.get_cut_representation_parameters())

    # private api
    def _setup_ui(self):
        """
        Arrange the widgets on the window
        """
        self.layout = QVBoxLayout()
        self.layout.sizeHint()
        self.layout.setContentsMargins(5, 0, 0, 0)
        self.layout.setSpacing(0)

        self._setup_table_widget()
        self._setup_figure_widget()
        self.setLayout(self.layout)

    def _setup_table_widget(self):
        """
        Make a table showing
        :return: A QTableWidget object which will contain plot widgets
        """
        table_widget = QTableWidget(3, 7, self)
        table_widget.setVerticalHeaderLabels(['u1', 'u2', 'u3'])
        col_headers = ['a*', 'b*', 'c*'] if self.frame == SpecialCoordinateSystem.HKL else ['Qx', 'Qy', 'Qz']
        col_headers.extend(['start', 'stop', 'nbins', 'step'])
        table_widget.setHorizontalHeaderLabels(col_headers)
        table_widget.setFixedHeight(table_widget.verticalHeader().defaultSectionSize()*(table_widget.rowCount()+1))  # +1 to include headers
        for icol in range(table_widget.columnCount()):
            table_widget.setColumnWidth(icol, 50)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.table = table_widget
        self.layout.addWidget(self.table)

    def _setup_figure_widget(self):
        fig, _, _, _ = create_subplots(1)
        fig.axes[0].autoscale(enable=True, tight=False)
        self.figure = fig
        self.figure.canvas = FigureCanvas(self.figure)
        toolbar = MantidNavigationToolbar(self.figure.canvas, self)
        self.figure_layout = QVBoxLayout()
        self.figure_layout.addWidget(toolbar)
        self.figure_layout.addWidget(self.figure.canvas)
        self.layout.addLayout(self.figure_layout)

    def _init_slice_table(self):
        for icol in range(self.table.columnCount()):
            for irow in range(self.table.rowCount()):
                item = QTableWidgetItem()
                if icol == 5:
                    item.setData(Qt.EditRole, int(1))
                else:
                    item.setData(Qt.EditRole, float(1))
                if irow == self.table.rowCount()-1:
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)  # disable editing in last row (out of plane dim)
                    item.setBackground(QColor(250, 250, 250))
                else:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.table.setItem(irow, icol, item)

    def _format_cut_figure(self):
        self.figure.axes[0].ignore_existing_data_limits = True
        self.figure.axes[0].autoscale_view()
        self._format_cut_xabel()
        for textobj in self.figure.findobj(text.Text):
            textobj.set_fontsize(8)
        self.figure.tight_layout()

    def _format_cut_xabel(self):
        xlab = self.figure.axes[0].get_xlabel()
        istart = xlab.index('(')
        iend = xlab.index(')')
        xunit_str = xlab[iend + 1:].replace('Ang^-1', '$\\AA^{-1}$)').replace('(', '').replace(')', '')
        xlab = xlab[0:istart] + xlab[istart:iend + 1].replace(' ', ', ') + xunit_str
        self.figure.axes[0].set_xlabel(xlab)


class CutRepresentation:
    def __init__(self, canvas, notify_on_release, xmin, xmax, ymin, ymax, thickness):
        self.notify_on_release = notify_on_release
        self.canvas = canvas
        self.ax = canvas.figure.axes[0]
        self.thickness = thickness
        self.start = self.ax.plot(xmin, ymin, 'ow', label='start')[0]
        self.end = self.ax.plot(xmax, ymax, 'ow', label='end')[0]
        self.line = None
        self.mid = None
        self.box = None
        self.mid_box_top = None
        self.mid_box_bot = None
        self.current_artist = None
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.draw()

    def remove(self):
        self.clear_lines(all_lines=True)
        for cid in [self.cid_release, self.cid_press, self.cid_motion]:
            self.canvas.mpl_disconnect(cid)

    def draw(self):
        self.draw_line()
        self.draw_box()
        self.canvas.draw()

    def get_start_end_points(self):
        xmin, xmax = self.start.get_xdata()[0], self.end.get_xdata()[0]
        ymin, ymax = self.start.get_ydata()[0], self.end.get_ydata()[0]
        return xmin, xmax, ymin, ymax

    def get_mid_point(self):
        xmin, xmax, ymin, ymax = self.get_start_end_points()
        return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

    def get_perp_dir(self):
        # vector perp to line
        xmin, xmax, ymin, ymax = self.get_start_end_points()
        u = np.array([xmax - xmin, ymax - ymin, 0])
        w = np.cross(u, [0, 0, 1])[0:-1]
        what = w / np.sqrt(np.sum(w ** 2))
        return what

    def draw_line(self):
        xmin, xmax, ymin, ymax = self.get_start_end_points()
        self.mid = self.ax.plot(np.mean([xmin, xmax]), np.mean([ymin, ymax]),
                                label='mid', marker='o', color='w', markerfacecolor='w')[0]
        self.line = self.ax.plot([xmin, xmax], [ymin, ymax], '-w')[0]

    def draw_box(self):
        xmin, xmax, ymin, ymax = self.get_start_end_points()
        start = np.array([xmin, ymin])
        end = np.array([xmax, ymax])
        vec = self.get_perp_dir()
        points = np.zeros((5, 2))
        points[0, :] = start + self.thickness * vec / 2
        points[1, :] = start - self.thickness * vec / 2
        points[2, :] = end - self.thickness * vec / 2
        points[3, :] = end + self.thickness * vec / 2
        points[4, :] = points[0, :]  # close the loop
        self.box = self.ax.plot(points[:, 0], points[:, 1], '--r')[0]
        # plot mid points
        mid = 0.5 * (start + end)
        mid_top = mid + self.thickness * vec / 2
        mid_bot = mid - self.thickness * vec / 2
        self.mid_box_top = self.ax.plot(mid_top[0], mid_top[1], 'or', label='mid_box_top',
                                        markerfacecolor='w')[0]
        self.mid_box_bot = self.ax.plot(mid_bot[0], mid_bot[1], 'or', label='mid_box_bot',
                                        markerfacecolor='w')[0]

    def clear_lines(self, all_lines=False):
        lines_to_clear = [self.mid, self.line, self.box, self.mid_box_top, self.mid_box_bot]
        if all_lines:
            lines_to_clear.extend([self.start, self.end])  # normally don't delete these as artist data kept updated
        for line in lines_to_clear:
            if line in self.ax.lines:
                self.ax.lines.remove(line)

    def on_press(self, event):
        if event.inaxes == self.ax and self.current_artist is None:
            x, y = event.xdata, event.ydata
            dx = np.diff(self.ax.get_xlim())[0]
            dy = np.diff(self.ax.get_ylim())[0]
            for line in [self.start, self.end, self.mid, self.mid_box_top, self.mid_box_bot]:
                if abs(x - line.get_xdata()[0]) < dx / 100 and abs(y - line.get_ydata()[0]) < dy / 100:
                    self.current_artist = line
                    break

    def on_motion(self, event):
        if event.inaxes == self.ax and self.current_artist is not None:
            self.clear_lines()
            if len(self.current_artist.get_xdata()) == 1:
                if 'mid' in self.current_artist.get_label():
                    x0, y0 = self.get_mid_point()
                    dx = event.xdata - x0
                    dy = event.ydata - y0
                    if self.current_artist.get_label() == 'mid':
                        for line in [self.start, self.end]:
                            line.set_data([line.get_xdata()[0] + dx], [line.get_ydata()[0] + dy])
                    else:
                        vec = self.get_perp_dir()
                        self.thickness = 2 * abs(np.dot(vec, [dx, dy]))
                else:
                    self.current_artist.set_data([event.xdata], [event.ydata])
            self.draw()  # should draw artists rather than remove and re-plot

    def on_release(self, event):
        if event.inaxes == self.ax and self.current_artist is not None:
            self.current_artist = None
            if self.end.get_xdata()[0] < self.start.get_xdata()[0]:
                self.start, self.end = self.end, self.start
            self.notify_on_release(*self.get_start_end_points(), self.thickness)
