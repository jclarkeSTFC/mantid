# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.
import unittest

from mantidqt.widgets.sliceviewer.models.transform import NonOrthogonalTransform
from numpy.testing import assert_allclose
import numpy as np


class TransformTest(unittest.TestCase):
    def test_nonorthogonal_transform_skews_as_expected(self):
        transform = NonOrthogonalTransform(angle=np.radians(45.0))  # 45deg

        x = np.array([0, 1])
        y = np.array([1, 0])
        xp, yp = transform.tr(x, y)

        assert_allclose(xp, np.array([1.0 / np.sqrt(2.0), 1.0]))
        assert_allclose(yp, np.array([1.0 / np.sqrt(2.0), 0.0]))

    def test_nonorthogonal_transform_round_trip(self):
        transform = NonOrthogonalTransform(angle=np.radians(40.0))
        x, y = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        xp, yp = transform.tr(x, y)
        xpinv, ypinv = transform.inv_tr(xp, yp)

        assert_allclose(x, xpinv)
        assert_allclose(y, ypinv)

    def test_nonorthogonal_origin_unaltered(self):
        transform = NonOrthogonalTransform(angle=np.radians(40.0))
        x, y = 0.0, 0.0
        xp, yp = transform.tr(x, y)

        self.assertEqual(xp, 0)
        self.assertEqual(yp, 0)


if __name__ == "__main__":
    unittest.main()
