// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "DllOption.h"

#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/SpectrumInfo.h"
#include "MantidQtWidgets/Common/CoordinateConversion.h"
#include "MantidQtWidgets/Common/ImageInfoModel.h"

namespace Mantid {
namespace Geometry {
class IComponent;
class Instrument;
} // namespace Geometry
namespace API {
class SpectrumInfo;
}
} // namespace Mantid

namespace MantidQt {
namespace MantidWidgets {

class EXPORT_OPT_MANTIDQT_COMMON ImageInfoModelMatrixWS
    : public ImageInfoModel {

public:
  ImageInfoModelMatrixWS(const Mantid::API::MatrixWorkspace_sptr &ws,
                         CoordinateConversion &coordConversion);

  // Creates a list containing pairs of strings with information about the
  // coordinates in the workspace.
  std::vector<std::string> getInfoList(const double x, const double y,
                                       const double z) override;

private:
  Mantid::API::MatrixWorkspace_sptr m_workspace;
  const Mantid::API::SpectrumInfo *m_spectrumInfo;
  std::shared_ptr<const Mantid::Geometry::Instrument> m_instrument;
  std::shared_ptr<const Mantid::Geometry::IComponent> m_source;
  std::shared_ptr<const Mantid::Geometry::IComponent> m_sample;
  CoordinateConversion &m_coordConversion;
  double m_xMin;
  double m_xMax;
  double m_yMax;
};

} // namespace MantidWidgets
} // namespace MantidQt
