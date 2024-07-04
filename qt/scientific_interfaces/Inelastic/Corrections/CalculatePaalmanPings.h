// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "CorrectionsTab.h"
#include "DllConfig.h"
#include "ui_CalculatePaalmanPings.h"

#include "MantidGeometry/Instrument_fwd.h"

#include <boost/optional.hpp>
#include <memory>

namespace MantidQt {
namespace CustomInterfaces {

class MANTIDQT_INELASTIC_DLL CalculatePaalmanPings : public CorrectionsTab {
  Q_OBJECT

public:
  CalculatePaalmanPings(QWidget *parent = nullptr);

private slots:
  void absCorComplete(bool error);
  void postProcessComplete(bool error);
  void getBeamWidthFromWorkspace(const QString &wsName);
  void fillCorrectionDetails(const QString &wsName);
  void validateChemical();
  void saveClicked();
  void runClicked();
  void setSampleDensityOptions(QString const &method);
  void setCanDensityOptions(QString const &method);
  void setSampleDensityUnit(QString const &text);
  void setCanDensityUnit(QString const &text);
  void setSampleDensityValue(QString const &text);
  void setCanDensityValue(QString const &text);
  void changeSampleMaterialOptions(int index);
  void changeCanMaterialOptions(int index);
  void setSampleDensity(double value);
  void setCanDensity(double value);

private:
  void run() override;
  bool validate() override;
  void loadSettings(const QSettings &settings) override;
  void setFileExtensionsByName(bool filter) override;

  bool doValidation(bool silent = false);

  void addShapeSpecificSampleOptions(const Mantid::API::IAlgorithm_sptr &alg, const QString &shape);
  void addShapeSpecificCanOptions(const Mantid::API::IAlgorithm_sptr &alg, const QString &shape);

  void setComboBoxOptions(QComboBox *combobox, std::vector<std::string> const &options);

  std::vector<std::string> getDensityOptions(QString const &method) const;
  std::string getDensityType(std::string const &type) const;
  std::string getNumberDensityUnit(std::string const &type) const;
  QString getDensityUnit(QString const &type) const;
  double getSampleDensityValue(QString const &type) const;
  double getCanDensityValue(QString const &type) const;

  void setRunEnabled(bool enabled);
  void setSaveResultEnabled(bool enabled);
  void setButtonsEnabled(bool enabled);
  void setRunIsRunning(bool running);

  boost::optional<double> getInstrumentParameter(const Mantid::Geometry::Instrument_const_sptr &instrument,
                                                 const std::string &parameterName);

  Ui::CalculatePaalmanPings m_uiForm;

  std::shared_ptr<Densities> m_sampleDensities;
  std::shared_ptr<Densities> m_canDensities;
};

} // namespace CustomInterfaces
} // namespace MantidQt
