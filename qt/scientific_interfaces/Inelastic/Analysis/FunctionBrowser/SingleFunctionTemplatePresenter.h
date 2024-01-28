// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "Analysis/IDAFunctionParameterEstimation.h"
#include "Analysis/ParameterEstimation.h"
#include "DllConfig.h"
#include "ITemplatePresenter.h"
#include "SingleFunctionTemplateModel.h"
#include <QMap>
#include <QWidget>

class QtProperty;

namespace MantidQt {
namespace MantidWidgets {
class EditLocalParameterDialog;
}
namespace CustomInterfaces {
namespace IDA {

class SingleFunctionTemplateBrowser;

/**
 * Class FunctionTemplateBrowser implements QtPropertyBrowser to display
 * and set properties that can be used to generate a fit function.
 *
 */
class MANTIDQT_INELASTIC_DLL SingleFunctionTemplatePresenter : public QObject, public ITemplatePresenter {
  Q_OBJECT
public:
  explicit SingleFunctionTemplatePresenter(SingleFunctionTemplateBrowser *view,
                                           std::unique_ptr<SingleFunctionTemplateModel> functionModel);

  void init() override;
  void updateAvailableFunctions(const std::map<std::string, std::string> &functionInitialisationStrings) override;

  void setNumberOfDatasets(int) override;
  int getNumberOfDatasets() const override;
  int getCurrentDataset() override;

  void setFitType(std::string const &name) override;

  void setFunction(std::string const &funStr) override;
  IFunction_sptr getGlobalFunction() const override;
  IFunction_sptr getFunction() const override;

  std::vector<std::string> getGlobalParameters() const override;
  std::vector<std::string> getLocalParameters() const override;
  void setGlobalParameters(std::vector<std::string> const &globals) override;
  void setGlobal(std::string const &parameterName, bool on) override;

  void updateMultiDatasetParameters(const IFunction &fun) override;
  void updateParameters(const IFunction &fun) override;

  void setCurrentDataset(int i) override;
  void setDatasets(const QList<FunctionModelDataset> &datasets) override;

  EstimationDataSelector getEstimationDataSelector() const override;
  void updateParameterEstimationData(DataForParameterEstimationCollection &&data) override;
  void estimateFunctionParameters() override;

  void handleEditLocalParameter(std::string const &parameterName) override;

signals:
  void functionStructureChanged();

private slots:
  void editLocalParameterFinish(int result);
  void viewChangedParameterValue(std::string const &parameterName, double value);

private:
  void setErrorsEnabled(bool enabled);
  QStringList getDatasetNames() const;
  QStringList getDatasetDomainNames() const;
  double getLocalParameterValue(std::string const &parameterName, int i) const;
  bool isLocalParameterFixed(std::string const &parameterName, int i) const;
  std::string getLocalParameterTie(std::string const &parameterName, int i) const;
  std::string getLocalParameterConstraint(std::string const &parameterName, int i) const;
  void setLocalParameterValue(std::string const &parameterName, int i, double value);
  void setLocalParameterFixed(std::string const &parameterName, int i, bool fixed);
  void setLocalParameterTie(std::string const &parameterName, int i, std::string const &tie);
  void updateView();
  SingleFunctionTemplateBrowser *m_view;
  std::unique_ptr<SingleFunctionTemplateModel> m_model;
  EditLocalParameterDialog *m_editLocalParameterDialog;
};

} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt
