#ifndef MANTID_CUSTOMINTERFACES_ALCBASELINEMODELLINGMODELTEST_H_
#define MANTID_CUSTOMINTERFACES_ALCBASELINEMODELLINGMODELTEST_H_

#include <cxxtest/TestSuite.h>

#include "MantidAPI/FrameworkManager.h"
#include "MantidAPI/FunctionFactory.h"
#include "MantidAPI/ITableWorkspace.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/WorkspaceFactory.h"

#include "MantidQtCustomInterfaces/Muon/ALCBaselineModellingModel.h"

#include <QtTest/QSignalSpy>

using namespace Mantid::API;
using namespace MantidQt::CustomInterfaces;

class ALCBaselineModellingModelTest : public CxxTest::TestSuite {
  ALCBaselineModellingModel *m_model;

public:
  // This pair of boilerplate methods prevent the suite being created statically
  // This means the constructor isn't called when running other tests
  static ALCBaselineModellingModelTest *createSuite() {
    return new ALCBaselineModellingModelTest();
  }
  static void destroySuite(ALCBaselineModellingModelTest *suite) {
    delete suite;
  }

  ALCBaselineModellingModelTest() {
    FrameworkManager::Instance(); // To make sure everything is initialized
  }

  void setUp() override { m_model = new ALCBaselineModellingModel(); }

  void tearDown() override { delete m_model; }

  void test_setData() {
    std::vector<double> yTestData = {100, 1, 2, 100, 100, 3, 4, 5, 100};
    std::vector<double> xTestData = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    MatrixWorkspace_sptr data = WorkspaceFactory::Instance().create(
        "Workspace2D", 1, yTestData.size(), yTestData.size());
    data->mutableY(0) = yTestData;
    data->mutableX(0) = xTestData;

    QSignalSpy spy(m_model, SIGNAL(dataChanged()));

    TS_ASSERT_THROWS_NOTHING(m_model->setData(data));

    TS_ASSERT_EQUALS(spy.size(), 1);

    MatrixWorkspace_const_sptr modelData = m_model->data();

    TS_ASSERT_EQUALS(modelData->x(0).rawData(), data->x(0).rawData());
    TS_ASSERT_EQUALS(modelData->y(0).rawData(), data->y(0).rawData());
    TS_ASSERT_EQUALS(modelData->e(0).rawData(), data->e(0).rawData());
  }

  void test_fit() {
    std::vector<double> eTestData = {10.0, 1.0, 1.41, 10.0, 10.0, 1.73, 2.0, 2.5, 10.0};
    std::vector<double> yTestData = {100, 1, 2, 100, 100, 3, 4, 5, 100};
    std::vector<double> xTestData = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    MatrixWorkspace_sptr data = WorkspaceFactory::Instance().create(
        "Workspace2D", 1, yTestData.size(), yTestData.size());
    data->mutableE(0) = eTestData;
    data->mutableY(0) = yTestData;
    data->mutableX(0) = xTestData;

    m_model->setData(data);

    IFunction_const_sptr func = FunctionFactory::Instance().createInitialized(
        "name=FlatBackground,A0=0");

    std::vector<IALCBaselineModellingModel::Section> sections;
    sections.emplace_back(2, 3);
    sections.emplace_back(6, 8);

    // TODO: test that the appropriate signals are thrown
    TS_ASSERT_THROWS_NOTHING(m_model->fit(func, sections));

    IFunction_const_sptr fittedFunc = m_model->fittedFunction();
    TS_ASSERT(fittedFunc);

    if (fittedFunc) {
      TS_ASSERT_EQUALS(fittedFunc->name(), "FlatBackground");
      TS_ASSERT_DELTA(fittedFunc->getParameter("A0"), 2.13979, 1E-5);
      TS_ASSERT_DELTA(fittedFunc->getError(0), 0.66709, 1E-5);
    }

    MatrixWorkspace_const_sptr corrected = m_model->correctedData();
    TS_ASSERT(corrected);

    if (corrected) {
      TS_ASSERT_EQUALS(corrected->getNumberHistograms(), 1);
      TS_ASSERT_EQUALS(corrected->blocksize(), 9);

      TS_ASSERT_DELTA(corrected->y(0).rawData()[0], 97.86021, 1E-5);
      TS_ASSERT_DELTA(corrected->y(0).rawData()[2], -0.13979, 1E-5);
      TS_ASSERT_DELTA(corrected->y(0).rawData()[5], 0.86021, 1E-5);
      TS_ASSERT_DELTA(corrected->y(0).rawData()[8], 97.86021, 1E-5);

      TS_ASSERT_EQUALS(corrected->e(0).rawData(), data->e(0).rawData());
    }

    ITableWorkspace_sptr parameters = m_model->parameterTable();
    TS_ASSERT(parameters);

    if (parameters) {
      // Check table dimensions
      TS_ASSERT_EQUALS(parameters->rowCount(), 2);
      TS_ASSERT_EQUALS(parameters->columnCount(), 3);

      // Check table entries
      TS_ASSERT_EQUALS(parameters->String(0, 0), "A0");
      TS_ASSERT_DELTA(parameters->Double(0, 1), 2.13978, 1E-5);
      TS_ASSERT_DELTA(parameters->Double(0, 2), 0.66709, 1E-5);
      TS_ASSERT_EQUALS(parameters->String(1, 0), "Cost function value");
      TS_ASSERT_DELTA(parameters->Double(1, 1), 0.46627, 1E-5);
      TS_ASSERT_EQUALS(parameters->Double(1, 2), 0);
    }

    TS_ASSERT_EQUALS(m_model->sections(), sections);
  }

  void test_exportWorkspace() {
    TS_ASSERT_THROWS_NOTHING(m_model->exportWorkspace());
  }

  void test_exportTable() {
    TS_ASSERT_THROWS_NOTHING(m_model->exportSections());
  }

  void test_exportModel() { TS_ASSERT_THROWS_NOTHING(m_model->exportModel()); }

  void test_noData() {
    // Set a null shared pointer
    MatrixWorkspace_const_sptr data = MatrixWorkspace_const_sptr();
    m_model->setData(data);

    TS_ASSERT_THROWS_NOTHING(m_model->data());
    TS_ASSERT_THROWS_NOTHING(m_model->correctedData());
  }
};

class ALCBaselineModellingModelTestPerformance : public CxxTest::TestSuite {
// ALCBaselineModellingModelPerformance *m_model;

public:
  // This pair of boilerplate methods prevent the suite being created statically
  // This means the constructor isn't called when running other tests
  static ALCBaselineModellingModelTestPerformance *createSuite() {
    return new ALCBaselineModellingModelTestPerformance();
  }
  static void destroySuite(ALCBaselineModellingModelTestPerformance *suite) {
    delete suite;
  }

  ALCBaselineModellingModelTestPerformance() {
    FrameworkManager::Instance(); // To make sure everything is initialized
  }
  void setUp() override { 
	m_model = new ALCBaselineModellingModel();
	xTestData= {10.0, 1.0, 1.41, 10.0, 10.0, 1.73, 2.0, 2.5, 10.0};
    	yTestData= {100, 1, 2, 100, 100, 3, 4, 5, 100};
    	eTestData= {1, 2, 3, 4, 5, 6, 7, 8, 9};

 }

  void tearDown() override { delete m_model; }



  void test_setData() {

    MatrixWorkspace_sptr data = WorkspaceFactory::Instance().create(
        "Workspace2D", 1, yTestData.size(), yTestData.size());
    data->mutableY(0) = yTestData;
    data->mutableX(0) = xTestData;

    QSignalSpy spy(m_model, SIGNAL(dataChanged()));
    MatrixWorkspace_const_sptr modelData = m_model->data();
  }
  void test_fit() {
    MatrixWorkspace_sptr data = WorkspaceFactory::Instance().create(
        "Workspace2D", 1, yTestData.size(), yTestData.size());
    data->mutableE(0) = eTestData;
    data->mutableY(0) = yTestData;
    data->mutableX(0) = xTestData;

    m_model->setData(data);

    IFunction_const_sptr func = FunctionFactory::Instance().createInitialized(
        "name=FlatBackground,A0=0");

    std::vector<IALCBaselineModellingModel::Section> sections;
    sections.emplace_back(2, 3);
    sections.emplace_back(6, 8);
    m_model->disableUnwantedPoitns(data,sections);

    MatrixWorkspace_const_sptr corrected = m_model->correctedData();
    ITableWorkspace_sptr parameters = m_model->parameterTable();

  }



private:
    ALCBaselineModellingModel *m_model;
    std::vector<double> eTestData;
    std::vector<double> yTestData; 
    std::vector<double> xTestData;
};
#endif /* MANTID_CUSTOMINTERFACES_ALCBASELINEMODELLINGMODELTEST_H_ */
