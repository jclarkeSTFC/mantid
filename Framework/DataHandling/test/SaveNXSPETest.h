// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cxxtest/TestSuite.h>

#include "MantidAPI/FrameworkManager.h"
#include "MantidAPI/NumericAxis.h"
#include "MantidDataHandling/LoadInstrument.h"
#include "MantidDataHandling/SaveNXSPE.h"
#include "MantidFrameworkTestHelpers/ComponentCreationHelper.h"
#include "MantidFrameworkTestHelpers/WorkspaceCreationHelper.h"
#include "MantidGeometry/Instrument/DetectorInfo.h"
#include "MantidIndexing/IndexInfo.h"
#include "MantidKernel/DeltaEMode.h"
#include "MantidKernel/PropertyWithValue.h"
#include "MantidKernel/UnitFactory.h"

#include "boost/tuple/tuple.hpp"
#include <memory>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <Poco/File.h>

#include <limits>

using namespace Mantid::API;
using namespace Mantid::DataHandling;
using Mantid::Geometry::IDetector_const_sptr;
using Mantid::Geometry::Instrument;
using Mantid::Geometry::ParameterMap;
using Mantid::Kernel::UnitFactory;

static const int THEMASKED = 1;

class SaveNXSPETest : public CxxTest::TestSuite {
public:
  static SaveNXSPETest *createSuite() { return new SaveNXSPETest(); }
  static void destroySuite(SaveNXSPETest *suite) { delete suite; }

  void testName() {
    SaveNXSPE saver;
    TS_ASSERT_EQUALS(saver.name(), "SaveNXSPE");
  }

  void testVersion() {
    SaveNXSPE saver;
    TS_ASSERT_EQUALS(saver.version(), 1);
  }

  void testInit() {
    SaveNXSPE saver;
    TS_ASSERT_THROWS_NOTHING(saver.initialize());
    TS_ASSERT(saver.isInitialized());

    TS_ASSERT_EQUALS(static_cast<int>(saver.getProperties().size()), 6);
  }

  void test_Saving_Workspace_Smaller_Than_Chunk_Size() {
    // Create a small test workspace
    const int nhist(3), nx(10);

    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);
    auto loadedData = saveAndReloadWorkspace(input, 10);
    auto dims = loadedData.get<0>();
    auto signal = loadedData.get<1>();
    auto error = loadedData.get<2>();

    double tolerance(1e-08);
    TS_ASSERT_EQUALS(nhist, dims[0]);
    TS_ASSERT_EQUALS(nx, dims[1]);
    // element 0,0
    TS_ASSERT_DELTA(0.0, signal[0], tolerance);
    TS_ASSERT_DELTA(0.0, error[0], tolerance);
    // element 0,9
    TS_ASSERT_DELTA(9.0, signal[9], tolerance);
    TS_ASSERT_DELTA(18.0, error[9], tolerance);
    // element 1,2 in 2D flat buffer
    TS_ASSERT(std::isnan(signal[1 * dims[1] + 2]));
    TS_ASSERT_DELTA(0.0, error[1 * dims[1] + 2], tolerance);
    // final element
    TS_ASSERT_DELTA(29.0, signal[dims[0] * dims[1] - 1], tolerance);
    TS_ASSERT_DELTA(58.0, error[dims[0] * dims[1] - 1], tolerance);
  }

  void test_Ei_onProperty_superseeds_logs() {
    // Create a small test workspace
    const int nhist(3), nx(10);
    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);
    std::unique_ptr<Mantid::Kernel::PropertyWithValue<std::string>> mode(
        new Mantid::Kernel::PropertyWithValue<std::string>("deltaE-mode", "Direct"));
    // pointer is owned by run, no need to delete
    input->mutableRun().addLogData(mode.release());

    std::unique_ptr<Mantid::Kernel::PropertyWithValue<double>> EiLog(
        new Mantid::Kernel::PropertyWithValue<double>("Ei", 12.0));
    input->mutableRun().addLogData(EiLog.release());

    auto loadedData = saveAndReloadWorkspace(input, 10, true);
    auto dims = loadedData.get<0>();
    auto signal = loadedData.get<1>();
    auto error = loadedData.get<2>();
    auto Ei = loadedData.get<3>();

    double tolerance(1e-08);
    TS_ASSERT_DELTA(10.0, Ei[0], tolerance);
    //
    TS_ASSERT_EQUALS(nhist, dims[0]);
    TS_ASSERT_EQUALS(nx, dims[1]);
    // element 0,0
    TS_ASSERT_DELTA(0.0, signal[0], tolerance);
    TS_ASSERT_DELTA(0.0, error[0], tolerance);
    // element 0,9
    TS_ASSERT_DELTA(9.0, signal[9], tolerance);
    TS_ASSERT_DELTA(18.0, error[9], tolerance);
    // element 1,2 in 2D flat buffer
    TS_ASSERT(std::isnan(signal[1 * dims[1] + 2]));
    TS_ASSERT_DELTA(0.0, error[1 * dims[1] + 2], tolerance);
    // final element
    TS_ASSERT_DELTA(29.0, signal[dims[0] * dims[1] - 1], tolerance);
    TS_ASSERT_DELTA(58.0, error[dims[0] * dims[1] - 1], tolerance);
  }

  void test_Saving_Workspace_Small_EiOnLog_direct() {
    // Create a small test workspace
    const int nhist(3), nx(10);
    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);
    std::unique_ptr<Mantid::Kernel::PropertyWithValue<std::string>> mode(
        new Mantid::Kernel::PropertyWithValue<std::string>("deltaE-mode", "Direct"));
    // pointer is owned by run, no need to delete
    input->mutableRun().addLogData(mode.release());

    std::unique_ptr<Mantid::Kernel::PropertyWithValue<double>> EiLog(
        new Mantid::Kernel::PropertyWithValue<double>("Ei", 12.0));
    input->mutableRun().addLogData(EiLog.release());

    auto loadedData = saveAndReloadWorkspace(input, 10, false);
    auto dims = loadedData.get<0>();
    auto signal = loadedData.get<1>();
    auto error = loadedData.get<2>();
    auto Ei = loadedData.get<3>();

    double tolerance(1e-08);
    TS_ASSERT_DELTA(12.0, Ei[0], tolerance);
    //
    TS_ASSERT_EQUALS(nhist, dims[0]);
    TS_ASSERT_EQUALS(nx, dims[1]);
    // element 0,0
    TS_ASSERT_DELTA(0.0, signal[0], tolerance);
    TS_ASSERT_DELTA(0.0, error[0], tolerance);
    // element 0,9
    TS_ASSERT_DELTA(9.0, signal[9], tolerance);
    TS_ASSERT_DELTA(18.0, error[9], tolerance);
    // element 1,2 in 2D flat buffer
    TS_ASSERT(std::isnan(signal[1 * dims[1] + 2]));
    TS_ASSERT_DELTA(0.0, error[1 * dims[1] + 2], tolerance);
    // final element
    TS_ASSERT_DELTA(29.0, signal[dims[0] * dims[1] - 1], tolerance);
    TS_ASSERT_DELTA(58.0, error[dims[0] * dims[1] - 1], tolerance);
  }

  void test_Saving_Workspace_Small_EiOnLog_indirect() {
    // Create a small test workspace
    const int nhist(3), nx(10);
    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);

    std::vector<double> EiValue(1, 12.0);
    input = makeWS_indirect(input, EiValue);

    auto loadedData = saveAndReloadWorkspace(input, 10, false);
    auto Ei = loadedData.get<3>();

    double tolerance(1e-08);
    TS_ASSERT_DELTA(12.0, Ei[0], tolerance);
  }

  void test_Saving_Workspace_Larger_Than_Chunk_Size() {
    // Create a test workspace
    const int nhist(5250), nx(100);
    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);
    auto loadedData = saveAndReloadWorkspace(input, 10);
    auto dims = loadedData.get<0>();
    auto signal = loadedData.get<1>();
    auto error = loadedData.get<2>();

    double tolerance(1e-08);
    TS_ASSERT_EQUALS(nhist, dims[0]);
    TS_ASSERT_EQUALS(nx, dims[1]);
    // element 0,0
    TS_ASSERT_DELTA(0.0, signal[0], tolerance);
    TS_ASSERT_DELTA(0.0, error[0], tolerance);
    // element 0,9
    TS_ASSERT_DELTA(99.0, signal[99], tolerance);
    TS_ASSERT_DELTA(198.0, error[99], tolerance);
    // element 1,2 in 2D flat buffer
    TS_ASSERT(std::isnan(signal[1 * dims[1] + 2]));
    TS_ASSERT_DELTA(0.0, error[1 * dims[1] + 2], tolerance);
    // final element
    TS_ASSERT_DELTA(524999.0, signal[dims[0] * dims[1] - 1], tolerance);
    TS_ASSERT_DELTA(1049998.0, error[dims[0] * dims[1] - 1], tolerance);
  }

  void testExecWithParFile() {
    MatrixWorkspace_sptr input = makeWorkspace();

    SaveNXSPE saver;
    saver.initialize();
    saver.setChild(true);
    TS_ASSERT_THROWS_NOTHING(saver.setProperty("InputWorkspace", input));
    TS_ASSERT_THROWS_NOTHING(saver.setProperty("ParFile", "testParFile.par"));
    std::string outputFile("SaveNXSPETest_testExecWithParFile.nxspe");
    TS_ASSERT_THROWS_NOTHING(saver.setPropertyValue("Filename", outputFile));
    outputFile = saver.getPropertyValue("Filename"); // get absolute path

    // throws file not exist from ChildAlgorithm
    saver.setRethrows(true);
    TS_ASSERT_THROWS(saver.execute(), const Mantid::Kernel::Exception::FileError &);
    TS_ASSERT(Poco::File(outputFile).exists());

    if (Poco::File(outputFile).exists())
      Poco::File(outputFile).remove();
  }

  void xestSaveNXSPE_extractIndirectDetectorsEn() {
    // TODO: for a future, I do not know how to set energy for a detector
    // and modern Indirect have same EI anyway
    const int nhist(3), nx(10);
    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);

    std::vector<double> EiValues(nhist);
    for (int i = 0; i < nhist; i++)
      EiValues[i] = double(i + 2);

    input = makeWS_indirect(input, EiValues);

    auto loadedData = saveAndReloadWorkspace(input, 10, false);
    auto detEi = loadedData.get<3>();

    for (int i = 0; i < nhist; i++) {
      TS_ASSERT_DELTA(EiValues[i], detEi[i], 1.e-8);
    }
  }
  void xest_saveIndirectDetectorsEn() {
    // TODO: for a future, I do not know how to set energy for a detector
    // and ISIS Indirect instruments have same EI anyway
    const int nhist(3), nx(10);
    MatrixWorkspace_sptr input = makeWorkspace(nhist, nx);

    std::vector<double> EiValues(nhist);
    for (int i = 0; i < nhist; i++)
      EiValues[i] = double(i + 2);

    input = makeWS_indirect(input, EiValues);

    SaveNXSPE saver;

    auto detEi = saver.getIndirectEfixed(input);
    for (int i = 0; i < nhist; i++) {
      TS_ASSERT_EQUALS(EiValues[i], detEi[i]);
    }
  }

private:
  MatrixWorkspace_sptr makeWS_indirect(MatrixWorkspace_sptr inputWS, std::vector<double> &detEi) {
    std::unique_ptr<Mantid::Kernel::PropertyWithValue<std::string>> mode(
        new Mantid::Kernel::PropertyWithValue<std::string>("deltaE-mode", "Indirect"));
    // pointer is owned by run, no need to delete it
    inputWS->mutableRun().addLogData(mode.release());
    if (detEi.size() == 1) {
      std::unique_ptr<Mantid::Kernel::PropertyWithValue<double>> EiLog(
          new Mantid::Kernel::PropertyWithValue<double>("Ei", detEi[0]));
      inputWS->mutableRun().addLogData(EiLog.release());
    } else {
      const auto nHist = static_cast<size_t>(inputWS->getNumberHistograms());
      // auto &specInfo = inputWS->mutableSpectrumInfo(); //TODO: may be used to change detector energy. May be not
      for (size_t i = 0; i < nHist; i++) {
        // TODO:
        // here should be code which sets different energies from detEi to each mutable detector
        // possibly retrieved from spectrumInfo or instrument
        continue;
      }
    }
    return inputWS;
  }

  MatrixWorkspace_sptr makeWorkspace(int nhist = 3, int nx = 10) {
    auto testWS = WorkspaceCreationHelper::create2DWorkspaceBinned(nhist, nx, 1.0);
    // Fill workspace with increasing counter to properly check saving
    for (int i = 0; i < nhist; ++i) {
      auto &outY = testWS->mutableY(i);
      auto &outE = testWS->mutableE(i);
      for (int j = 0; j < nx; ++j) {
        outY[j] = i * nx + j;
        outE[j] = outY[j] * 2.0;
      }
    }
    return setUpWorkspace(testWS);
  }

  MatrixWorkspace_sptr setUpWorkspace(MatrixWorkspace_sptr inputWS) {
    inputWS->getAxis(0)->unit() = Mantid::Kernel::UnitFactory::Instance().create("DeltaE");
    // Create an instrument but we don't care where they are
    std::vector<double> dummy(inputWS->getNumberHistograms(), 0.0);
    auto testInst = ComponentCreationHelper::createCylInstrumentWithDetInGivenPositions(dummy, dummy, dummy);
    inputWS->setInstrument(testInst);
    inputWS->setIndexInfo(Mantid::Indexing::IndexInfo(inputWS->getNumberHistograms()));

    // mask the detector
    inputWS->mutableDetectorInfo().setMasked(THEMASKED, true);

    // required to get it passed the algorthms validator
    inputWS->setDistribution(true);

    return inputWS;
  }

  using DataHolder = boost::tuple<std::vector<hsize_t>, std::vector<double>, std::vector<double>, std::vector<double>>;

  DataHolder saveAndReloadWorkspace(const MatrixWorkspace_sptr &inputWS, const double efix_value,
                                    bool set_efixed = true) {
    SaveNXSPE saver;
    saver.initialize();
    saver.setChild(true);
    TS_ASSERT_THROWS_NOTHING(saver.setProperty("InputWorkspace", inputWS));
    std::string outputFile("SaveNXSPETest_testEXEC.nxspe");
    TS_ASSERT_THROWS_NOTHING(saver.setPropertyValue("Filename", outputFile));
    outputFile = saver.getPropertyValue("Filename"); // get absolute path

    if (set_efixed)
      TS_ASSERT_THROWS_NOTHING(saver.setProperty("Efixed", efix_value));

    TS_ASSERT_THROWS_NOTHING(saver.setProperty("Psi", 0.0));
    TS_ASSERT_THROWS_NOTHING(saver.setProperty("KiOverKfScaling", true));
    TS_ASSERT_THROWS_NOTHING(saver.execute());
    TS_ASSERT(saver.isExecuted());

    TS_ASSERT(Poco::File(outputFile).exists());
    if (!Poco::File(outputFile).exists()) {
      return boost::make_tuple(std::vector<hsize_t>(), std::vector<double>(), std::vector<double>(),
                               std::vector<double>());
    }

    auto h5file = H5Fopen(outputFile.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    const char *dset = "/mantid_workspace/data/data";
    int rank(0);
    herr_t status = H5LTget_dataset_ndims(h5file, dset, &rank);
    TS_ASSERT_EQUALS(0, status);
    TS_ASSERT_EQUALS(2, rank);

    std::vector<hsize_t> dims(rank);
    H5T_class_t classId(H5T_NO_CLASS);
    size_t typeSize(0);
    status = H5LTget_dataset_info(h5file, dset, dims.data(), &classId, &typeSize);
    TS_ASSERT_EQUALS(0, status);
    TS_ASSERT_EQUALS(H5T_FLOAT, classId);
    TS_ASSERT_EQUALS(8, typeSize);

    size_t bufferSize(dims[0] * dims[1]);
    std::vector<double> signal(bufferSize), error(bufferSize);
    status = H5LTread_dataset_double(h5file, dset, signal.data());
    TS_ASSERT_EQUALS(0, status);

    const char *dsetErr = "/mantid_workspace/data/error";
    status = H5LTread_dataset_double(h5file, dsetErr, error.data());
    TS_ASSERT_EQUALS(0, status);
    //---------------------------------------------------------------
    // check efixed
    const char *efixed_dset = "/mantid_workspace/NXSPE_info/fixed_energy";
    status = H5LTget_dataset_ndims(h5file, efixed_dset, &rank);
    TS_ASSERT_EQUALS(0, status);
    TS_ASSERT_EQUALS(1, rank);

    std::vector<hsize_t> efix_dims(rank);
    status = H5LTget_dataset_info(h5file, efixed_dset, efix_dims.data(), &classId, &typeSize);
    TS_ASSERT_EQUALS(0, status);
    TS_ASSERT_EQUALS(H5T_FLOAT, classId);
    TS_ASSERT_EQUALS(8, typeSize);

    size_t EnBuffer(efix_dims[0]);
    std::vector<double> efixed(EnBuffer);
    status = H5LTread_dataset_double(h5file, efixed_dset, efixed.data());
    TS_ASSERT_EQUALS(0, status);
    if (set_efixed) {
      TS_ASSERT_EQUALS(EnBuffer, 1);
      TS_ASSERT_DELTA(efixed[0], efix_value, 1.e-8);
    }

    H5Fclose(h5file);
    // Poco::File(outputFile).remove();

    return boost::make_tuple(dims, signal, error, efixed);
  }
};
