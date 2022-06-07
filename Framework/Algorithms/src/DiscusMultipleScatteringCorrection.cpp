// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidAlgorithms/DiscusMultipleScatteringCorrection.h"
#include "MantidAPI/Axis.h"
#include "MantidAPI/BinEdgeAxis.h"
#include "MantidAPI/ISpectrum.h"
#include "MantidAPI/InstrumentValidator.h"
#include "MantidAPI/NumericAxis.h"
#include "MantidAPI/Sample.h"
#include "MantidAPI/SpectrumInfo.h"
#include "MantidAPI/WorkspaceFactory.h"
#include "MantidAPI/WorkspaceGroup.h"
#include "MantidAPI/WorkspaceUnitValidator.h"
#include "MantidDataObjects/Workspace2D.h"
#include "MantidDataObjects/WorkspaceCreation.h"
#include "MantidGeometry/Instrument.h"
#include "MantidGeometry/Instrument/DetectorInfo.h"
#include "MantidGeometry/Instrument/ReferenceFrame.h"
#include "MantidKernel/BoundedValidator.h"
#include "MantidKernel/CompositeValidator.h"
#include "MantidKernel/EnabledWhenProperty.h"
#include "MantidKernel/EqualBinsChecker.h"
#include "MantidKernel/ListValidator.h"
#include "MantidKernel/Material.h"
#include "MantidKernel/MersenneTwister.h"
#include "MantidKernel/PhysicalConstants.h"
#include "MantidKernel/VectorHelper.h"

using namespace Mantid::API;
using namespace Mantid::Kernel;
using Mantid::DataObjects::Workspace2D;
namespace PhysicalConstants = Mantid::PhysicalConstants;

namespace {
constexpr int DEFAULT_NPATHS = 1000;
constexpr int DEFAULT_SEED = 123456789;
constexpr int DEFAULT_NSCATTERINGS = 2;
constexpr int DEFAULT_LATITUDINAL_DETS = 5;
constexpr int DEFAULT_LONGITUDINAL_DETS = 10;

/// These local unit conversions are used in preference to the Unit classes because they need to be as fast
/// as possible and the sqrt function is faster than pow(x, 0.5) which is what the Unit::quickConversion uses
/// Energy (meV) to wavevector (angstroms-1)
inline double toWaveVector(double energy) { return sqrt(energy / PhysicalConstants::E_mev_toNeutronWavenumberSq); }

/// wavevector (angstroms-1) to Energy (meV)
inline double fromWaveVector(double wavevector) {
  return PhysicalConstants::E_mev_toNeutronWavenumberSq * wavevector * wavevector;
}

struct EFixedProvider {
  explicit EFixedProvider(const ExperimentInfo &expt) : m_expt(expt), m_emode(expt.getEMode()), m_EFixed(0.0) {
    if (m_emode == DeltaEMode::Direct) {
      m_EFixed = m_expt.getEFixed();
    }
  }
  inline DeltaEMode::Type emode() const { return m_emode; }
  inline double value(const Mantid::detid_t detID) const {
    if (m_emode != DeltaEMode::Indirect)
      return m_EFixed;
    else
      return m_expt.getEFixed(detID);
  }

private:
  const ExperimentInfo &m_expt;
  const DeltaEMode::Type m_emode;
  double m_EFixed;
};
} // namespace

namespace Mantid::Algorithms {

// Register the algorithm into the AlgorithmFactory
DECLARE_ALGORITHM(DiscusMultipleScatteringCorrection)

/**
 * Initialize the algorithm
 */
void DiscusMultipleScatteringCorrection::init() {
  // The input workspace must have an instrument
  auto wsValidator = std::make_shared<InstrumentValidator>();

  declareProperty(
      std::make_unique<WorkspaceProperty<>>("InputWorkspace", "", Direction::Input, wsValidator),
      "The name of the input workspace.  The input workspace must have X units of Momentum (k) for elastic "
      "calculations and units of energy transfer (DeltaE) for inelastic calculations. This is used to "
      "supply the sample details, the detector positions and the x axis range to calculate corrections for");

  declareProperty(std::make_unique<WorkspaceProperty<Workspace>>("StructureFactorWorkspace", "", Direction::Input),
                  "The name of the workspace containing S'(q) or S'(q, w).  For elastic calculations, the input "
                  "workspace must contain a single spectrum and have X units of momentum transfer. A workspace group "
                  "containing one workspace per component can also be supplied if a calculation is being run on a "
                  "workspace with a sample environment specified");
  declareProperty(std::make_unique<WorkspaceProperty<WorkspaceGroup>>("OutputWorkspace", "", Direction::Output),
                  "Name for the WorkspaceGroup that will be created. Each workspace in the "
                  "group contains a calculated weight for a particular number of "
                  "scattering events. The number of scattering events varies from 1 up to "
                  "the number supplied in the NumberOfScatterings parameter. The group "
                  "will also include an additional workspace for a calculation with a "
                  "single scattering event where the absorption post scattering has been "
                  "set to zero");
  auto wsKValidator = std::make_shared<WorkspaceUnitValidator>("Momentum");
  declareProperty(std::make_unique<WorkspaceProperty<>>("ScatteringCrossSection", "", Direction::Input,
                                                        PropertyMode::Optional, wsKValidator),
                  "A workspace containing the scattering cross section as a function of k, :math:`\\sigma_s(k)`. Note "
                  "- this parameter would normally be left empty which results in the tabulated cross section data "
                  "being used instead which implies no wavelength dependence");

  auto positiveInt = std::make_shared<Kernel::BoundedValidator<int>>();
  positiveInt->setLower(1);
  declareProperty("NumberOfSimulationPoints", EMPTY_INT(), positiveInt,
                  "The number of points on the input workspace x axis for which a simulation is attempted");

  declareProperty("NeutronPathsSingle", DEFAULT_NPATHS, positiveInt,
                  "The number of \"neutron\" paths to generate for single scattering");
  declareProperty("NeutronPathsMultiple", DEFAULT_NPATHS, positiveInt,
                  "The number of \"neutron\" paths to generate for multiple scattering");
  declareProperty("SeedValue", DEFAULT_SEED, positiveInt, "Seed the random number generator with this value");
  auto nScatteringsValidator = std::make_shared<Kernel::BoundedValidator<int>>();
  nScatteringsValidator->setLower(1);
  nScatteringsValidator->setUpper(5);
  declareProperty("NumberScatterings", DEFAULT_NSCATTERINGS, nScatteringsValidator, "Number of scatterings");

  auto interpolateOpt = createInterpolateOption();
  declareProperty(interpolateOpt->property(), interpolateOpt->propertyDoc());
  declareProperty("SparseInstrument", false,
                  "Enable simulation on special "
                  "instrument with a sparse grid of "
                  "detectors interpolating the "
                  "results to the real instrument.");
  auto threeOrMore = std::make_shared<Kernel::BoundedValidator<int>>();
  threeOrMore->setLower(3);
  declareProperty("NumberOfDetectorRows", DEFAULT_LATITUDINAL_DETS, threeOrMore,
                  "Number of detector rows in the detector grid of the sparse instrument.");
  setPropertySettings("NumberOfDetectorRows",
                      std::make_unique<EnabledWhenProperty>("SparseInstrument", ePropertyCriterion::IS_NOT_DEFAULT));
  auto twoOrMore = std::make_shared<Kernel::BoundedValidator<int>>();
  twoOrMore->setLower(2);
  declareProperty("NumberOfDetectorColumns", DEFAULT_LONGITUDINAL_DETS, twoOrMore,
                  "Number of detector columns in the detector grid "
                  "of the sparse instrument.");
  setPropertySettings("NumberOfDetectorColumns",
                      std::make_unique<EnabledWhenProperty>("SparseInstrument", ePropertyCriterion::IS_NOT_DEFAULT));
  declareProperty("ImportanceSampling", false,
                  "Enable importance sampling on the Q value chosen on multiple scatters based on Q.S(Q)");
  // Control the number of attempts made to generate a random point in the object
  declareProperty("MaxScatterPtAttempts", 5000, positiveInt,
                  "Maximum number of tries made to generate a scattering point "
                  "within the sample. Objects with holes in them, e.g. a thin "
                  "annulus can cause problems if this number is too low.\n"
                  "If a scattering point cannot be generated by increasing "
                  "this value then there is most likely a problem with "
                  "the sample geometry.");
  declareProperty("SimulateEnergiesIndependently", false,
                  "For inelastic calculation, whether the results for adjacent energy transfer bins are simulated "
                  "separately. Currently applies to Direct geometry only");
}

/**
 * Validate the input properties.
 * @return a map where keys are property names and values the found issues
 */
std::map<std::string, std::string> DiscusMultipleScatteringCorrection::validateInputs() {
  std::map<std::string, std::string> issues;
  MatrixWorkspace_sptr inputWS = getProperty("InputWorkspace");
  if (inputWS == nullptr) {
    // Mainly aimed at groups. Group ws pass the property validation on MatrixWorkspace type if all members are
    // MatrixWorkspaces. We output a WorkspaceGroup for a single input workspace so can't manage input groups
    issues["InputWorkspace"] = "Input workspace must be a matrix workspace";
    return issues;
  }
  Geometry::IComponent_const_sptr sample = inputWS->getInstrument()->getSample();
  if (!sample) {
    issues["InputWorkspace"] = "Input workspace does not have a Sample";
  } else {
    if (inputWS->sample().getMaterial().numberDensity() == 0)
      issues["InputWorkspace"] = "Sample must have a material set up with a non-zero number density";
  }

  std::vector<MatrixWorkspace_sptr> SQWSs;
  Workspace_sptr SQWSBase = getProperty("StructureFactorWorkspace");
  auto SQWSGroup = std::dynamic_pointer_cast<WorkspaceGroup>(SQWSBase);
  if (SQWSGroup) {
    auto groupMembers = SQWSGroup->getAllItems();
    int nEnvComponents = 0;
    if (inputWS->sample().hasEnvironment())
      nEnvComponents = inputWS->sample().getEnvironment().nelements();
    std::set<std::string> materialNames;
    materialNames.insert(inputWS->sample().getMaterial().name());
    for (size_t i = 0; i < nEnvComponents; i++)
      materialNames.insert(inputWS->sample().getEnvironment().getComponent(i).id());

    for (auto &materialName : materialNames) {
      auto wsIt = std::find_if(groupMembers.begin(), groupMembers.end(),
                               [materialName](Workspace_sptr &ws) { return ws->getName() == materialName; });
      if (wsIt == groupMembers.end()) {
        issues["StructureFactorWorkspace"] =
            "No workspace for material  " + materialName + " found in S(Q,w) workspace group";
      } else
        SQWSs.push_back(std::dynamic_pointer_cast<MatrixWorkspace>(*wsIt));
    }
  } else
    SQWSs.push_back(std::dynamic_pointer_cast<MatrixWorkspace>(SQWSBase));

  if (inputWS->getEMode() == Kernel::DeltaEMode::Elastic) {
    if (inputWS->getAxis(0)->unit()->unitID() != "Momentum")
      issues["InputWorkspace"] += "Input workspace must have units of Momentum (k) for elastic instrument\n";
    for (auto &SQWS : SQWSs) {
      if (SQWS->getNumberHistograms() != 1)
        issues["StructureFactorWorkspace"] += "S(Q) workspace must contain a single spectrum for elastic mode\n";

      if (SQWS->getAxis(0)->unit()->unitID() != "MomentumTransfer")
        issues["StructureFactorWorkspace"] += "S(Q) workspace must have units of MomentumTransfer\n";
    }
  } else {
    for (auto &SQWS : SQWSs) {
      if (inputWS->getAxis(0)->unit()->unitID() != "DeltaE")
        issues["InputWorkspace"] = "Input workspace must have units of DeltaE for inelastic instrument\n";
      std::set<std::string> axisUnits;
      axisUnits.insert(SQWS->getAxis(0)->unit()->unitID());
      axisUnits.insert(SQWS->getAxis(1)->unit()->unitID());
      if (axisUnits != std::set<std::string>{"DeltaE", "MomentumTransfer"})
        issues["StructureFactorWorkspace"] +=
            "S(Q, w) workspace must have units of Energy Transfer and MomentumTransfer\n";

      if (SQWS->getAxis(1)->isSpectra())
        issues["StructureFactorWorkspace"] += "S(Q, w) must have a numeric spectrum axis\n";
      std::vector<double> wValues;
      if (SQWS->getAxis(0)->unit()->unitID() == "DeltaE")
        wValues = SQWS->x(0).rawData();
      else {
        auto wAxis = dynamic_cast<NumericAxis *>(SQWS->getAxis(1));
        if (wAxis)
          wValues = wAxis->getValues();
      }
      Kernel::EqualBinsChecker checker(wValues, 1E-7, -1);
      if (!checker.validate().empty())
        issues["StructureFactorWorkspace"] += "S(Q,w) must have equally spaced w values";
    }
  }

  for (auto &SQWS : SQWSs) {
    for (size_t i = 0; i < SQWS->getNumberHistograms(); i++) {
      auto &y = SQWS->y(i);
      if (std::any_of(y.cbegin(), y.cend(), [](const auto yval) { return yval < 0 || std::isnan(yval); }))
        issues["StructureFactorWorkspace"] += "S(Q) workspace must have all y >= 0";
    }
  }

  const int nSimulationPoints = getProperty("NumberOfSimulationPoints");
  if (!isEmpty(nSimulationPoints)) {
    InterpolationOption interpOpt;
    const std::string interpValue = getPropertyValue("Interpolation");
    interpOpt.set(interpValue, false, false);
    const auto nSimPointsIssue = interpOpt.validateInputSize(nSimulationPoints);
    if (!nSimPointsIssue.empty())
      issues["NumberOfSimulationPoints"] = nSimPointsIssue;
  }

  const bool simulateEnergiesIndependently = getProperty("SimulateEnergiesIndependently");
  if (simulateEnergiesIndependently) {
    if (inputWS->getEMode() == Kernel::DeltaEMode::Elastic)
      issues["SimulateEnergiesIndependently"] =
          "SimulateEnergiesIndependently is only applicable to inelastic direct geometry calculations";
    if (inputWS->getEMode() == Kernel::DeltaEMode::Indirect)
      issues["SimulateEnergiesIndependently"] =
          "SimulateEnergiesIndependently is only applicable to inelastic direct geometry calculations. Different "
          "energy transfer bins are always simulated separately for indirect geometry";
  }

  return issues;
}
/**
 * This is a variation on the function MatrixWorkspace::getXMinMax with some additional logic
 * eg if x values are all NaN values it raises an error
 * @param ws Workspace to scan for min and max x values
 * @param xmin In/out parameter for min x value found
 * @param xmax In/out parameter for max x value found
 */
void DiscusMultipleScatteringCorrection::getXMinMax(const Mantid::API::MatrixWorkspace &ws, double &xmin,
                                                    double &xmax) const {
  // set to crazy values to start
  xmin = std::numeric_limits<double>::max();
  xmax = -1.0 * xmin;
  size_t numberOfSpectra = ws.getNumberHistograms();
  const auto &spectrumInfo = ws.spectrumInfo();

  // determine the data range - only return min > 0. Bins with x=0 will be skipped later on
  for (size_t wsIndex = 0; wsIndex < numberOfSpectra; wsIndex++) {
    if (spectrumInfo.hasDetectors(wsIndex) && !spectrumInfo.isMonitor(wsIndex) && !spectrumInfo.isMasked(wsIndex)) {
      const auto &dataX = ws.points(wsIndex);
      const double xfront = dataX.front();
      const double xback = dataX.back();
      if (std::isnormal(xfront) && std::isnormal(xback)) {
        if (xfront < xmin)
          xmin = xfront;
        if (xback > xmax)
          xmax = xback;
      }
    }
  }
  // workspace not partitioned at this point so don't replicate code using m_indexInfo->communicator
  if (xmin > xmax)
    throw std::runtime_error("Unable to determine min and max x values for workspace");
}

void DiscusMultipleScatteringCorrection::prepareStructureFactor() {
  // avoid repeated conversion of bin edges to points inside loop by converting to point data
  for (auto &SQWSMapping : m_SQWSs) {
    auto &SQWS = SQWSMapping.SQ;
    convertWsBothAxesToPoints(SQWS);
    // if S(Q,w) has been supplied ensure Q is along the x axis of each spectrum (so same as S(Q))
    if (SQWS->getAxis(1)->unit()->unitID() == "MomentumTransfer") {
      auto transposeAlgorithm = this->createChildAlgorithm("Transpose");
      transposeAlgorithm->initialize();
      transposeAlgorithm->setProperty("InputWorkspace", SQWS);
      transposeAlgorithm->setProperty("OutputWorkspace", "_");
      transposeAlgorithm->execute();
      SQWS = transposeAlgorithm->getProperty("OutputWorkspace");
    } else if (SQWS->getAxis(1)->isSpectra()) {
      // for elastic set w=0 on the spectrum axis to align code with inelastic
      auto newAxis = std::make_unique<NumericAxis>(std::vector<double>{0.});
      newAxis->setUnit("DeltaE");
      SQWS->replaceAxis(1, std::move(newAxis));
    }
    SQWSMapping.logSQ = SQWSMapping.SQ->clone();
    convertToLogWorkspace(SQWSMapping.logSQ);
  }
}

/**
 * Convert x axis of a workspace to points if it's bin edges. If the spectrum axis is a numeric axis and also bin
 * edges convert that to points as well
 * @param ws The workspace that will potentially be converted
 */
void DiscusMultipleScatteringCorrection::convertWsBothAxesToPoints(MatrixWorkspace_sptr &ws) {
  if (ws->isHistogramData()) {
    if (!m_importanceSampling) {
      auto pointDataAlgorithm = this->createChildAlgorithm("ConvertToPointData");
      pointDataAlgorithm->initialize();
      pointDataAlgorithm->setProperty("InputWorkspace", ws);
      pointDataAlgorithm->setProperty("OutputWorkspace", "_");
      pointDataAlgorithm->execute();
      ws = pointDataAlgorithm->getProperty("OutputWorkspace");
    } else {
      // flat interpolation is later used on S(Q) so convert to points by assigning Y value to LH bin edge
      MatrixWorkspace_sptr SQWSPoints =
          API::WorkspaceFactory::Instance().create(ws, ws->getNumberHistograms(), ws->blocksize(), ws->blocksize());
      SQWSPoints->setSharedY(0, ws->sharedY(0));
      SQWSPoints->setSharedE(0, ws->sharedE(0));
      std::vector<double> newX = ws->histogram(0).dataX();
      newX.pop_back();
      SQWSPoints->setSharedX(0, HistogramData::Points(newX).cowData());
      ws = SQWSPoints;
    }
  }
  auto binAxis = dynamic_cast<BinEdgeAxis *>(ws->getAxis(1));
  if (binAxis) {
    auto edges = binAxis->getValues();
    std::vector<double> centres;
    VectorHelper::convertToBinCentre(edges, centres);
    auto newAxis = std::make_unique<NumericAxis>(centres);
    newAxis->setUnit(ws->getAxis(1)->unit()->unitID());
    ws->replaceAxis(1, std::move(newAxis));
  }
}

/**
 * Execution code
 */
void DiscusMultipleScatteringCorrection::exec() {
  g_log.warning(
      "DiscusMultipleScatteringCorrection is in the beta stage of development. Its name, properties and behaviour "
      "may change without warning.");
  if (!getAlwaysStoreInADS())
    throw std::runtime_error("This algorithm explicitly stores named output workspaces in the ADS so must be run with "
                             "AlwaysStoreInADS set to true");
  const MatrixWorkspace_sptr inputWS = getProperty("InputWorkspace");

  Workspace_sptr SQWS = getProperty("StructureFactorWorkspace");
  auto SQWSGroup = std::dynamic_pointer_cast<WorkspaceGroup>(SQWS);
  if (SQWSGroup) {
    auto groupMembers = SQWSGroup->getAllItems();
    for (auto &ws : groupMembers) {
      auto matrixWs = std::dynamic_pointer_cast<MatrixWorkspace>(ws);
      m_SQWSs.push_back(MaterialWorkspaceMapping{matrixWs->getName(), matrixWs});
    }
  } else {
    auto nEnvComponents = 0;
    if (inputWS->sample().hasEnvironment())
      nEnvComponents = inputWS->sample().getEnvironment().nelements();
    m_SQWSs.push_back(MaterialWorkspaceMapping{inputWS->sample().getMaterial().name(),
                                               std::dynamic_pointer_cast<MatrixWorkspace>(SQWS)});
    for (size_t i = 0; i < nEnvComponents; i++) {
      MatrixWorkspace_sptr isotropicSQ = DataObjects::create<Workspace2D>(
          *m_SQWSs[0].SQ, static_cast<size_t>(1),
          HistogramData::Histogram(HistogramData::Points{0.}, HistogramData::Frequencies{1.}));
      std::string_view matName = inputWS->sample().getEnvironment().getComponent(i).material().name();
      g_log.information() << "Creating isotropic structure factor for " << matName << std::endl;
      m_SQWSs.push_back(MaterialWorkspaceMapping{matName, isotropicSQ});
    }
  }

  prepareStructureFactor();

  MatrixWorkspace_sptr sigmaSSWS = getProperty("ScatteringCrossSection");
  if (sigmaSSWS)
    m_sigmaSS = std::make_shared<DataObjects::Histogram1D>(sigmaSSWS->getSpectrum(0));

  // for inelastic we could calculate the qmax based on the min\max w in the S(Q,w) but that
  // would bake as assumption that S(Q,w)=0 beyond the limits of the supplied data
  double qmax = std::numeric_limits<float>::max();
  EFixedProvider efixed(*inputWS);
  m_EMode = efixed.emode();
  g_log.information("EMode=" + DeltaEMode::asString(m_EMode) + " detected");
  if (m_EMode == Kernel::DeltaEMode::Elastic) {
    double kmin, kmax;
    getXMinMax(*inputWS, kmin, kmax);
    qmax = 2 * kmax;
  }
  prepareQSQ(qmax);

  m_simulateEnergiesIndependently = getProperty("SimulateEnergiesIndependently");
  // call this function with dummy efixed to determine total possible simulation points
  const auto inputNbins = generateInputKOutputWList(-1.0, inputWS->points(0).rawData()).size();

  int nSimulationPointsInt = getProperty("NumberOfSimulationPoints");
  size_t nSimulationPoints = static_cast<size_t>(nSimulationPointsInt);

  if (isEmpty(nSimulationPoints)) {
    nSimulationPoints = inputNbins;
  } else if (nSimulationPoints > inputNbins) {
    g_log.warning() << "The requested number of simulation points is larger "
                       "than the maximum number of simulations per spectra. "
                       "Defaulting to "
                    << inputNbins << ".\n ";
    nSimulationPoints = inputNbins;
  }

  const bool useSparseInstrument = getProperty("SparseInstrument");
  SparseWorkspace_sptr sparseWS;
  if (useSparseInstrument) {
    const int latitudinalDets = getProperty("NumberOfDetectorRows");
    const int longitudinalDets = getProperty("NumberOfDetectorColumns");
    sparseWS = createSparseWorkspace(*inputWS, nSimulationPoints, latitudinalDets, longitudinalDets);
  }
  const int nScatters = getProperty("NumberScatterings");
  m_maxScatterPtAttempts = getProperty("MaxScatterPtAttempts");
  std::vector<MatrixWorkspace_sptr> simulationWSs;
  std::vector<MatrixWorkspace_sptr> outputWSs;

  auto noAbsOutputWS = createOutputWorkspace(*inputWS);
  auto noAbsSimulationWS = useSparseInstrument ? sparseWS->clone() : noAbsOutputWS;
  for (int i = 0; i < nScatters; i++) {
    auto outputWS = createOutputWorkspace(*inputWS);
    MatrixWorkspace_sptr simulationWS = useSparseInstrument ? sparseWS->clone() : outputWS;
    simulationWSs.emplace_back(simulationWS);
    outputWSs.emplace_back(outputWS);
  }
  const MatrixWorkspace &instrumentWS = useSparseInstrument ? *sparseWS : *inputWS;

  m_refframe = inputWS->getInstrument()->getReferenceFrame();
  m_sourcePos = inputWS->getInstrument()->getSource()->getPos();
  const auto nhists = useSparseInstrument ? sparseWS->getNumberHistograms() : inputWS->getNumberHistograms();

  m_sampleShape = inputWS->sample().getShapePtr();
  // generate the bounding box before the multithreaded section
  m_sampleShape->getBoundingBox();
  try {
    m_env = &inputWS->sample().getEnvironment();
  } catch (std::runtime_error &) {
    // swallow this as no defined environment from getEnvironment
  }

  const int nSingleScatterEvents = getProperty("NeutronPathsSingle");
  const int nMultiScatterEvents = getProperty("NeutronPathsMultiple");

  const int seed = getProperty("SeedValue");

  InterpolationOption interpolateOpt;
  interpolateOpt.set(getPropertyValue("Interpolation"), false, true);

  m_importanceSampling = getProperty("ImportanceSampling");

  Progress prog(this, 0.0, 1.0, nhists * nSimulationPoints);
  prog.setNotifyStep(0.01);
  const std::string reportMsg = "Computing corrections";

  bool enableParallelFor = true;
  enableParallelFor = std::all_of(simulationWSs.cbegin(), simulationWSs.cend(),
                                  [](const MatrixWorkspace_sptr &ws) { return Kernel::threadSafe(*ws); });

  enableParallelFor = enableParallelFor && Kernel::threadSafe(*noAbsOutputWS);

  const auto &spectrumInfo = instrumentWS.spectrumInfo();

  PARALLEL_FOR_IF(enableParallelFor)
  for (int64_t i = 0; i < static_cast<int64_t>(nhists); ++i) { // signed int for openMP loop
    PARALLEL_START_INTERRUPT_REGION
    auto &spectrum = instrumentWS.getSpectrum(i);
    Mantid::specnum_t specNo = spectrum.getSpectrumNo();
    MersenneTwister rng(seed + specNo);
    // no two theta for monitors

    if (spectrumInfo.hasDetectors(i) && !spectrumInfo.isMonitor(i) && !spectrumInfo.isMasked(i)) {

      const double eFixedValue = efixed.value(spectrumInfo.detector(i).getID());
      auto xPoints = instrumentWS.points(i).rawData();

      auto kInW = generateInputKOutputWList(eFixedValue, xPoints);

      const auto nbins = kInW.size();
      // step size = index range / number of steps requested
      const size_t nsteps = std::max(static_cast<size_t>(1), nSimulationPoints - 1);
      const size_t xStepSize = nbins == 1 ? 1 : (nbins - 1) / nsteps;

      const auto detPos = spectrumInfo.position(i);

      // create copy of the SQ workspaces vector
      auto materialWorkspaces = m_SQWSs;

      if (m_importanceSampling)
        // prep invPOfQ outside the bin loop to avoid costly construction\destruction
        createInvPOfQWorkspaces(materialWorkspaces, 2);

      for (size_t bin = 0; bin < nbins; bin += xStepSize) {
        const double kinc = std::get<0>(kInW[bin]);
        if (kinc <= 0) {
          g_log.warning("Skipping calculation for bin with x<=0, workspace index=" + std::to_string(i) +
                        " bin index=" + std::to_string(std::get<1>(kInW[bin])));
          continue;
        }
        std::vector<double> wValues = std::get<1>(kInW[bin]) == -1 ? xPoints : std::vector{std::get<2>(kInW[bin])};

        if (m_importanceSampling)
          prepareCumulativeProbForQ(kinc, materialWorkspaces);
        calculateQSQIntegralAsFunctionOfK(materialWorkspaces, kinc);

        auto weights = simulatePaths(nSingleScatterEvents, 1, rng, materialWorkspaces, kinc, wValues, detPos, true);
        if (std::get<1>(kInW[bin]) == -1) {
          noAbsSimulationWS->getSpectrum(i).mutableY() += weights;
        } else {
          noAbsSimulationWS->getSpectrum(i).dataY()[std::get<1>(kInW[bin])] = weights[0];
        }

        for (int ne = 0; ne < nScatters; ne++) {
          int nEvents = ne == 0 ? nSingleScatterEvents : nMultiScatterEvents;

          weights = simulatePaths(nEvents, ne + 1, rng, materialWorkspaces, kinc, wValues, detPos, false);
          if (std::get<1>(kInW[bin]) == -1.0) {
            simulationWSs[ne]->getSpectrum(i).mutableY() += weights;
          } else {
            simulationWSs[ne]->getSpectrum(i).dataY()[std::get<1>(kInW[bin])] = weights[0];
          }
        }

        prog.report(reportMsg);

        // Ensure we have the last point for the interpolation
        if (xStepSize > 1 && bin + xStepSize >= nbins && bin + 1 != nbins) {
          bin = nbins - xStepSize - 1;
        }
      } // bins

      // interpolate through points not simulated. Simulation WS only has
      // reduced X values if using sparse instrument so no interpolation
      // required
      if (!useSparseInstrument && xStepSize > 1) {
        auto histNoAbs = noAbsSimulationWS->histogram(i);
        if (xStepSize < nbins) {
          interpolateOpt.applyInplace(histNoAbs, xStepSize);
        } else {
          std::fill(histNoAbs.mutableY().begin() + 1, histNoAbs.mutableY().end(), histNoAbs.y()[0]);
        }
        noAbsOutputWS->setHistogram(i, histNoAbs);

        for (size_t ne = 0; ne < static_cast<size_t>(nScatters); ne++) {
          auto histnew = simulationWSs[ne]->histogram(i);
          if (xStepSize < nbins) {
            interpolateOpt.applyInplace(histnew, xStepSize);
          } else {
            std::fill(histnew.mutableY().begin() + 1, histnew.mutableY().end(), histnew.y()[0]);
          }
          outputWSs[ne]->setHistogram(i, histnew);
        }
      }
    }

    PARALLEL_END_INTERRUPT_REGION
  }
  PARALLEL_CHECK_INTERRUPT_REGION

  if (useSparseInstrument) {
    Poco::Thread::sleep(200); // to ensure prog message changes
    const std::string reportMsgSpatialInterpolation = "Spatial Interpolation";
    prog.report(reportMsgSpatialInterpolation);
    interpolateFromSparse(*noAbsOutputWS, *std::dynamic_pointer_cast<SparseWorkspace>(noAbsSimulationWS),
                          interpolateOpt);
    for (size_t ne = 0; ne < static_cast<size_t>(nScatters); ne++) {
      interpolateFromSparse(*outputWSs[ne], *std::dynamic_pointer_cast<SparseWorkspace>(simulationWSs[ne]),
                            interpolateOpt);
    }
  }

  // Create workspace group that holds output workspaces
  auto wsgroup = std::make_shared<WorkspaceGroup>();
  auto outputGroupWSName = getPropertyValue("OutputWorkspace");
  if (AnalysisDataService::Instance().doesExist(outputGroupWSName))
    API::AnalysisDataService::Instance().deepRemoveGroup(outputGroupWSName);

  const std::string wsNamePrefix = outputGroupWSName + "_Scatter_";
  std::string wsName = wsNamePrefix + "1_NoAbs";
  setWorkspaceName(noAbsOutputWS, wsName);
  wsgroup->addWorkspace(noAbsOutputWS);

  for (size_t i = 0; i < outputWSs.size(); i++) {
    wsName = wsNamePrefix + std::to_string(i + 1);
    setWorkspaceName(outputWSs[i], wsName);
    wsgroup->addWorkspace(outputWSs[i]);

    auto integratedWorkspace = integrateWS(outputWSs[i]);
    setWorkspaceName(integratedWorkspace, wsName + "_Integrated");
    wsgroup->addWorkspace(integratedWorkspace);
  }

  if (outputWSs.size() > 1) {
    // create sum of multiple scatter workspaces for use in subtraction method
    auto summedMScatOutput = createOutputWorkspace(*inputWS);
    for (size_t i = 1; i < outputWSs.size(); i++) {
      summedMScatOutput = summedMScatOutput + outputWSs[i];
    }
    wsName = wsNamePrefix + "2_" + std::to_string(outputWSs.size()) + "_Summed";
    setWorkspaceName(summedMScatOutput, wsName);
    wsgroup->addWorkspace(summedMScatOutput);
    // create sum of all scattering order workspaces for use in ratio method
    auto summedAllScatOutput = createOutputWorkspace(*inputWS);
    summedAllScatOutput = summedMScatOutput + outputWSs[0];
    wsName = wsNamePrefix + "1_" + std::to_string(outputWSs.size()) + "_Summed";
    setWorkspaceName(summedAllScatOutput, wsName);
    wsgroup->addWorkspace(summedAllScatOutput);
    // create ratio of single to all scatter
    auto ratioOutput = createOutputWorkspace(*inputWS);
    ratioOutput = outputWSs[0] / summedAllScatOutput;
    wsName = outputGroupWSName + "_Ratio_Single_To_All";
    setWorkspaceName(ratioOutput, wsName);
    wsgroup->addWorkspace(ratioOutput);

    // ConvFit method being investigated by Spencer for inelastic currently uses the opposite ratio
    if (m_EMode != DeltaEMode::Elastic) {
      auto invRatioOutput = 1 / ratioOutput;
      auto replaceNans = this->createChildAlgorithm("ReplaceSpecialValues");
      replaceNans->setChild(true);
      replaceNans->initialize();
      replaceNans->setProperty("InputWorkspace", invRatioOutput);
      replaceNans->setProperty("OutputWorkspace", invRatioOutput);
      replaceNans->setProperty("NaNValue", 0.0);
      replaceNans->setProperty("InfinityValue", 0.0);
      replaceNans->execute();
      wsName = outputGroupWSName + "_Ratio_All_To_Single";
      setWorkspaceName(invRatioOutput, wsName);
      wsgroup->addWorkspace(invRatioOutput);
    }
  }

  // set the output property
  setProperty("OutputWorkspace", wsgroup);

  if (g_log.is(Kernel::Logger::Priority::PRIO_INFORMATION)) {
    for (auto &kv : m_attemptsToGenerateInitialTrack)
      g_log.information() << "Generating initial track required " + std::to_string(kv.first) + " attempts on " +
                                 std::to_string(kv.second) + " occasions.\n";
    g_log.information() << "Calls to interceptSurface= " + std::to_string(m_callsToInterceptSurface) + "\n";
  }
}

/**
 * Generate a list of the k and w points where calculation results are required. The w points are expressed
 * as bin indices and values.
 * The special bin index value -1 means calculate results for all w bins in the innermost calculation loop using a
 * single set of simulated tracks
 * @param efixed The fixed energy (or zero if an elastic calculation)
 * @param xPoints The x points either in momentum (elastic) or energy transfer (inelastic)
 */
std::vector<std::tuple<double, int, double>>
DiscusMultipleScatteringCorrection::generateInputKOutputWList(const double efixed, const std::vector<double> &xPoints) {
  std::vector<std::tuple<double, int, double>> kInW;
  const double kFixed = toWaveVector(efixed);
  if (m_EMode == DeltaEMode::Elastic) {
    int index = 0;
    std::transform(xPoints.begin(), xPoints.end(), std::back_inserter(kInW), [&index](double d) {
      auto t = std::make_tuple(d, index, 0.);
      index++;
      return t;
    });
  } else {
    if ((!m_simulateEnergiesIndependently) && (m_EMode == DeltaEMode::Direct))
      kInW.emplace_back(std::make_tuple(kFixed, -1, 0.));
    else {
      for (int i = 0; i < static_cast<int>(xPoints.size()); i++) {
        if (m_EMode == DeltaEMode::Direct)
          kInW.emplace_back(std::make_tuple(kFixed, i, xPoints[i]));
        else if (m_EMode == DeltaEMode::Indirect) {
          const double initialE = efixed + xPoints[i];
          if (initialE > 0) {
            const double kin = toWaveVector(initialE);
            kInW.emplace_back(std::make_tuple(kin, i, xPoints[i]));
          } else
            // negative kinc is filtered out later
            kInW.emplace_back(std::make_tuple(-1.0, i, xPoints[i]));
        }
      }
    }
  }
  return kInW;
}

/**
 * Prepare a profile of Q*S(Q) that will later be used to calculate a cumulative probability distribution
 * for use in importance sampling
 * @param qmax The maxmimum q value required based on the data in the InputWorkspace
 * @return A pointer to a histogram containing the Q*S(Q) profile
 */
void DiscusMultipleScatteringCorrection::prepareQSQ(double qmax) {
  for (auto &SQWSMapping : m_SQWSs) {
    auto &SQWS = SQWSMapping.SQ;
    MatrixWorkspace_sptr outputWS = DataObjects::create<Workspace2D>(*SQWS);
    std::vector<double> IOfQYFull;
    // loop through the S(Q) spectra for the different energy transfer values
    for (size_t iW = 0; iW < SQWS->getNumberHistograms(); iW++) {
      std::vector<double> qValues = SQWS->histogram(iW).readX();
      std::vector<double> SQValues = SQWS->histogram(iW).readY();
      // add terminating points at 0 and qmax before multiplying by Q so no extrapolation problems
      if (qValues.front() > 0.) {
        qValues.insert(qValues.begin(), 0.);
        SQValues.insert(SQValues.begin(), SQValues.front());
      }
      if (qValues.back() < qmax) {
        if (m_EMode != DeltaEMode::Elastic) {
          // in case qmax is DBL_MAX add a few extra points beyond supplied q range that are closer. Useful for
          // m_QSQIntegral where flat interpolation means final point at qmax probably won't be used
          double maxSuppliedQ = qValues.back();
          if (2 * maxSuppliedQ < qmax) {
            qValues.push_back(2 * maxSuppliedQ);
            SQValues.push_back(SQValues.back());
          }
          if (4 * maxSuppliedQ < qmax) {
            qValues.push_back(4 * maxSuppliedQ);
            SQValues.push_back(SQValues.back());
          }
        }
        qValues.push_back(qmax);
        SQValues.push_back(SQValues.back());
      }
      // add some extra points to help the Q.S(Q) integral get the right answer
      for (size_t i = 1; i < qValues.size(); i++) {
        if (std::abs(SQValues[i] - SQValues[i - 1]) >
            std::numeric_limits<double>::epsilon() * std::min(SQValues[i - 1], SQValues[i])) {
          qValues.insert(qValues.begin() + i, std::nextafter(qValues[i], -DBL_MAX));
          SQValues.insert(SQValues.begin() + i, SQValues[i - 1]);
          i++;
        }
      }

      std::vector<double> QSQValues;
      std::transform(SQValues.begin(), SQValues.end(), qValues.begin(), std::back_inserter(QSQValues),
                     std::multiplies<double>());

      outputWS->dataX(iW).resize(qValues.size());
      outputWS->dataX(iW) = qValues;
      outputWS->dataY(iW).resize(QSQValues.size());
      outputWS->dataY(iW) = QSQValues;
    }
    SQWSMapping.QSQ = outputWS;
  }
}

/**
 * Integrate QSQ over Q and w over the kinematic range accessible for a given kinc
 * @param kinc The incident wavenumber
 * @param QSQ A workspace containing Q.S(Q,w) with each spectra S(Q) at a particular w
 * @return a tuple containing a cumulative integral as a function of a pseudo variable based on the q values
 * for each w concatenated into a single 1D sequence, the q values corresponding to each value of the pseudo
 * variable, the w values correspodning to each value of the pseudo variable
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
DiscusMultipleScatteringCorrection::integrateQSQ(const API::MatrixWorkspace_sptr &QSQ, double kinc) {
  std::vector<double> IOfQYFull, qValuesFull, wIndices;
  double IOfQMaxPreviousRow = 0.;

  auto wAxis = dynamic_cast<NumericAxis *>(QSQ->getAxis(1));
  if (!wAxis)
    throw std::invalid_argument("Cannot calculate cumulative probability for S(Q,w) without a numeric w axis");
  auto &wValues = wAxis->getValues();
  std::vector<double> wBinEdges;
  wBinEdges.reserve(wValues.size() + 1);
  VectorHelper::convertToBinBoundary(wValues, wBinEdges);

  double wMax = fromWaveVector(kinc);
  auto it = std::lower_bound(wValues.begin(), wValues.end(), wMax);
  size_t iFirstInaccessibleW = std::distance(wValues.begin(), it);
  auto nAccessibleWPoints = iFirstInaccessibleW;
  // shrink bin edges list down to accessible bins only
  wBinEdges.resize(nAccessibleWPoints + 1);

  // loop through the S(Q) spectra for the different energy transfer values
  for (size_t iW = 0; iW < nAccessibleWPoints; iW++) {
    auto kf = getKf(wValues[iW], kinc);
    auto [qmin, qrange] = getKinematicRange(kf, kinc);
    std::vector<double> IOfQX, IOfQY;
    integrateCumulative(QSQ->histogram(iW), qmin, qmin + qrange, IOfQX, IOfQY);
    qValuesFull.insert(qValuesFull.end(), IOfQX.begin(), IOfQX.end());
    wIndices.insert(wIndices.end(), IOfQX.size(), static_cast<double>(iW));
    // w bin width for elastic will equal 1
    double wBinWidth = wBinEdges[iW + 1] - wBinEdges[iW];
    std::transform(IOfQY.begin(), IOfQY.end(), IOfQY.begin(),
                   [IOfQMaxPreviousRow, wBinWidth](double d) -> double { return d * wBinWidth + IOfQMaxPreviousRow; });
    IOfQMaxPreviousRow = IOfQY.back();
    IOfQYFull.insert(IOfQYFull.end(), IOfQY.begin(), IOfQY.end());
  }
  return {IOfQYFull, qValuesFull, wIndices};
}

/**
 * Calculate a cumulative probability distribution for use in importance sampling. The distribution
 * is the inverse function P^-1(t4) where P(Q) = I(Q)/I(2k) and I(x) = integral of Q.S(Q)dQ between 0 and x
 * @param kinc The incident wavenumber
 * @param materialWorkspaces List of workspaces for each material. Of particular interest here is the inverted
 * cumulative probability distribution which is updated by this method
 */
void DiscusMultipleScatteringCorrection::prepareCumulativeProbForQ(
    double kinc, const MaterialWorkspaceMappings &materialWorkspaces) {
  for (size_t i = 0; i < materialWorkspaces.size(); i++) {
    auto QSQ = materialWorkspaces[i].QSQ;
    auto [IOfQYFull, qValuesFull, wIndices] = integrateQSQ(QSQ, kinc);
    auto IOfQYAtQMax = IOfQYFull.empty() ? 0. : IOfQYFull.back();
    if (IOfQYAtQMax == 0.)
      throw std::runtime_error("Integral of Q * S(Q) is zero so can't generate probability distribution");
    // normalise probability range to 0-1
    std::vector<double> IOfQYNorm;
    std::transform(IOfQYFull.begin(), IOfQYFull.end(), std::back_inserter(IOfQYNorm),
                   [IOfQYAtQMax](double d) -> double { return d / IOfQYAtQMax; });
    // Store the normalized integral (= cumulative probability) on the x axis
    // The y values in the two spectra store Q, w (or w index to be precise)
    auto &InvPOfQ = materialWorkspaces[i].InvPOfQ;
    for (size_t i = 0; i < InvPOfQ->getNumberHistograms(); i++) {
      InvPOfQ->dataX(i).resize(IOfQYNorm.size());
      InvPOfQ->dataX(i) = IOfQYNorm;
    }
    InvPOfQ->dataY(0).resize(qValuesFull.size());
    InvPOfQ->dataY(0) = qValuesFull;
    InvPOfQ->dataY(1).resize(wIndices.size());
    InvPOfQ->dataY(1) = wIndices;
  }
}

void DiscusMultipleScatteringCorrection::convertToLogWorkspace(const API::MatrixWorkspace_sptr &SOfQ) {
  // generate log of the structure factor to support gaussian interpolation

  for (size_t i = 0; i < SOfQ->getNumberHistograms(); i++) {
    auto &ySQ = SOfQ->mutableY(i);

    std::transform(ySQ.begin(), ySQ.end(), ySQ.begin(), [](double d) -> double {
      const double exp_that_gives_close_to_zero = -20.0;
      if (d == 0.)
        return exp_that_gives_close_to_zero;
      else
        return std::log(d);
    });
  }
}

/**
 * This is a generalised version of the normalisation done in the original Discus algorithm
 * The original algorithm only considered two scatters so there was only ever one scatter
 * with a free direction after scatter that got a contribution from the q_dir function. This
 * meant that the k value going into the scatter was always fixed and equal to the overall kinc
 * The approach here will cope with multiple scatters by calculating a sumQSS at multiple
 * kinc values. These will be interpolated as required later on
 */
void DiscusMultipleScatteringCorrection::calculateQSQIntegralAsFunctionOfK(MaterialWorkspaceMappings &matWSs,
                                                                           const double specialK) {
  for (auto &SQWSMapping : matWSs) {
    auto &QSQWS = SQWSMapping.QSQ;
    std::vector<double> IOfQYFull;
    std::vector<double> finalkValues, QSQIntegrals;
    std::vector<double> kValues = {specialK};
    // Calculate the integral for a range of k values. Not massively important which k values but choose them here
    // based on the q points in the QS(Q) profile and the initial k values incident on the sample
    const std::vector<double> qValues = QSQWS->histogram(0).readX();
    double lastq = -1;
    for (auto q : qValues) {
      // extra points were added to make QSQ integral more accurate but don't need a separate QSQ integral at these
      if (q > 0 && q > std::nextafter(lastq, DBL_MAX)) {
        double k = q / 2;
        kValues.insert(std::upper_bound(kValues.begin(), kValues.end(), k), k);
      }
      lastq = q;
    }
    for (auto k : kValues) {
      std::tie(IOfQYFull, std::ignore, std::ignore) = integrateQSQ(QSQWS, k);
      auto IOfQYAtQMax = IOfQYFull.empty() ? 0. : IOfQYFull.back();
      // going to divide by this so storing zero results not useful - and don't want to interpolate a zero value
      // into a k region where the integral is actually non-zero
      if (IOfQYAtQMax > 0) {
        double normalisedIntegral = IOfQYAtQMax / (2 * k * k);
        finalkValues.push_back(k);
        QSQIntegrals.push_back(normalisedIntegral);
      }
    }
    auto QSQIntegral = std::make_shared<DataObjects::Histogram1D>(HistogramData::Histogram::XMode::Points,
                                                                  HistogramData::Histogram::YMode::Frequencies);
    QSQIntegral->dataX() = finalkValues;
    QSQIntegral->dataY() = QSQIntegrals;
    SQWSMapping.QSQIntegral = QSQIntegral;
  }
}

/**
 * Integrate a distribution between the supplied xmin and xmax values using trapezoid rule
 * without any extrapolation on either end of the distribution
 * Return the integral for each x value between xmin and xmax. This is the main reason the Integration
 * algorithm isn't used. The Integration algorithm also has some unexpected behaviour for points datasets
 * Return two vectors rather than a histogram for performance reasons and to make transposing it easier
 * @param h Histogram object containing the distribution to integrate
 * @param xmin The lower integration limit
 * @param xmax The upper integration limit
 * @param resultX The x values at which the integral has been calculated
 * @param resultY the values of the integral at various x values up to xmax
 */
void DiscusMultipleScatteringCorrection::integrateCumulative(const Mantid::HistogramData::Histogram &h,
                                                             const double xmin, const double xmax,
                                                             std::vector<double> &resultX,
                                                             std::vector<double> &resultY) {
  const std::vector<double> &xValues = h.readX();
  const std::vector<double> &yValues = h.readY();
  bool isPoints = h.xMode() == HistogramData::Histogram::XMode::Points;

  // set the integral to zero at xmin
  resultX.emplace_back(xmin);
  resultY.emplace_back(0.);
  double sum = 0;

  // ensure there's a point at xmin
  if (xValues.front() > xmin)
    throw std::runtime_error("Distribution doesn't extend as far as lower integration limit, x=" +
                             std::to_string(xmin));
  // ...and a terminating point. Q.S(Q) generally not flat so assuming flat extrapolation not v useful
  if (xValues.back() < xmax)
    throw std::runtime_error("Distribution doesn't extend as far as upper integration limit, x=" +
                             std::to_string(xmax));

  auto iter = std::upper_bound(h.x().cbegin(), h.x().cend(), xmin);
  auto iRight = static_cast<size_t>(std::distance(h.x().cbegin(), iter));

  auto linearInterp = [&xValues, &yValues](const double x, const size_t lIndex, const size_t rIndex) -> double {
    return (yValues[lIndex] * (xValues[rIndex] - x) + yValues[rIndex] * (x - xValues[lIndex])) /
           (xValues[rIndex] - xValues[lIndex]);
  };
  double yToUse;

  // deal with partial initial segments
  if (xmin > xValues[iRight - 1]) {
    if (xmax >= xValues[iRight]) {
      if (isPoints) {
        double interpY = linearInterp(xmin, iRight - 1, iRight);
        yToUse = 0.5 * (interpY + yValues[iRight]);
      } else
        yToUse = yValues[iRight - 1];
      sum += yToUse * (xValues[iRight] - xmin);
      resultX.push_back(xValues[iRight]);
      resultY.push_back(sum);
      iRight++;
    } else {
      if (isPoints) {
        double interpY1 = linearInterp(xmin, iRight - 1, iRight);
        double interpY2 = linearInterp(xmax, iRight - 1, iRight);
        yToUse = 0.5 * (interpY1 + interpY2);
      } else
        yToUse = yValues[iRight - 1];
      sum += yToUse * (xmax - xmin);
      resultX.push_back(xmax);
      resultY.push_back(sum);
      iRight++;
    }
  }

  // integrate the intervals between each pair of points. Do this until right point is at end of vector or > xmax
  for (; iRight < xValues.size() && xValues[iRight] <= xmax; iRight++) {
    yToUse = isPoints ? 0.5 * (yValues[iRight] + yValues[iRight - 1]) : yValues[iRight - 1];
    sum += yToUse * (xValues[iRight] - xValues[iRight - 1]);
    if (xValues[iRight] > std::nextafter(xValues[iRight - 1], DBL_MAX)) {
      resultX.emplace_back(xValues[iRight]);
      resultY.emplace_back(sum);
    }
  }

  // integrate a partial final interval if xmax is between points
  if ((xmax > xValues[iRight - 1]) && (xmin <= xValues[iRight - 1])) {
    if (isPoints) {
      double interpY = linearInterp(xmax, iRight - 1, iRight);
      yToUse = 0.5 * (yValues[iRight - 1] + interpY);
    } else
      yToUse = yValues[iRight - 1];
    sum += yToUse * (xmax - xValues[iRight - 1]);
    resultX.emplace_back(xmax);
    resultY.emplace_back(sum);
  }
}

API::MatrixWorkspace_sptr DiscusMultipleScatteringCorrection::integrateWS(const API::MatrixWorkspace_sptr &ws) {
  auto wsIntegrals = DataObjects::create<Workspace2D>(*ws, HistogramData::Points{0.});
  for (size_t i = 0; i < ws->getNumberHistograms(); i++) {
    std::vector<double> IOfQX, IOfQY;
    integrateCumulative(ws->histogram(i), ws->x(i).front(), ws->x(i).back(), IOfQX, IOfQY);
    wsIntegrals->mutableY(i) = IOfQY.back();
  }
  return wsIntegrals;
}

/**
 * Calculate a total cross section using a k-specific scattering cross section
 * Note - a separate tabulated scattering cross section is used elsewhere in the
 * calculation
 * @param material The sample material
 * @param k The wavenumber where the cross sections are required
 * @param specialSingleScatterCalc Boolean indicating whether special single
 * scatter calculation should be performed
 * @return A tuple containing the total cross section and the scattering cross section
 */
std::tuple<double, double> DiscusMultipleScatteringCorrection::new_vector(const Material &material, double k,
                                                                          bool specialSingleScatterCalc) {
  double scatteringXSection, absorbXsection;
  if (specialSingleScatterCalc) {
    absorbXsection = 0;
  } else {
    const double wavelength = 2 * M_PI / k;
    absorbXsection = material.absorbXSection(wavelength);
  }
  if (m_sigmaSS) {
    scatteringXSection = interpolateFlat(*m_sigmaSS, k);
  } else {
    scatteringXSection = material.totalScatterXSection();
  }

  const auto sig_total = scatteringXSection + absorbXsection;
  return {sig_total, scatteringXSection};
}

/**
 * Use importance sampling to choose a Q and w value for the scatter
 * @param CumulativeProb The inverse of the cumulative probability distribution. Both spectra have x set to 0-1.
 * The first spectrum has y set to Q values and the second spectrum as y set to w index values
 * @param x A randomly chosen value between 0 and 1
 * @return A tuple containing the sampled Q value and the index of the sampled w value in the S(Q,w) distribution
 */
std::tuple<double, int> DiscusMultipleScatteringCorrection::sampleQW(const MatrixWorkspace_sptr &CumulativeProb,
                                                                     double x) {
  return {interpolateSquareRoot(CumulativeProb->getSpectrum(0), x),
          static_cast<int>(interpolateFlat(CumulativeProb->getSpectrum(1), x))};
}

/**
 * Interpolate function of the form y = a * sqrt(x - b) ie inverse of a quadratic
 * Used to lookup value in the cumulative probability distribution of Q S(Q) which
 * for flat S(Q) will be a quadratic
 */
double DiscusMultipleScatteringCorrection::interpolateSquareRoot(const ISpectrum &histToInterpolate, double x) {
  const auto &histx = histToInterpolate.x();
  const auto &histy = histToInterpolate.y();
  assert(histToInterpolate.histogram().xMode() == HistogramData::Histogram::XMode::Points);
  if (x > histx.back()) {
    return histy.back();
  }
  if (x < histx.front()) {
    return histy.front();
  }
  const auto iter = std::upper_bound(histx.cbegin(), histx.cend(), x);
  const auto idx = static_cast<size_t>(std::distance(histx.cbegin(), iter) - 1);
  const double x0 = histx[idx];
  const double x1 = histx[idx + 1];
  const double asq = (pow(histy[idx + 1], 2) - pow(histy[idx], 2)) / (x1 - x0);
  if (asq == 0.) {
    throw std::runtime_error("Cannot perform square root interpolation on supplied distribution");
  }
  const double b = x0 - pow(histy[idx], 2) / asq;
  return sqrt(asq * (x - b));
}

/**
 * Interpolate function using flat interpolation from previous point
 */
double DiscusMultipleScatteringCorrection::interpolateFlat(const ISpectrum &histToInterpolate, double x) {
  auto &xHisto = histToInterpolate.x();
  auto &yHisto = histToInterpolate.y();
  if (x > xHisto.back()) {
    return yHisto.back();
  }
  if (x < xHisto.front()) {
    return yHisto.front();
  }
  auto iter = std::upper_bound(xHisto.cbegin(), xHisto.cend(), x);
  auto idx = static_cast<size_t>(std::distance(xHisto.cbegin(), iter) - 1);
  return yHisto[idx];
}

/**
 * Interpolate a value from a spectrum containing Gaussian peaks. The log of the spectrum has previously
  been taken so this method does a quadratic interpolation and returns e^y
 * @param histToInterpolate The histogram containing the data to interpolate
 * @param x The x value to interpolate at
 * @return The exponential of the interpolated value
 */
double DiscusMultipleScatteringCorrection::interpolateGaussian(const ISpectrum &histToInterpolate, double x) {
  // could have written using points() method so it also worked on histogram data but found that the points
  // method was bottleneck on multithreaded code due to cow_ptr atomic_load
  assert(histToInterpolate.histogram().xMode() == HistogramData::Histogram::XMode::Points);
  if (x > histToInterpolate.x().back()) {
    return exp(histToInterpolate.y().back());
  }
  if (x < histToInterpolate.x().front()) {
    return exp(histToInterpolate.y().front());
  }
  // assume log(cross section) is quadratic in k
  auto deltax = histToInterpolate.x()[1] - histToInterpolate.x()[0];

  auto iter = std::upper_bound(histToInterpolate.x().cbegin(), histToInterpolate.x().cend(), x);
  auto idx = static_cast<size_t>(std::distance(histToInterpolate.x().cbegin(), iter) - 1);

  // need at least two points to the right of the x value for the quadratic
  // interpolation to work
  auto ny = histToInterpolate.y().size();
  if (ny < 3) {
    throw std::runtime_error("Need at least 3 y values to perform quadratic interpolation");
  }
  if (idx > ny - 3) {
    idx = ny - 3;
  }
  // this interpolation assumes the set of 3 bins\point have the same width
  // U=0 on point or bin edge to the left of where x lies
  const auto U = (x - histToInterpolate.x()[idx]) / deltax;
  const auto &y = histToInterpolate.y();
  const auto A = (y[idx] - 2 * y[idx + 1] + y[idx + 2]) / 2;
  const auto B = (-3 * y[idx] + 4 * y[idx + 1] - y[idx + 2]) / 2;
  const auto C = y[idx];
  return exp(A * U * U + B * U + C);
}

/**
 * Interpolate value on S(Q,w) surface given a Q and w. For now there is no interpolation between
 * w values so the nearest one is taken. Also S(Q,w) is assumed to be zero for w beyond the w limits
 * of the supplied surface. S(Q,w) is assumed to equal the extreme value for q beyond the q limits
 * @param SOfQ A workspace containing the structure factor to interpolate
 * @param w The energy transfer (w) value to interpolate at
 * @param q The momentum transfer (q) value to interpolate at
 * @return The interpolated S(Q,w) value
 */
double DiscusMultipleScatteringCorrection::Interpolate2D(const MaterialWorkspaceMapping &SQWSMapping, double w,
                                                         double q) {
  double SQ = 0.;
  int iW = -1;
  auto wAxis = dynamic_cast<NumericAxis *>(SQWSMapping.SQ->getAxis(1));
  if (!wAxis)
    throw std::invalid_argument("Cannot perform 2D interpolation on S(Q,w) that doesn't have a numeric w axis");
  auto &wValues = wAxis->getValues();
  try {
    // required w values will often equal the points in the S(Q,w) distribution so pick nearest value
    iW = static_cast<int>(Kernel::VectorHelper::indexOfValueFromCentersNoThrow(wValues, w));
  } catch (std::out_of_range &) {
  }
  if (iW >= 0) {
    if (m_importanceSampling)
      // the square root interpolation used to look up Q, w in InvPOfQ is based on flat interpolation of S(Q) so use
      // same interpolation here for consistency
      SQ = interpolateFlat(SQWSMapping.SQ->getSpectrum(iW), q);
    else
      SQ = interpolateGaussian(SQWSMapping.logSQ->getSpectrum(iW) /* interpolateFlat(SQWSMapping.SQ->getSpectrum(iW)*/,
                               q);
  }

  return SQ;
}

/**
 * Simulates a set of neutron paths through the sample to a specific detector
 * position with each path containing the specified number of scattering events.
 * Each path represents a group of neutrons and the proportion of neutrons
 * making it to the destination without being scattered or absorbed is
 * calculated as a weight using the cross section information from the sample
 * material. The average weight across all the simulated paths is returned
 * @param nPaths The number of paths to simulate
 * @param nScatters The number of scattering events to simulate along each path
 * @param rng Random number generator
 * @param invPOfQ Inverse of the cumulative prob distribution of Q (used in importance sampling)
 * @param kinc The incident wavevector
 * @param wValues A vector of overall energy transfers
 * @param detPos The position of the detector we're currently calculating a correction for
 * @param specialSingleScatterCalc Boolean indicating whether special single
 * @return An average weight across all of the paths
 */
std::vector<double> DiscusMultipleScatteringCorrection::simulatePaths(
    const int nPaths, const int nScatters, Kernel::PseudoRandomNumberGenerator &rng,
    const MaterialWorkspaceMappings &materialWorkspaces, const double kinc, const std::vector<double> &wValues,
    const Kernel::V3D &detPos, bool specialSingleScatterCalc) {
  std::vector<double> sumOfWeights(wValues.size(), 0.);
  int countZeroWeights = 0; // for debugging and analysis of where importance sampling may help

  for (int ie = 0; ie < nPaths; ie++) {
    auto [success, weights] =
        scatter(nScatters, rng, materialWorkspaces, kinc, wValues, detPos, specialSingleScatterCalc);
    if (success) {
      std::transform(weights.begin(), weights.end(), sumOfWeights.begin(), sumOfWeights.begin(), std::plus<double>());
      if (std::all_of(weights.begin(), weights.end(), [](double d) { return d == 0; }))
        countZeroWeights++;
    } else
      ie--;
  }
  for (size_t i = 0; i < wValues.size(); i++) {
    sumOfWeights[i] = sumOfWeights[i] / nPaths;
  }

  return sumOfWeights;
}

/**
 * Simulates a single neutron path through the sample to a specific detector
 * position containing the specified number of scattering events.
 * Each path represents a group of neutrons and the proportion of neutrons
 * making it to the destination without being scattered or absorbed is
 * calculated as a weight using the cross section information from the sample
 * material
 * @param nScatters The number of scattering events to simulate along each path
 * @param rng Random number generator
 * @param materialWorkspaces List of S(Q,w) related workspaces for each material in the sample\environment
 * @param kinc The incident wavevector
 * @param wValues A vector of overall energy transfers
 * @param detPos The detector position xyz coordinates
 * @param specialSingleScatterCalc Boolean indicating whether special single
 * scatter calculation should be performed
 * @return A tuple containing a success/fail boolean and the calculated weights
 * across the n-1 multiple scatters
 */
std::tuple<bool, std::vector<double>> DiscusMultipleScatteringCorrection::scatter(
    const int nScatters, Kernel::PseudoRandomNumberGenerator &rng, const MaterialWorkspaceMappings &materialWorkspaces,
    const double kinc, const std::vector<double> &wValues, const Kernel::V3D &detPos, bool specialSingleScatterCalc) {
  double weight = 1;

  auto track = start_point(rng);
  auto shapeObjectWithScatter = updateWeightAndPosition(track, weight, kinc, rng, specialSingleScatterCalc);
  double scatteringXSection;
  std::tie(std::ignore, scatteringXSection) =
      new_vector(shapeObjectWithScatter->material(), kinc, specialSingleScatterCalc);

  auto currentMaterialWorkspaces = materialWorkspaces;
  double k = kinc;
  for (int iScat = 0; iScat < nScatters - 1; iScat++) {
    if ((k != kinc)) {
      if (m_importanceSampling) {
        auto newMaterialWorkspaces = materialWorkspaces;
        for (auto &SQWSMapping : currentMaterialWorkspaces)
          SQWSMapping.InvPOfQ = SQWSMapping.InvPOfQ->clone();
        prepareCumulativeProbForQ(k, newMaterialWorkspaces);
        currentMaterialWorkspaces = newMaterialWorkspaces;
      }
    }
    auto trackStillAlive = q_dir(track, shapeObjectWithScatter->material().name(), currentMaterialWorkspaces, k,
                                 scatteringXSection, rng, weight);
    if (!trackStillAlive)
      return {true, std::vector<double>(wValues.size(), 0.)};
    int nlinks = m_sampleShape->interceptSurface(track);
    if (m_env) {
      nlinks += m_env->interceptSurfaces(track);
      m_callsToInterceptSurface += m_env->nelements();
    }
    m_callsToInterceptSurface++;
    if (nlinks == 0) {
      return {false, {0.}};
    }
    shapeObjectWithScatter = updateWeightAndPosition(track, weight, k, rng, specialSingleScatterCalc);
    std::tie(std::ignore, scatteringXSection) =
        new_vector(shapeObjectWithScatter->material(), k, specialSingleScatterCalc);
  }

  Kernel::V3D directionToDetector = detPos - track.startPoint();
  Kernel::V3D prevDirection = track.direction();
  directionToDetector.normalize();
  track.reset(track.startPoint(), directionToDetector);
  int nlinks = m_sampleShape->interceptSurface(track);
  m_callsToInterceptSurface++;
  if (m_env) {
    nlinks += m_env->interceptSurfaces(track);
    m_callsToInterceptSurface += m_env->nelements();
  }
  // due to VALID_INTERCEPT_POINT_SHIFT some tracks that skim the surface
  // of a CSGObject sample may not generate valid tracks. Start over again
  // for this event
  if (nlinks == 0) {
    return {false, {0.}};
  }
  std::vector<double> weights;
  auto scatteringXSectionFull = shapeObjectWithScatter->material().totalScatterXSection();
  // Step through required overall energy transfer (w) values and work out what
  // w that means for the final scatter. There will be a single w value for elastic
  // Slightly different approach to original DISCUS code. It stepped through the w values
  // in the supplied S(Q,w) distribution and applied each one to the final scatter. If
  // this resulted in an overall w that equalled one of the required w values it was output.
  // That approach implicitly assumed S(Q,w)=0 where not specified and that no interpolation
  // on w would be needed - this may be what's required but seems possible it might not always be
  for (auto &w : wValues) {
    const double finalE = fromWaveVector(kinc) - w;
    if (finalE > 0) {
      const double kout = toWaveVector(finalE);
      const auto qVector = directionToDetector * kout - prevDirection * k;
      const double q = qVector.norm();
      const double finalW = fromWaveVector(k) - finalE;
      auto SQWSIt =
          std::find_if(m_SQWSs.begin(), m_SQWSs.end(), [shapeObjectWithScatter](const MaterialWorkspaceMapping &SQWS) {
            return SQWS.materialName == shapeObjectWithScatter->material().name();
          });
      double SQ = Interpolate2D(*SQWSIt, finalW, q);
      double AT2 = 1;
      for (auto it = track.cbegin(); it != track.cend(); it++) {
        double sigma_total;
        auto &materialPassingThrough = it->object->material();
        std::tie(sigma_total, std::ignore) = new_vector(materialPassingThrough, kout, specialSingleScatterCalc);
        double numberDensity = materialPassingThrough.numberDensityEffective();
        double vmu = 100 * numberDensity * sigma_total;
        if (specialSingleScatterCalc)
          vmu = 0;
        const double dl = it->distInsideObject;
        AT2 *= exp(-dl * vmu);
      }
      weights.emplace_back(weight * AT2 * SQ * scatteringXSectionFull / (4 * M_PI));
    } else {
      weights.emplace_back(0.);
    }
  }
  return {true, weights};
}

double DiscusMultipleScatteringCorrection::getKf(const double deltaE, const double kinc) {
  double kf;
  if (deltaE == 0.) {
    kf = kinc; // avoid costly sqrt
  } else {
    // slightly concerned that rounding errors moving between k and E may mean we take the sqrt of
    // a negative number in here. deltaE was capped using a threshold calculated using fromWaveVector so
    // hopefully any rounding will affect fromWaveVector(kinc) in same direction
    kf = toWaveVector(fromWaveVector(kinc) - deltaE);
    assert(!std::isnan(kf));
  }
  return kf;
} // namespace Mantid::Algorithms

/**
 * Get the range of q values accessible for a particular kinc and kf. Since the kinc value is known
 * during the simulation this is similar to direct geometry kinematics
 *  w
 *  |  ---
 *  |_/___\_____Q
 *  | \    \
 *  |  \    \
 * @param kf The wavevector after the scatter event
 * @param ki The wavevector before the scatter event
 * @return a tuple containing qmin and the qrange
 */
std::tuple<double, double> DiscusMultipleScatteringCorrection::getKinematicRange(double kf, double ki) {
  const double qmin = abs(kf - ki);
  const double qrange = 2 * std::min(ki, kf);
  return {qmin, qrange};
}

double DiscusMultipleScatteringCorrection::getQSQIntegral(const ISpectrum &QSQIntegral, double k) {
  // the QSQIntegrals were divided by k^2 so in theory they should be ~flat
  return interpolateFlat(QSQIntegral, k) * 2 * k * k;
}

// update track direction and weight
bool DiscusMultipleScatteringCorrection::q_dir(Geometry::Track &track, const std::string &materialName,
                                               const MaterialWorkspaceMappings &materialWorkspaces, double &k,
                                               const double scatteringXSection,
                                               Kernel::PseudoRandomNumberGenerator &rng, double &weight) {
  const double kinc = k;
  double QQ, SQ;
  int iW;
  auto materialWSIt = std::find_if(
      materialWorkspaces.begin(), materialWorkspaces.end(),
      [materialName](const MaterialWorkspaceMapping &materialWS) { return materialWS.materialName == materialName; });
  if (m_importanceSampling) {
    std::tie(QQ, iW) = sampleQW(materialWSIt->InvPOfQ, rng.nextValue());
    k = getKf(materialWSIt->SQ->getAxis(1)->getValue(iW), kinc);
    // S(Q) not strictly needed here but useful to see if the most popular values are indeed being returned
    SQ = interpolateFlat(materialWSIt->SQ->getSpectrum(iW), QQ);
    weight = weight * scatteringXSection;
  } else {
    auto &wValues = dynamic_cast<NumericAxis *>(/*m_logSQ*/ materialWSIt->SQ->getAxis(1))->getValues();
    // in order to keep integration limits constant sample full range of w even if some not kinematically accessible
    // Note - Discus took different approach where it sampled q,w from kinematically accessible range only but it
    // only calculated for double scattering and easier to normalise in that case
    double wRange;
    /*// the energy transfer must always be less than the positive value corresponding to energy going from ki to 0
    // Note - this is still the case for indirect because on a multiple scatter the kf isn't kfixed
    double wMax = fromWaveVector(kinc);
    // find largest w bin centre that is < wmax and then sample w up to the next bin edge
    auto it = std::lower_bound(wValues.begin(), wValues.end(), wMax);
    int iWMax = static_cast<int>(std::distance(wValues.begin(), it) - 1);*/
    if (wValues.size() == 1) {
      iW = 0;
      wRange = 1;
    } else {
      std::vector<double> wBinEdges;
      wBinEdges.reserve(wValues.size() + 1);
      VectorHelper::convertToBinBoundary(wValues, wBinEdges);
      // w bins not necessarily equal so don't just sample w index
      wRange = /*std::min(wMax, wBinEdges[iWMax + 1])*/ wBinEdges.back() - wBinEdges.front();
      double w = wBinEdges.front() + rng.nextValue() * wRange;
      iW = static_cast<int>(Kernel::VectorHelper::indexOfValueFromCentersNoThrow(wValues, w));
    }

    // if w inaccessible return (ie treat as zero weight) rather than retry so that integration stays over full w range
    if (fromWaveVector(kinc) - wValues[iW] <= 0)
      return false;
    k = toWaveVector(fromWaveVector(kinc) - wValues[iW]);
    double maxkf = toWaveVector(fromWaveVector(kinc) - wValues.front());
    double qrange = kinc + maxkf;
    QQ = qrange * rng.nextValue();
    SQ = interpolateFlat /* interpolateGaussian(m_logSQ->getSpectrum(iW)*/ (materialWSIt->SQ->getSpectrum(iW), QQ);
    // integrate over rectangular area of qw space
    weight = weight * scatteringXSection * SQ * QQ * qrange * wRange;
    if (SQ > 0) {
      double integralQSQ = getQSQIntegral(*materialWSIt->QSQIntegral, kinc);
      assert(integralQSQ != 0.);
      weight = weight / integralQSQ;
    } else
      return false;
  }
  // T = 2theta
  const double cosT = (kinc * kinc + k * k - QQ * QQ) / (2 * kinc * k);
  // if q not accessible return rather than retry so that integration stays over rectangular area
  if (std::abs(cosT) > 1.0)
    return false;

  updateTrackDirection(track, cosT, rng.nextValue() * 2 * M_PI);
  return true;
}

/**
 * Update the track's direction following a scatter event given theta and phi angles
 * @param track The track whose direction will be updated
 * @param cosT Cos two theta. two theta is scattering angle
 * @param phi Phi (radians) of after track. Measured in plane perpendicular to initial trajectory
 */
void DiscusMultipleScatteringCorrection::updateTrackDirection(Geometry::Track &track, const double cosT,
                                                              const double phi) {
  const auto B3 = sqrt(1 - cosT * cosT);
  const auto B2 = cosT;
  // possible to do this using the Quat class instead??
  // Quat(const double _deg, const V3D &_axis);
  // Quat(acos(cosT)*180/M_PI,
  // Kernel::V3D(track.direction()[],track.direction()[],0))

  // Rodrigues formula with final term equal to zero
  // v_rot = cosT * v + sinT(k x v)
  // with rotation axis k orthogonal to v
  // Define k by first creating two vectors orthogonal to v:
  // (vy, -vx, 0) by inspection
  // and then (-vz * vx, -vy * vz, vx * vx + vy * vy) as cross product
  // Then define k as combination of these:
  // sin(phi) * (vy, -vx, 0) + cos(phi) * (-vx * vz, -vy * vz, 1 - vz * vz)
  // ...with division by normalisation factor of sqrt(vx * vx + vy * vy)
  // Note: xyz convention here isn't the standard Mantid one. x=beam, z=up
  const auto vy = track.direction()[0];
  const auto vz = track.direction()[1];
  const auto vx = track.direction()[2];
  double UKX, UKY, UKZ;
  if (vz * vz < 1.0) {
    // calculate A2 from vx^2 + vy^2 rather than 1-vz^2 to reduce floating point rounding error when vz close to
    // 1
    auto A2 = sqrt(vx * vx + vy * vy);
    auto UQTZ = cos(phi) * A2;
    auto UQTX = -cos(phi) * vz * vx / A2 + sin(phi) * vy / A2;
    auto UQTY = -cos(phi) * vz * vy / A2 - sin(phi) * vx / A2;
    UKX = B2 * vx + B3 * UQTX;
    UKY = B2 * vy + B3 * UQTY;
    UKZ = B2 * vz + B3 * UQTZ;
  } else {
    // definition of phi in general formula is dependent on v. So may see phi "redefinition" as vx and vy tend
    // to zero and you move from general formula to this special case
    UKX = B3 * cos(phi);
    UKY = B3 * sin(phi);
    UKZ = B2 * vz;
  }
  track.reset(track.startPoint(), Kernel::V3D(UKY, UKZ, UKX));
}

/**
 * Repeatedly attempt to generate an initial track starting at the source and entering the sample at a random
 * point on its front surface. After each attempt check the track has at least one intercept with sample shape
 * (sometimes for tracks very close to the surface this can sometimes be zero due to numerical precision)
 * @param rng Random number generator
 * @return a track intercepting the sample
 */
Geometry::Track DiscusMultipleScatteringCorrection::start_point(Kernel::PseudoRandomNumberGenerator &rng) {
  for (int i = 0; i < m_maxScatterPtAttempts; i++) {
    auto t = generateInitialTrack(rng);
    int nlinks = m_sampleShape->interceptSurface(t);
    m_callsToInterceptSurface++;
    if (m_env) {
      nlinks += m_env->interceptSurfaces(t);
      m_callsToInterceptSurface += m_env->nelements();
    }
    if (nlinks > 0) {
      if (i > 0) {
        if (g_log.is(Kernel::Logger::Priority::PRIO_WARNING)) {
          m_attemptsToGenerateInitialTrack[i + 1]++;
        }
      }
      return t;
    }
  }
  throw std::runtime_error(
      "DiscusMultipleScatteringCorrection::start_point() - Unable to generate entry point into sample after " +
      std::to_string(m_maxScatterPtAttempts) + " attempts. Try increasing MaxScatterPtAttempts");
}

/** update track start point and weight. The weight is based on a change of variables from length to t1
 * as described in Mancinelli paper
 * @param track A track defining the current trajectory
 * @param weight The weight for the current path that is about to be updated
 * @param k The wavevector of the track
 * @param rng Random number generator
 * @param specialSingleScatterCalc Boolean indicating whether special single scatter calculation should be performed
 * @return the shape object for the component where the scatter occurred
 */

const Geometry::IObject *
DiscusMultipleScatteringCorrection::updateWeightAndPosition(Geometry::Track &track, double &weight, const double k,
                                                            Kernel::PseudoRandomNumberGenerator &rng,
                                                            bool specialSingleScatterCalc) {
  // Work out maximum distance to next scatter point dl
  // At the moment this doesn't cope if sample shape is concave eg if track has more than one segment inside the
  // sample with segment outside sample in between
  double totalMuL = 0.;
  auto nlinks = track.count();
  // Ideally want vector of ptrs to materials but Material class doesn't expose the pointer so use IObject ptr
  // Set default size to 5 (same as in LineIntersectVisit.h)
  boost::container::small_vector<std::tuple<const Geometry::IObject *, double, double>, 5> geometryObjects;
  geometryObjects.reserve(nlinks);
  double newWeight = 0.;
  for (auto it = track.cbegin(); it != track.cend(); it++) {
    const double trackSegLength = it->distInsideObject;
    const auto geometryObj = it->object;
    auto [sigma_total, scatteringXSection] = new_vector(geometryObj->material(), k, specialSingleScatterCalc);
    double vmu = 100 * geometryObj->material().numberDensityEffective() * sigma_total;
    double muL = trackSegLength * vmu;
    totalMuL += muL;
    geometryObjects.push_back(std::make_tuple(geometryObj, vmu, muL));
    double b4 = (1.0 - exp(-muL));
    newWeight += b4 / sigma_total;
  }
  weight = weight * newWeight;

  // randomly sample distance travelled across a total muL and work out which component this sits in
  double b4Overall = (1.0 - exp(-totalMuL));
  double muL = -log(1 - rng.nextValue() * b4Overall);
  double vl = 0.;
  double scatterXsection = 0.;
  std::tuple<const Geometry::IObject *, double, double> geometryObject;
  for (size_t i = 0; i < geometryObjects.size(); i++) {
    geometryObject = geometryObjects[i];
    auto muL_i = std::get<2>(geometryObject);
    auto vmu_i = std::get<1>(geometryObject);
    if (muL - muL_i > 0) {
      vl += muL_i / vmu_i;
      muL = muL - muL_i;
    } else {
      vl += muL / vmu_i;
      break;
    }
  }
  // Note - this clears the track intersections but the sample\environment shapes live on
  inc_xyz(track, vl);
  return std::get<0>(geometryObject);
}

/**
 * Generate an initial track starting at the source and entering
 * the sample at a random point on its front surface
 * @param rng Random number generator
 * @return a track
 */
Geometry::Track DiscusMultipleScatteringCorrection::generateInitialTrack(Kernel::PseudoRandomNumberGenerator &rng) {
  auto &sampleBox = m_sampleShape->getBoundingBox();
  // generate random point on front surface of sample bounding box
  // The change of variables from length to t1 means this still samples the points fairly in the integration
  // volume even in shapes like cylinders where the depth varies across xy
  auto sampleBoxWidth = sampleBox.width();
  auto ptx = sampleBox.minPoint()[m_refframe->pointingHorizontal()] +
             rng.nextValue() * sampleBoxWidth[m_refframe->pointingHorizontal()];
  auto pty =
      sampleBox.minPoint()[m_refframe->pointingUp()] + rng.nextValue() * sampleBoxWidth[m_refframe->pointingUp()];

  // perhaps eventually also generate random point on the beam profile?
  auto ptOnBeamProfile = Kernel::V3D();
  ptOnBeamProfile[m_refframe->pointingHorizontal()] = ptx;
  ptOnBeamProfile[m_refframe->pointingUp()] = pty;
  ptOnBeamProfile[m_refframe->pointingAlongBeam()] = m_sourcePos[m_refframe->pointingAlongBeam()];
  auto toSample = Kernel::V3D();
  toSample[m_refframe->pointingAlongBeam()] = 1.;
  return Geometry::Track(ptOnBeamProfile, toSample);
}

/**
 * Update the x, y, z position of the neutron (or dV volume element
 * to integrate over). Save new start point in to the track object
 * supplied along
 * @param track A track defining the current trajectory
 * @param vl A distance to move along the current trajectory
 */
void DiscusMultipleScatteringCorrection::inc_xyz(Geometry::Track &track, double vl) {
  Kernel::V3D position = track.front().entryPoint;
  Kernel::V3D direction = track.direction();
  const auto x = position[0] + vl * direction[0];
  const auto y = position[1] + vl * direction[1];
  const auto z = position[2] + vl * direction[2];
  const auto startPoint = V3D(x, y, z);
  track.clearIntersectionResults();
  track.reset(startPoint, track.direction());
}

/**
 * Factory method to return an instance of the required SparseInstrument class
 * @param modelWS The full workspace that the sparse one will be based on
 * @param nXPoints The number of x points (k or w) to include in the
 * histograms in the sparse workspace
 * @param rows The number of rows of detectors to create
 * @param columns The number of columns of detectors to create
 * @return a pointer to an SparseInstrument object
 */
std::shared_ptr<SparseWorkspace>
DiscusMultipleScatteringCorrection::createSparseWorkspace(const API::MatrixWorkspace &modelWS, const size_t nXPoints,
                                                          const size_t rows, const size_t columns) {
  auto sparseWS = std::make_shared<SparseWorkspace>(modelWS, nXPoints, rows, columns);
  return sparseWS;
}

void DiscusMultipleScatteringCorrection::createInvPOfQWorkspaces(MaterialWorkspaceMappings &matWSs, size_t nhists) {
  for (auto &SQWSMapping : matWSs) {
    auto &QSQ = SQWSMapping.QSQ;
    size_t expectedMaxSize = QSQ->size();
    MatrixWorkspace_sptr ws = DataObjects::create<Workspace2D>(nhists, HistogramData::Points{0.});
    ws->dataX(0).reserve(expectedMaxSize);
    for (int i = 0; i < nhists; i++)
      ws->dataY(i).reserve(expectedMaxSize);
    SQWSMapping.InvPOfQ = ws;
  }
}

MatrixWorkspace_sptr DiscusMultipleScatteringCorrection::createOutputWorkspace(const MatrixWorkspace &inputWS) const {
  MatrixWorkspace_uptr outputWS = DataObjects::create<Workspace2D>(inputWS);
  // The algorithm computes the signal values at bin centres so they should
  // be treated as a distribution
  outputWS->setDistribution(true);
  outputWS->setYUnit("");
  outputWS->setYUnitLabel("Scattered Weight");
  return outputWS;
}

/**
 * Factory method to return an instance of the required InterpolationOption
 * class
 * @return a pointer to an InterpolationOption object
 */
std::unique_ptr<InterpolationOption> DiscusMultipleScatteringCorrection::createInterpolateOption() {
  auto interpolationOpt = std::make_unique<InterpolationOption>();
  return interpolationOpt;
}

void DiscusMultipleScatteringCorrection::interpolateFromSparse(
    MatrixWorkspace &targetWS, const SparseWorkspace &sparseWS,
    const Mantid::Algorithms::InterpolationOption &interpOpt) {
  const auto &spectrumInfo = targetWS.spectrumInfo();
  const auto refFrame = targetWS.getInstrument()->getReferenceFrame();
  PARALLEL_FOR_IF(Kernel::threadSafe(targetWS, sparseWS))
  for (int64_t i = 0; i < static_cast<decltype(i)>(spectrumInfo.size()); ++i) {
    PARALLEL_START_INTERRUPT_REGION
    if (spectrumInfo.hasDetectors(i) && !spectrumInfo.isMonitor(i)) {
      double lat, lon;
      std::tie(lat, lon) = spectrumInfo.geographicalAngles(i);
      const auto spatiallyInterpHisto = sparseWS.bilinearInterpolateFromDetectorGrid(lat, lon);
      if (spatiallyInterpHisto.size() > 1) {
        auto targetHisto = targetWS.histogram(i);
        interpOpt.applyInPlace(spatiallyInterpHisto, targetHisto);
        targetWS.setHistogram(i, targetHisto);
      } else {
        targetWS.mutableY(i) = spatiallyInterpHisto.y().front();
      }
    }
    PARALLEL_END_INTERRUPT_REGION
  }
  PARALLEL_CHECK_INTERRUPT_REGION
}

/**
 * Adjust workspace name in case of clash in the ADS.
 * Was mainly of value when member workspaces didn't have the group name as a prefix but
 * have left this in place in case there is a clash for any reason
 * @param wsName The name to set on the workspace
 */
void DiscusMultipleScatteringCorrection::correctForWorkspaceNameClash(std::string &wsName) {
  bool noClash(false);

  for (int i = 0; !noClash; ++i) {
    std::string wsIndex; // dont use an index if there is no other
                         // workspace
    if (i > 0) {
      wsIndex = "_" + std::to_string(i);
    }

    bool wsExists = AnalysisDataService::Instance().doesExist(wsName + wsIndex);
    if (!wsExists) {
      wsName += wsIndex;
      noClash = true;
    }
  }
}

/**
 * Set the name on a workspace, adjusting for potential clashes in the ADS.
 * Used to set the names on the output workspace group members. N
 * @param ws The ws to set the name on
 * @param wsName The name to set on the workspace
 */
void DiscusMultipleScatteringCorrection::setWorkspaceName(const API::MatrixWorkspace_sptr &ws, std::string wsName) {
  correctForWorkspaceNameClash(wsName);
  API::AnalysisDataService::Instance().addOrReplace(wsName, ws);
}

} // namespace Mantid::Algorithms
