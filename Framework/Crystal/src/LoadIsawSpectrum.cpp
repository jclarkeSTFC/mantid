// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCrystal/LoadIsawSpectrum.h"
#include "MantidAPI/Axis.h"
#include "MantidAPI/FileProperty.h"
#include "MantidAPI/WorkspaceFactory.h"
#include "MantidDataObjects/Workspace2D.h"
#include "MantidGeometry/Instrument/RectangularDetector.h"
#include "MantidKernel/BoundedValidator.h"
#include "MantidKernel/ListValidator.h"
#include "MantidKernel/OptionalBool.h"
#include "MantidKernel/PhysicalConstants.h"
#include "MantidKernel/Unit.h"
#include "MantidKernel/UnitFactory.h"
#include "MantidKernel/Utils.h"

#include <fstream>

using namespace Mantid::Geometry;
using namespace Mantid::DataObjects;
using namespace Mantid::Kernel;
using namespace Mantid::API;
using namespace Mantid::PhysicalConstants;

namespace Mantid::Crystal {

// Register the algorithm into the AlgorithmFactory
DECLARE_ALGORITHM(LoadIsawSpectrum)

/** Initialize the algorithm's properties.
 */
void LoadIsawSpectrum::init() {
  declareProperty(std::make_unique<FileProperty>("SpectraFile", "", API::FileProperty::Load, ".dat"),
                  "Incident spectrum and detector efficiency correction file.");
  declareProperty(std::make_unique<WorkspaceProperty<MatrixWorkspace>>("OutputWorkspace", "", Direction::Output),
                  "An output Workspace containing spectra for each detector bank.");
  // 3 properties for getting the right instrument
  getInstrument3WaysInit(this);
}

/** Execute the algorithm.
 */
void LoadIsawSpectrum::exec() {
  Instrument_const_sptr inst = getInstrument3Ways(this);

  // If sample not at origin, shift cached positions.
  const V3D samplePos = inst->getSample()->getPos();
  const V3D pos = inst->getSource()->getPos() - samplePos;
  double l1 = pos.norm();

  std::string STRING;
  std::ifstream infile;
  std::string spectraFile = getPropertyValue("SpectraFile");
  infile.open(spectraFile.c_str());

  size_t a = -1;
  std::vector<std::vector<double>> spectra;
  std::vector<std::vector<double>> time;
  int iSpec = 0;
  for (int wi = 0; wi < 8; wi++)
    getline(infile, STRING); // Saves the line in STRING.
  while (!infile.eof())      // To get you all the lines.
  {
    time.resize(a + 1);
    spectra.resize(a + 1);
    getline(infile, STRING); // Saves the line in STRING.
    if (infile.eof())
      break;
    std::stringstream ss(STRING);
    if (STRING.find("Bank") == std::string::npos) {
      double time0, spectra0;
      ss >> time0 >> spectra0;
      time[a].emplace_back(time0);
      spectra[a].emplace_back(spectra0);

    } else {
      a++;
    }
  }
  infile.close();
  // Build a list of Rectangular Detectors
  std::vector<std::shared_ptr<RectangularDetector>> detList;
  for (int i = 0; i < inst->nelements(); i++) {
    std::shared_ptr<RectangularDetector> det;
    std::shared_ptr<ICompAssembly> assem;
    std::shared_ptr<ICompAssembly> assem2;

    det = std::dynamic_pointer_cast<RectangularDetector>((*inst)[i]);
    if (det) {
      detList.emplace_back(det);
    } else {
      // Also, look in the first sub-level for RectangularDetectors (e.g. PG3).
      // We are not doing a full recursive search since that will be very long
      // for lots of pixels.
      assem = std::dynamic_pointer_cast<ICompAssembly>((*inst)[i]);
      if (assem) {
        for (int j = 0; j < assem->nelements(); j++) {
          det = std::dynamic_pointer_cast<RectangularDetector>((*assem)[j]);
          if (det) {
            detList.emplace_back(det);
          } else {
            // Also, look in the second sub-level for RectangularDetectors (e.g.
            // PG3).
            // We are not doing a full recursive search since that will be very
            // long for lots of pixels.
            assem2 = std::dynamic_pointer_cast<ICompAssembly>((*assem)[j]);
            if (assem2) {
              for (int k = 0; k < assem2->nelements(); k++) {
                det = std::dynamic_pointer_cast<RectangularDetector>((*assem2)[k]);
                if (det) {
                  detList.emplace_back(det);
                }
              }
            }
          }
        }
      }
    }
  }

  if (spectra.size() < 1)
    throw std::runtime_error("The number of spectra in the loaded file is zero.");

  MatrixWorkspace_sptr outWS = std::dynamic_pointer_cast<MatrixWorkspace>(
      API::WorkspaceFactory::Instance().create("Workspace2D", spectra.size(), spectra[0].size(), spectra[0].size()));
  outWS->setInstrument(inst);
  outWS->getAxis(0)->setUnit("TOF");
  outWS->setYUnit("Counts");
  outWS->setDistribution(true);
  outWS->rebuildSpectraMapping(false);

  // Go through each point at this run / bank
  for (size_t i = 0; i < spectra.size(); i++) {
    auto &outSpec = outWS->getSpectrum(i);
    outSpec.clearDetectorIDs();
    for (int j = 0; j < detList[i]->xpixels(); j++)
      for (int k = 0; k < detList[i]->ypixels(); k++)
        outSpec.addDetectorID(static_cast<detid_t>(detList[i]->getDetectorIDAtXY(j, k)));
    auto &outX = outSpec.mutableX();
    auto &outY = outSpec.mutableY();
    auto &outE = outSpec.mutableE();
    // This is the scattered beam direction
    V3D dir = detList[i]->getPos() - samplePos;

    // Find spectra at wavelength of 1 for normalization
    std::vector<double> xdata(1, 1.0); // wl = 1
    std::vector<double> ydata;
    double l2 = dir.norm();
    // Two-theta = polar angle = scattering angle = between +Z vector and the
    // scattered beam
    double theta2 = dir.angle(V3D(0.0, 0.0, 1.0));

    Mantid::Kernel::Unit_sptr unit = UnitFactory::Instance().create("Wavelength");
    unit->toTOF(xdata, ydata, l1, 0,
                {
                    {UnitParams::l2, l2},
                    {UnitParams::twoTheta, theta2},
                });
    double one = xdata[0];
    double spect1 = spectrumCalc(one, iSpec, time, spectra, i);

    for (size_t j = 0; j < spectra[i].size(); j++) {
      double spect = spectra[i][j];

      double relSigSpect = std::sqrt((1.0 / spect) + (1.0 / spect1));
      if (spect1 != 0.0) {
        spect /= spect1;
        outX[j] = time[i][j];
        outY[j] = spect;
        outE[j] = relSigSpect;
      } else {
        throw std::runtime_error("Wavelength for normalizing to spectrum is out of range.");
      }
    }
  }

  Algorithm_sptr convertAlg = createChildAlgorithm("ConvertToHistogram", 0.0, 0.2);
  convertAlg->setProperty<MatrixWorkspace_sptr>("InputWorkspace", outWS);
  // Now execute the convert Algorithm but allow any exception to bubble up
  convertAlg->execute();
  outWS = convertAlg->getProperty("OutputWorkspace");

  setProperty("OutputWorkspace", outWS);
}

double LoadIsawSpectrum::spectrumCalc(double TOF, int iSpec, const std::vector<std::vector<double>> &time,
                                      const std::vector<std::vector<double>> &spectra, size_t id) {
  double spect = 0;
  if (iSpec == 1) {
    //"Calculate the spectrum using spectral coefficients for the GSAS Type 2
    // incident spectrum."
    double T = TOF / 1000.; // time-of-flight in milliseconds

    double c1 = spectra[id][0];
    double c2 = spectra[id][1];
    double c3 = spectra[id][2];
    double c4 = spectra[id][3];
    double c5 = spectra[id][4];
    double c6 = spectra[id][5];
    double c7 = spectra[id][6];
    double c8 = spectra[id][7];
    double c9 = spectra[id][8];
    double c10 = spectra[id][9];
    double c11 = spectra[id][10];

    spect = c1 + c2 * exp(-c3 / std::pow(T, 2)) / std::pow(T, 5) + c4 * exp(-c5 * std::pow(T, 2)) +
            c6 * exp(-c7 * std::pow(T, 3)) + c8 * exp(-c9 * std::pow(T, 4)) + c10 * exp(-c11 * std::pow(T, 5));
  } else {
    size_t i = 1;
    for (i = 1; i < spectra[0].size() - 1; ++i)
      if (TOF < time[id][i])
        break;
    spect = spectra[id][i - 1] +
            (TOF - time[id][i - 1]) / (time[id][i] - time[id][i - 1]) * (spectra[id][i] - spectra[id][i - 1]);
  }

  return spect;
}
//----------------------------------------------------------------------------------------------
/** For use by getInstrument3Ways, initializes the properties
 * @param alg :: algorithm to which to add the properties.
 * */
void LoadIsawSpectrum::getInstrument3WaysInit(Algorithm *alg) {
  std::string grpName("Specify the Instrument");

  alg->declareProperty(
      std::make_unique<WorkspaceProperty<>>("InputWorkspace", "", Direction::Input, PropertyMode::Optional),
      "Optional: An input workspace with the instrument we want to use.");

  alg->declareProperty(std::make_unique<PropertyWithValue<std::string>>("InstrumentName", "", Direction::Input),
                       "Optional: Name of the instrument to base the "
                       "GroupingWorkspace on which to base the "
                       "GroupingWorkspace.");

  alg->declareProperty(std::make_unique<FileProperty>("InstrumentFilename", "", FileProperty::OptionalLoad, ".xml"),
                       "Optional: Path to the instrument definition file on "
                       "which to base the GroupingWorkspace.");

  alg->setPropertyGroup("InputWorkspace", grpName);
  alg->setPropertyGroup("InstrumentName", grpName);
  alg->setPropertyGroup("InstrumentFilename", grpName);
}

//----------------------------------------------------------------------------------------------
/** Get a pointer to an instrument in one of 3 ways: InputWorkspace,
 * InstrumentName, InstrumentFilename
 * @param alg :: algorithm from which to get the property values.
 * */
Geometry::Instrument_const_sptr LoadIsawSpectrum::getInstrument3Ways(Algorithm *alg) {
  MatrixWorkspace_sptr inWS = alg->getProperty("InputWorkspace");
  std::string InstrumentName = alg->getPropertyValue("InstrumentName");
  std::string InstrumentFilename = alg->getPropertyValue("InstrumentFilename");

  // Some validation
  int numParams = 0;
  if (inWS)
    numParams++;
  if (!InstrumentName.empty())
    numParams++;
  if (!InstrumentFilename.empty())
    numParams++;

  if (numParams > 1)
    throw std::invalid_argument("You must specify exactly ONE way to get an "
                                "instrument (workspace, instrument name, or "
                                "IDF file). You specified more than one.");
  if (numParams == 0)
    throw std::invalid_argument("You must specify exactly ONE way to get an "
                                "instrument (workspace, instrument name, or "
                                "IDF file). You specified none.");

  // ---------- Get the instrument one of 3 ways ---------------------------
  Instrument_const_sptr inst;
  if (inWS) {
    inst = inWS->getInstrument();
  } else {
    Algorithm_sptr childAlg = alg->createChildAlgorithm("LoadInstrument", 0.0, 0.2);
    MatrixWorkspace_sptr tempWS(new Workspace2D());
    childAlg->setProperty<MatrixWorkspace_sptr>("Workspace", tempWS);
    childAlg->setPropertyValue("Filename", InstrumentFilename);
    childAlg->setPropertyValue("InstrumentName", InstrumentName);
    childAlg->setProperty("RewriteSpectraMap", Mantid::Kernel::OptionalBool(false));
    childAlg->executeAsChildAlg();
    inst = tempWS->getInstrument();
  }

  return inst;
}

} // namespace Mantid::Crystal
