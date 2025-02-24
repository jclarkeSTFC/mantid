// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidDataHandling/ModifyDetectorDotDatFile.h"
#include "MantidAPI/ExperimentInfo.h"
#include "MantidAPI/FileProperty.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/Workspace.h"
#include "MantidGeometry/Instrument.h"
#include "MantidGeometry/Instrument/DetectorInfo.h"
#include "MantidGeometry/Instrument/RectangularDetector.h"
#include <fstream>

using namespace Mantid::Kernel;
using namespace Mantid::API;
using namespace Mantid::Geometry;

namespace Mantid::DataHandling {

// Register the algorithm into the AlgorithmFactory
DECLARE_ALGORITHM(ModifyDetectorDotDatFile)

//----------------------------------------------------------------------------------------------
/** Initialize the algorithm's properties.
 */
void ModifyDetectorDotDatFile::init() {
  declareProperty(std::make_unique<WorkspaceProperty<Workspace>>("InputWorkspace", "", Direction::Input),
                  "Workspace with detectors in the positions to be put into the detector "
                  "dot dat file");

  std::initializer_list<std::string> exts = {".dat", ".txt"};

  declareProperty(std::make_unique<FileProperty>("InputFilename", "", FileProperty::Load, exts),
                  "Path to a detector dot dat file. Must be of type .dat or .txt");

  declareProperty(std::make_unique<FileProperty>("OutputFilename", "", FileProperty::Save, exts),
                  "Path to the modified detector dot dat file. Must be of type .dat or "
                  ".txt");
}

//----------------------------------------------------------------------------------------------
/** Execute the algorithm.
 */
void ModifyDetectorDotDatFile::exec() {
  std::string inputFilename = getPropertyValue("InputFilename");
  std::string outputFilename = getPropertyValue("OutputFilename");

  Workspace_sptr ws1 = getProperty("InputWorkspace");
  ExperimentInfo_sptr ws = std::dynamic_pointer_cast<ExperimentInfo>(ws1);

  // Check instrument
  Instrument_const_sptr inst = ws->getInstrument();
  if (!inst)
    throw std::runtime_error("No instrument in the Workspace. Cannot modify detector dot dat file");

  // Open files
  std::ifstream in;
  in.open(inputFilename.c_str());
  if (!in) {
    throw Exception::FileError("Can't open input file", inputFilename);
  }
  std::ofstream out;
  out.open(outputFilename.c_str());
  if (!out) {
    in.close();
    throw Exception::FileError("Can't open output file", outputFilename);
  }

  // Read first line, modify it and put into output file
  std::string str;
  getline(in, str);
  out << str << " and modified by MANTID algorithm ModifyDetectorDotDatFile \n";

  // Read second line to check number of detectors and columns
  int detectorCount, numColumns;
  getline(in, str);
  std::istringstream header2(str);
  // what you get from the header is the Number_of_user_table_parameters
  // while the number of columns must add the 5 required for the data format
  header2 >> detectorCount >> numColumns;
  numColumns += 5;
  out << str << "\n";
  // check that we have at least 1 detector and six columns
  // and a reasonable number of columns. This is because, if there is not column
  // specified, he will get a very large number of columns.
  if (detectorCount < 1 || numColumns < 5 || numColumns > 1000) {
    out.close();
    in.close();
    throw Exception::FileError("Incompatible file format found when reading line 2 in the input file", inputFilename);
  }

  // Copy column title line
  getline(in, str);
  out << str << "\n";
  // Format details
  int pOffset = 3; // Precision of Offset
  int pOther = 5;  // Precision of Other floats
  int wDet = 9;    // Field width of Detector ID
  int wOff = 8;    // Field width of Offset
  int wRad = 10;   // Field width of Radius
  int wCode = 6;   // Field width of Code
  int wAng = 12;   // Field width of angles

  const auto &detectorInfo = ws->detectorInfo();

  // Read input file line by line, modify line as necessary and put line into
  // output file
  while (getline(in, str)) {

    std::istringstream istr(str);

    detid_t detID;
    double offset;
    int code;
    float dump; // ignored data

    if (str.empty() || str[0] == '#') { // comments and empty lines are allowed and just copied
      out << str << "\n";
      continue;
    }

    // First five columns in the file, the detector ID and a code for the type
    // of detector CODE = 3 (psd gas tube)
    istr >> detID >> offset >> dump >> code >> dump;
    if (numColumns > 5)
      istr >> dump; // get phi

    if (code == 3) {
      try {
        // indexOf throws for invalided detID
        V3D pos = detectorInfo.position(detectorInfo.indexOf(detID));
        double l2;
        double theta;
        double phi;
        pos.getSpherical(l2, theta, phi);
        std::streampos width = istr.tellg(); // Amount of string to replace
        // Some experimenting with line manipulation
        std::ostringstream oss;
        oss << std::fixed << std::right;
        oss.precision(pOffset);
        oss << std::setw(wDet) << detID << std::setw(wOff) << offset;
        oss.precision(pOther);
        oss << std::setw(wRad) << l2 << std::setw(wCode) << code << std::setw(wAng) << theta << std::setw(wAng);
        if (numColumns > 5)
          oss << phi; // insert phi
        std::string prefix = oss.str();
        std::string suffix = str.substr(width, std::string::npos);
        out << prefix << suffix << "\n";
      } catch (std::out_of_range &) { // Detector not found, don't modify
        out << str << "\n";
      }
    } else {
      // We do not modify any other type of line
      out << str << "\n";
    }
  }

  out.close();
  in.close();
}

} // namespace Mantid::DataHandling
