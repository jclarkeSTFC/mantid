#ifndef MANTID_DATAHANDLING_LOADFITS_H_
#define MANTID_DATAHANDLING_LOADFITS_H_

#include "MantidAPI/IFileLoader.h"
#include <string>
#include <sstream>
#include <map>
#include <vector>

using namespace std;

struct FITSInfo {  
  vector<string> headerItems;
  map<string, string> headerKeys;
  int bitsPerPixel;
  int numberOfAxis;
  vector<int> axisPixelLengths;
  double tof;
  double timeBin;
  long int countsInImage;
  long int numberOfTriggers;
  string extension;
  string filePath;
}; 

namespace Mantid
{
namespace DataHandling
{
  /** LoadFITS : Load FITS files to TableWorkspace(s)
    
    Copyright &copy; 2014 ISIS Rutherford Appleton Laboratory & NScD Oak Ridge National Laboratory

    This file is part of Mantid.

    Mantid is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    Mantid is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    File change history is stored at: <https://github.com/mantidproject/mantid>
    Code Documentation is available at: <http://doxygen.mantidproject.org>
  */

  class DLLExport LoadFITS : public API::IFileLoader<Kernel::FileDescriptor>
  {
  public:
    LoadFITS() {}
    virtual ~LoadFITS() {}

    /// Algorithm's name for identification overriding a virtual method
    virtual const std::string name() const { return "LoadFITS" ;}

    ///Summary of algorithms purpose
    virtual const std::string summary() const {return "Load data from FITS files.";}

    /// Algorithm's version for identification overriding a virtual method
    virtual int version() const { return 1 ;}

    /// Algorithm's category for identification overriding a virtual method
    virtual const std::string category() const { return "DataHandling";}

    /// Returns a confidence value that this algorithm can load a file
    virtual int confidence(Kernel::FileDescriptor & descriptor) const;

  private:
    /// Initialisation code
    void init();
    /// Execution code
    void exec();
    
    /// Parses the header values for the FITS file
    bool parseHeader(FITSInfo &headerInfo);
    void loadSingleBinFromFile(Mantid::API::MatrixWorkspace_sptr &workspace, FITSInfo &fitsInfo, MantidVecPtr &x, long spetraCount, long binIndex);

    API::MatrixWorkspace_sptr initAndPopulateHistogramWorkspace();

    vector<FITSInfo> m_allHeaderInfo;

    ///// Implement abstract Algorithm methods
    //void init();
    ///// Implement abstract Algorithm methods
    //void exec();

    ///// Load file to a vector of strings
    //void loadFile(std::string filename, std::vector<std::string>& lines);

    ///// Get Histogram type
    //std::string getHistogramType(const std::vector<std::string>& lines);

    ///// Get Number of banks
    //size_t getNumberOfBanks(const std::vector<std::string>& lines);

    ///// Scan imported file for bank information
    //void scanBanks(const std::vector<std::string>& lines, std::vector<size_t>& bankStartIndex );

    ///// Parse bank in file to a map
    //void parseBank(std::map<std::string, double>& parammap, const std::vector<std::string>& lines, size_t bankid, size_t startlineindex, int nProf);

    ///// Find first INS line at or after lineIndex
    //size_t findINSLine(const std::vector<std::string>& lines, size_t lineIndex);

    ///// Generate output workspace
    //DataObjects::TableWorkspace_sptr genTableWorkspace(std::map<size_t, std::map<std::string, double> > bankparammap);


  };
  

} // namespace DataHandling
} // namespace Mantid

#endif // MANTID_DATAHANDLING_LOADFITS_H_
