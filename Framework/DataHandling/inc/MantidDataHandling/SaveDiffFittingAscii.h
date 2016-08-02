#ifndef MANTID_DATAHANDLING_SAVEDIFFFITTINGASCII_H_
#define MANTID_DATAHANDLING_SAVEDIFFFITTINGASCII_H_

//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
#include "MantidAPI/Algorithm.h"
#include "MantidAPI/ITableWorkspace_fwd.h"

namespace Mantid {
namespace DataHandling {

class DLLExport SaveDiffFittingAscii : public Mantid::API::Algorithm {
public:
  /// (Empty) Constructor
  SaveDiffFittingAscii();

  /// Algorithm's name
  const std::string name() const override { return "SaveDiffFittingAscii"; }

  /// Summary of algorithms purpose
  const std::string summary() const override {
    return "Saves the results after carrying out single peak fitting process "
           "or running "
           "EnggFitPeaks v1 algorithm to ASCII file";
  }

  /// Algorithm's version
  int version() const override { return (1); }

  /// Algorithm's category for identification overriding a virtual method
  const std::string category() const override { return "DataHandling\\Text"; }

private:
  /// Initialisation code
  void init() override;

  /// Execution code
  void exec() override;

  /// Process two groups and ensure the Result string is set properly on the
  /// final algorithm
  bool processGroups() override;

  /// Main exec routine, called for group or individual workspace processing.
  void processAll();

  void writeInfo(const std::string &runNumber, const std::string &bank,
                 std::ofstream &file);

  void writeHeader(std::vector<std::string> &columnHeadings,
                   std::ofstream &file);

  void writeData(API::ITableWorkspace_sptr workspace, std::ofstream &file,
                 size_t columnSize);

  void writeVal(std::string &val, std::ofstream &file, bool endline);

  /// the separator
  const char m_sep;

  /// next line
  const char m_endl;

  /// table_counter
  int m_counter;

  std::vector<API::ITableWorkspace_sptr> m_workspaces;
};
} // namespace DataHandling
} // namespace Mantid

#endif /*  MANTID_DATAHANDLING_SAVEDIFFFITTINGASCII_H_  */
