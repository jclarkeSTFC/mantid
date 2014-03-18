#ifndef MANTID_SINQ_POLDIPEAKCOLLECTION_H
#define MANTID_SINQ_POLDIPEAKCOLLECTION_H

#include "MantidSINQ/DllConfig.h"
#include "MantidSINQ/PoldiUtilities/PoldiPeak.h"
#include "MantidDataObjects/TableWorkspace.h"
#include "boost/shared_ptr.hpp"

#include "MantidAPI/IPeakFunction.h"

namespace Mantid {
namespace Poldi {

/** PoldiPeakCollection :
 *
  PoldiPeakCollection stores PoldiPeaks and acts as a bridge
  to TableWorkspace

    @author Michael Wedel, Paul Scherrer Institut - SINQ
    @date 15/03/2014

    Copyright © 2014 PSI-MSS

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

using namespace Mantid::DataObjects;
using namespace Mantid::API;
using namespace Mantid::CurveFitting;

class PoldiPeakCollection;

typedef boost::shared_ptr<PoldiPeakCollection> PoldiPeakCollection_sptr;

class MANTID_SINQ_DLL PoldiPeakCollection
{
public:
    PoldiPeakCollection();
    PoldiPeakCollection(TableWorkspace_sptr workspace);
    ~PoldiPeakCollection() {}

    void setProfileFunction(IPeakFunction_sptr peakProfile);
    void setBackgroundFunction(IFunction_sptr backgroundFunction);
    void setProfileTies(std::string profileTies);

    const std::string &getProfileTies() const;

    size_t peakCount() const;

    void addPeak(PoldiPeak_sptr newPeak);
    PoldiPeak_sptr peak(size_t index) const;

    IFunction_sptr getPeakProfile(size_t index) const;
    void setProfileParameters(size_t index, IFunction_sptr fittedFunction);

    TableWorkspace_sptr asTableWorkspace();

private:
    void prepareTable(TableWorkspace_sptr table);
    void peaksToTable(TableWorkspace_sptr table);

    void constructFromTableWorkspace(TableWorkspace_sptr tableWorkspace);
    bool checkColumns(TableWorkspace_sptr tableWorkspace);

    void addPeakFunctionFromTemplate();

    void updatePeakProfileFunctions();
    void updatePeakBackgroundFunctions();

    double getFwhmRelation(IPeakFunction_sptr peakFunction);

    IPeakFunction_sptr m_profileTemplate;
    IFunction_sptr m_backgroundTemplate;
    std::string m_ties;

    std::vector<PoldiPeak_sptr> m_peaks;
    std::vector<IPeakFunction_sptr> m_peakProfiles;
    std::vector<IFunction_sptr> m_backgrounds;
};

}
}

#endif // MANTID_SINQ_POLDIPEAKCOLLECTION_H
