//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
#include "MantidAlgorithms/Regroup.h"
#include "MantidAPI/Workspace.h"
#include "MantidAPI/WorkspaceValidators.h"
#include "MantidKernel/ArrayProperty.h"

#include <sstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <math.h>

#include <iostream>

namespace Mantid
{
namespace Algorithms
{

/// Custom validator for the "params" parameter of this algorithm
class RegroupParamsValidator: public Kernel::IValidator<std::vector<double> >
{
public:
  RegroupParamsValidator() {}
  virtual ~RegroupParamsValidator() {}

  const std::string getType() const { return ""; }
  const bool isValid( const std::vector<double> &value ) const
  {
    if (value.empty()) return false;
    if (value.size()/2*2 == value.size() && value.size() == 1) return false;
    double previous = value[0];
    for(int i=2;i<int(value.size());i+=2)
    {
      if (value[i] <= previous) return false;
      else previous = value[i];
    }
    return true;
  }

  Kernel::IValidator<std::vector<double> >* clone() { return new RegroupParamsValidator(*this); }
};

// Register the class into the algorithm factory
DECLARE_ALGORITHM(Regroup)

using namespace Kernel;
using API::WorkspaceProperty;
using API::Workspace_sptr;
using API::Workspace_const_sptr;
using API::Workspace;

// Get a reference to the logger
Logger& Regroup::g_log = Logger::get("Regroup");

/// Initialisation method. Declares properties to be used in algorithm.
void Regroup::init()
{
  API::CompositeValidator<> *wsVal = new API::CompositeValidator<>;
  wsVal->add(new API::HistogramValidator<>);
  wsVal->add(new API::CommonBinsValidator<>);
  declareProperty(new WorkspaceProperty<Workspace>("InputWorkspace","",Direction::Input, wsVal));
  declareProperty(new WorkspaceProperty<Workspace>("OutputWorkspace","",Direction::Output));

  declareProperty(new ArrayProperty<double>("params", new RegroupParamsValidator));
}

/** Executes the regroup algorithm
 *
 *  @throw runtime_error Thrown if
 */
void Regroup::exec()
{
  // retrieve the properties
  std::vector<double> rb_params=getProperty("params");

  // Get the input workspace
  Workspace_const_sptr inputW = getProperty("InputWorkspace");

  // can work only if all histograms have the same boundaries
  if (!hasSameBoundaries(inputW))
  {
    g_log.error("Histograms with different boundaries");
    throw std::runtime_error("Histograms with different boundaries");
  }

  bool dist = inputW->isDistribution();

  int histnumber = inputW->getNumberHistograms();
  std::vector<double> XValues_new;
  const std::vector<double> &XValues_old = inputW->dataX(0);
  std::vector<int> xoldIndex;// indeces of new x in XValues_old
  // create new output X axis
  int ntcnew = newAxis(rb_params,XValues_old,XValues_new,xoldIndex);

  // make output Workspace the same type is the input, but with new length of signal array
  API::Workspace_sptr outputW = API::WorkspaceFactory::Instance().create(inputW,histnumber,ntcnew,ntcnew-1);

  for (int hist=0; hist <  histnumber;hist++)
  {
    const API::IErrorHelper* e_ptr= inputW->errorHelper(hist);
    if(dynamic_cast<const API::GaussianErrorHelper*>(e_ptr) ==0)
    {
      g_log.error("Can only regroup Gaussian data");
      throw std::invalid_argument("Invalid input Workspace");
    }

    // get const references to input Workspace arrays (no copying)
    const std::vector<double>& XValues = inputW->dataX(hist);
    const std::vector<double>& YValues = inputW->dataY(hist);
    const std::vector<double>& YErrors = inputW->dataE(hist);

    //get references to output workspace data (no copying)
    std::vector<double>& YValues_new=outputW->dataY(hist);
    std::vector<double>& YErrors_new=outputW->dataE(hist);

    // output data arrays are implicitly filled by function
    rebin(XValues,YValues,YErrors,xoldIndex,YValues_new,YErrors_new, dist);

    // Populate the output workspace X values
    outputW->dataX(hist)=XValues_new;
  }

  outputW->isDistribution(dist);

  // Copy units
  if (outputW->getAxis(0)->unit().get())
    outputW->getAxis(0)->unit() = inputW->getAxis(0)->unit();
  try
  {
    if (inputW->getAxis(1)->unit().get())
      outputW->getAxis(1)->unit() = inputW->getAxis(1)->unit();
   }
  catch(Exception::IndexError) {
    // OK, so this isn't a Workspace2D
  }

  // Assign it to the output workspace property
  setProperty("OutputWorkspace",outputW);

  return;
}

/** Regroup the data according to new output X array
 *
 * @param xold - old x array of data
 * @param xoldIndex - indeces of new x in XValues_old
 * @param yold - old y array of data
 * @param ynew - new y array of data
 * @param eold - old error array of data
 * @param enew - new error array of data
 * @param distribution - flag defining if distribution data (1) or not (0)
 * @throw runtime_error Thrown if algorithm cannot execute
 * @throw invalid_argument Thrown if input to function is incorrect
 **/
void Regroup::rebin(const std::vector<double>& xold, const std::vector<double>& yold, const std::vector<double>& eold,
    const std::vector<int>& xoldIndex, std::vector<double>& ynew, std::vector<double>& enew, bool distribution)
{

  for(int i=0;i<int(xoldIndex.size()-1);i++)
  {

    int n = xoldIndex[i];// start the group
    int m = xoldIndex[i+1];// end the group
    double width = xold[m] - xold[n]; // width of the group
    //std::cerr<<n<<'-'<<m<<' '<<xold[n]
    if (width == 0.)
    {
      g_log.error("Zero bin width");
      throw std::runtime_error("Zero bin width");
    }
    /*
     *        yold contains counts/unit time, ynew contains counts
     *	       enew contains counts**2
     */
    if(distribution)
    {
      ynew[i] = 0.;
      enew[i] = 0.;
      for(int j=n;j<m;j++)
      {
        double wdt = xold[j+1] - xold[j]; // old bin width
        ynew[i] += yold[j]*wdt;
        enew[i] += eold[j]*eold[j]*wdt*wdt;
      }
      ynew[i] /= width;
      enew[i] = sqrt(enew[i])/width;
    }
    else// yold,eold data is not distribution but counts
    {
      ynew[i] = 0.;
      enew[i] = 0.;
      for(int j=n;j<m;j++)
      {
        ynew[i] += yold[j];
        enew[i] += eold[j]*eold[j];
      }
      enew[i] = sqrt(enew[i]);
    }
  }

  return; //without problems
}

/** Creates a new  output X array  according to specific boundary defnitions
 *
 * @param params    rebin parameters input [x_1, delta_1,x_2, ... ,x_n-1,delta_n-1,x_n)
 * @param xold      the current x array
 * @param xnew      new output workspace x array
 * @param xoldIndex indeces of new x in XValues_old
 **/
int Regroup::newAxis(const std::vector<double>& params,
    const std::vector<double>& xold, std::vector<double>& xnew,std::vector<int> &xoldIndex)
{
  double xcurr, xs;
  int ibound(2), istep(1), inew(0);
  int ibounds=params.size(); //highest index in params array containing a bin boundary
  int isteps=ibounds-1; // highest index in params array containing a step

  xcurr = params[0];
  std::vector<double>::const_iterator iup =
    std::find_if(xold.begin(),xold.end(),std::bind2nd(std::greater_equal<double>(),xcurr));
  if (iup != xold.end())
  {
    xcurr = *iup;
    xnew.push_back(xcurr);
    xoldIndex.push_back(inew);
    inew++;
  }
  else
    return 0;

  while( (ibound <= ibounds) && (istep <= isteps) )
  {
    // if step is negative then it is logarithmic step
    if ( params[istep] >= 0.0)
      xs = params[istep];
    else
      xs = xcurr * fabs(params[istep]);

    //xcurr += xs;

    // find nearest x_i that is >= xcurr
    iup = std::find_if(xold.begin(),xold.end(),std::bind2nd(std::greater_equal<double>(),xcurr+xs));
    if (iup != xold.end())
    {
      if (*iup <= params[ibound])
      {
        xcurr = *iup;
        xnew.push_back(xcurr);
        xoldIndex.push_back(inew);
        inew++;
      }
      else
      {
        ibound += 2;
        istep += 2;
      }
    }
    else
      return inew;
  }
  //returns length of new x array or -1 if failure
  return inew;
  //return( (ibound == ibounds) && (istep == isteps) ? inew : -1 );
}

/// Checks if all histograms have the same boundaries by comparing their sums
bool Regroup::hasSameBoundaries(const API::Workspace_const_sptr WS)
{
  if ( !WS->blocksize() || WS->getNumberHistograms() < 2) return true;
  double commonSum = std::accumulate(WS->dataX(0).begin(),WS->dataX(0).end(),0.);
  for (int i = 1; i < WS->getNumberHistograms(); ++i)
    //? Can the error be relative?
    if ( fabs( commonSum - std::accumulate(WS->dataX(i).begin(),WS->dataX(i).end(),0.) ) > 1e-7 )
      return false;
  return true;
}

} // namespace Algorithm
} // namespace Mantid
