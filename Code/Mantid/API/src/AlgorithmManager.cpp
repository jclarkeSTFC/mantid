#include <iomanip>
#include <iostream>
#include <vector>

#include "MantidAPI/IAlgorithm.h"
#include "MantidAPI/MantidKernel/StatusCode.h"
#include "MantidAPI/AlgorithmManager.h"
#include "MantidKernel/Exception.h"

namespace Mantid
{
namespace Kernel
{
	Logger& AlgorithmManager::g_log = Logger::get("AlgorithmManager");
	AlgorithmManager* AlgorithmManager::m_instance = 0;
	
  /// Private Constructor for singleton class
	AlgorithmManager::AlgorithmManager() : DynamicFactory<IAlgorithm>(),
	   no_of_alg(0)
	{
	}

  /** Private destructor
   *  Prevents client from calling 'delete' on the pointer handed 
   *  out by Instance
   */
	AlgorithmManager::~AlgorithmManager()
	{
		clear();
	}
        
  /** Creates an instance of an algorithm, but does not own that instance
   * 
   *  @param  algName The name of the algorithm required
   *  @return A pointer to the created algorithm
   *  @throw  NotFoundError Thrown if algorithm requested is not registered
   */
	IAlgorithm* AlgorithmManager::createUnmanaged(const std::string& algName) const
	{
	    return DynamicFactory<IAlgorithm>::create(algName);                // Throws on fail:
	}

  /** Creates an instance of an algorithm
   *
   *  @param  algName The name of the algorithm required
   *  @return A pointer to the created algorithm
   *  @throw  NotFoundError Thrown if algorithm requested is not registered
   *  @throw  std::runtime_error Thrown if properties string is ill-formed
   */
	IAlgorithm* AlgorithmManager::create(const std::string& algName)
	{
	   regAlg.push_back(DynamicFactory<IAlgorithm>::create(algName));      // Throws on fail:
	   StatusCode status = regAlg.back()->initialize();
	   if (status.isFailure())
	   {
	     throw std::runtime_error("AlgorithmManager:: Unable to initialise algorithm " + algName); 
	   }
	   no_of_alg++;		
	   return regAlg.back();
	}

  /** A static method which retrieves the single instance of the Algorithm Manager
  * 
  *  @returns A pointer to the Algorithm Manager instance
  */
	AlgorithmManager* AlgorithmManager::Instance()
	{
	     if (!m_instance) m_instance = new AlgorithmManager;	 		
	     return m_instance;
	}

  /// Finalizes and deletes all registered algorithms
	void AlgorithmManager::clear()
	{
	  int errOut(0);
	  std::vector<IAlgorithm*>::iterator vc;
	  for(vc=regAlg.begin();vc!=regAlg.end();vc++)
	  {
	    // no test for zero since impossible 
	    StatusCode status = (*vc)->finalize();
	    errOut+= status.isFailure();
	    delete (*vc);
		}
	  regAlg.clear();
	  no_of_alg=0;
	  if (errOut) throw std::runtime_error("AlgorithmManager:: Unable to finalise algorithm " ); 
	  return;
	}
	
	
} // namespace Kernel
} // namespace Mantid
