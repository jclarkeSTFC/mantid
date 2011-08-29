#include "MantidAPI/ImplicitFunctionFactory.h"
#include "MantidGeometry/MDGeometry/MDBoxImplicitFunction.h"
#include "MantidGeometry/MDGeometry/MDHistoDimension.h"
#include "MantidKernel/CPUTimer.h"
#include "MantidKernel/Strings.h"
#include "MantidKernel/System.h"
#include "MantidKernel/Utils.h"
#include "MantidMDEvents/BinToMDHistoWorkspace.h"
#include "MantidMDEvents/IMDBox.h"
#include "MantidMDEvents/MDBox.h"
#include "MantidMDEvents/MDBoxIterator.h"
#include "MantidMDEvents/MDEventFactory.h"
#include "MantidMDEvents/MDEventWorkspace.h"
#include "MantidMDEvents/MDHistoWorkspace.h"
#include <boost/algorithm/string.hpp>
#include "MantidMDEvents/CoordTransformAffineParser.h"
#include <Poco/DOM/Element.h>
#include <Poco/DOM/Document.h>
#include <Poco/DOM/DOMParser.h>
#include "MantidMDEvents/CoordTransformAligned.h"

using Mantid::Kernel::CPUTimer;

namespace Mantid
{
namespace MDEvents
{

  // Register the algorithm into the AlgorithmFactory
  DECLARE_ALGORITHM(BinToMDHistoWorkspace)

  using namespace Mantid::Kernel;
  using namespace Mantid::API;
  using namespace Mantid::Geometry;


  //----------------------------------------------------------------------------------------------
  /** Constructor
   */
  BinToMDHistoWorkspace::BinToMDHistoWorkspace()
  {
  }

  //----------------------------------------------------------------------------------------------
  /** Destructor
   */
  BinToMDHistoWorkspace::~BinToMDHistoWorkspace()
  {
  }


  //----------------------------------------------------------------------------------------------
  /// Sets documentation strings for this algorithm
  void BinToMDHistoWorkspace::initDocs()
  {
    this->setWikiSummary("Take a [[MDEventWorkspace]] and bin into into a dense, multi-dimensional histogram workspace ([[MDHistoWorkspace]]).");
    this->setOptionalMessage("Take a MDEventWorkspace and bin into into a dense, multi-dimensional histogram workspace (MDHistoWorkspace).");
    this->setWikiDescription(
        "This algorithm performs dense binning of the events in multiple dimensions of an input [[MDEventWorkspace]] "
        "and places them into a dense MDHistoWorkspace with 1-4 dimensions."
        "\n\n"
        "The input MDEventWorkspace may have more dimensions than the number of output dimensions. "
        "The names of the dimensions in the DimX, etc. parameters are used to find the corresponding dimensions that will "
        "be created in the output."
        "\n\n"
        "An ImplicitFunction can be defined using the ImplicitFunctionXML parameter; any points NOT belonging inside of the "
        "ImplicitFunction will be set as NaN (not-a-number). "
        "\n\n"
        "As of now, binning is only performed along axes perpendicular to the dimensions defined "
        "in the MDEventWorkspace."
        );
  }

  //----------------------------------------------------------------------------------------------
  /** Initialize the algorithm's properties.
   */
  void BinToMDHistoWorkspace::init()
  {
    std::string dimHelp = "Enter it as a comma-separated list of values with the format: 'name,minimum,maximum,number_of_bins'. Leave blank for NONE.";
    declareProperty(new WorkspaceProperty<IMDEventWorkspace>("InputWorkspace","",Direction::Input), "An input MDEventWorkspace.");

    declareProperty(new PropertyWithValue<std::string>("DimX","",Direction::Input), "Binning parameters for the X dimension.\n" + dimHelp);
    declareProperty(new PropertyWithValue<std::string>("DimY","",Direction::Input), "Binning parameters for the Y dimension.\n" + dimHelp);
    declareProperty(new PropertyWithValue<std::string>("DimZ","",Direction::Input), "Binning parameters for the Z dimension.\n" + dimHelp);
    declareProperty(new PropertyWithValue<std::string>("DimT","",Direction::Input), "Binning parameters for the T dimension.\n" + dimHelp);

    declareProperty(new PropertyWithValue<std::string>("ImplicitFunctionXML","",Direction::Input),
        "XML string describing the implicit function determining which bins to use.");
    declareProperty(new PropertyWithValue<bool>("IterateEvents",true,Direction::Input),
        "Alternative binning method where you iterate through every event, placing them in the proper bin.\n"
        "This may be faster for workspaces with few events and lots of output bins.");
    declareProperty(new PropertyWithValue<bool>("Parallel",false,Direction::Input),
        "Temporary parameter: true to run in parallel.");

    declareProperty(new PropertyWithValue<std::string>("TransformationXML","",Direction::Input),
        "XML string describing the coordinate transformation that converts from the MDEventWorkspace dimensions to the output dimensions.");

    declareProperty(new WorkspaceProperty<Workspace>("OutputWorkspace","",Direction::Output), "A name for the output MDHistoWorkspace.");

  }


  //----------------------------------------------------------------------------------------------
  /** Generate a MDHistoDimension_sptr from a comma-sep string.
   *
   * @param str :: name,minimum,maximum,number_of_bins
   * @return
   */
  MDHistoDimension_sptr makeMDHistoDimensionFromString(const std::string & str)
  {
    if (str.empty())
    {
      // Make a blank dimension
      MDHistoDimension_sptr out;
      return out;
    }
    else
    {
      std::string name, id;
      double min, max;
      int numBins = 0;
      std::vector<std::string> strs;
      boost::split(strs, str, boost::is_any_of(","));
      if (strs.size() != 4)
        throw std::invalid_argument("Wrong number of values (4 are expected) in the dimensions string: " + str);
      // Extract the arguments
      name = Strings::strip(strs[0]);
      id = name;
      Strings::convert(strs[1], min);
      Strings::convert(strs[2], max);
      Strings::convert(strs[3], numBins);
      if (name.size() == 0)
        throw std::invalid_argument("Name should not be blank.");
      if (min >= max)
        throw std::invalid_argument("Min should be > max.");
      if (numBins < 1)
        throw std::invalid_argument("Number of bins should be >= 1.");

      MDHistoDimension_sptr out(new MDHistoDimension(name, id, "", min, max, numBins));
      return out;
    }
  }



  //----------------------------------------------------------------------------------------------
  /** Bin the contents of a MDBox
   *
   * @param box :: pointer to the MDBox to bin
   * @param chunkMin :: the minimum index in each dimension to consider "valid" (inclusive)
   * @param chunkMax :: the maximum index in each dimension to consider "valid" (exclusive)
   */
  template<typename MDE, size_t nd>
  inline void BinToMDHistoWorkspace::binMDBox(MDBox<MDE, nd> * box, size_t * chunkMin, size_t * chunkMax)
  {
    // An array to hold the rotated/transformed coordinates
    coord_t * outCenter = new coord_t[outD];

    // Evaluate whether the entire box is in the same bin
    if (box->getNPoints() > (1 << nd) * 2)
    {
      // There is a check that the number of events is enough for it to make sense to do all this processing.
      size_t numVertexes = 0;
      coord_t * vertexes = box->getVertexesArray(numVertexes);

      // All vertexes have to be within THE SAME BIN = have the same linear index.
      size_t lastLinearIndex = 0;
      bool badOne = false;

      for (size_t i=0; i<numVertexes; i++)
      {
        // Cache the center of the event (again for speed)
        const coord_t * inCenter = vertexes + i * nd;

        // Now transform to the output dimensions
        m_transform->apply(inCenter, outCenter);

        // To build up the linear index
        size_t linearIndex = 0;
        // To mark VERTEXES outside range
        badOne = false;

        /// Loop through the dimensions on which we bin
        for (size_t bd=0; bd<outD; bd++)
        {
          // What is the bin index in that dimension
          coord_t x = outCenter[bd];
          size_t ix = size_t(x);
          // Within range (for this chunk)?
          if ((x >= 0) && (ix >= chunkMin[bd]) && (ix < chunkMax[bd]))
          {
            // Build up the linear index
            linearIndex += indexMultiplier[bd] * ix;
          }
          else
          {
            // Outside the range
            badOne = true;
            break;
          }
        } // (for each dim in MDHisto)

        // Is the vertex at the same place as the last one?
        if (!badOne)
        {
          if ((i > 0) && (linearIndex != lastLinearIndex))
          {
            // Change of index
            badOne = true;
            break;
          }
          lastLinearIndex = linearIndex;
        }

        // Was the vertex completely outside the range?
        if (badOne)
          break;
      } // (for each vertex)

      delete [] vertexes;

      if (!badOne)
      {
        // Yes, the entire box is within a single bin
//        std::cout << "Box at " << box->getExtentsStr() << " is within a single bin.\n";
        // Add the CACHED signal from the entire box
        signals[lastLinearIndex] += box->getSignal();
        errors[lastLinearIndex] += box->getErrorSquared();
        // And don't bother looking at each event. This may save lots of time loading from disk.
        delete [] outCenter;
        return;
      }
    }

    // If you get here, you could not determine that the entire box was in the same bin.
    // So you need to iterate through events.

    const std::vector<MDE> & events = box->getConstEvents();
    typename std::vector<MDE>::const_iterator it = events.begin();
    typename std::vector<MDE>::const_iterator it_end = events.end();
    for (; it != it_end; it++)
    {
      // Cache the center of the event (again for speed)
      const coord_t * inCenter = it->getCenter();

      // Now transform to the output dimensions
      m_transform->apply(inCenter, outCenter);

      // To build up the linear index
      size_t linearIndex = 0;
      // To mark events outside range
      bool badOne = false;

      /// Loop through the dimensions on which we bin
      for (size_t bd=0; bd<outD; bd++)
      {
        // What is the bin index in that dimension
        coord_t x = outCenter[bd];
        size_t ix = size_t(x);
        // Within range (for this chunk)?
        if ((x >= 0) && (ix >= chunkMin[bd]) && (ix < chunkMax[bd]))
        {
          // Build up the linear index
          linearIndex += indexMultiplier[bd] * ix;
        }
        else
        {
          // Outside the range
          badOne = true;
          break;
        }
      } // (for each dim in MDHisto)

      if (!badOne)
      {
        signals[linearIndex] += it->getSignal();
        errors[linearIndex] += it->getErrorSquared();
      }
    }
    // Done with the events list
    box->releaseEvents();

    delete [] outCenter;
  }


  //----------------------------------------------------------------------------------------------
  /** Create an implicit function for picking boxes, based on the indexes in the
   * output MDHistoWorkspace.
   * This needs to be in the space of the INPUT MDEventWorkspace
   *
   * @param chunkMin :: the minimum index in each dimension to consider "valid" (inclusive)
   * @param chunkMax :: the maximum index in each dimension to consider "valid" (exclusive)
   * @return MDImplicitFunction
   */
  template<typename MDE, size_t nd>
  MDImplicitFunction * BinToMDHistoWorkspace::getImplicitFunctionForChunk(typename MDEventWorkspace<MDE, nd>::sptr ws,
      size_t * chunkMin, size_t * chunkMax)
  {
    UNUSED_ARG(ws);

    if (m_axisAligned)
    {
      std::vector<coord_t> function_min(nd, -1e50); // default to all space if the dimension is not specified
      std::vector<coord_t> function_max(nd, +1e50); // default to all space if the dimension is not specified
      for (size_t bd=0; bd<outD; bd++)
      {
        // Dimension in the MDEventWorkspace
        size_t d = dimensionToBinFrom[bd];
        function_min[d] = binDimensions[bd]->getX(chunkMin[bd]);
        function_max[d] = binDimensions[bd]->getX(chunkMax[bd]);
      }
      MDBoxImplicitFunction * function = new MDBoxImplicitFunction(function_min, function_max);
      return function;
    }
    else
    {
      // General implicit function
      // TODO: Apply the transform!
      return new MDImplicitFunction;
    }
  }

  //----------------------------------------------------------------------------------------------
  /** Perform binning by iterating through every event and placing them in the output workspace
   *
   * @param ws :: MDEventWorkspace of the given type.
   */
  template<typename MDE, size_t nd>
  void BinToMDHistoWorkspace::binByIterating(typename MDEventWorkspace<MDE, nd>::sptr ws)
  {
    // Start with signal at 0.0
    outWS->setTo(0.0, 0.0);

    // Cache some data to speed up accessing them a bit
    indexMultiplier = new size_t[outD];
    for (size_t d=0; d<outD; d++)
    {
      if (d > 0)
        indexMultiplier[d] = outWS->getIndexMultiplier()[d-1];
      else
        indexMultiplier[d] = 1;
    }
    signals = outWS->getSignalArray();
    errors = outWS->getErrorSquaredArray();

    // The dimension (in the output workspace) along which we chunk for parallel processing
    // TODO: Find the smartest dimension to chunk against
    size_t chunkDimension = 0;

    // How many bins (in that dimension) per chunk.
    // Try to split it so each core will get 2 tasks:
    int chunkNumBins =  int(binDimensions[chunkDimension]->getNBins() / (Mantid::Kernel::ThreadPool::getNumPhysicalCores() * 2));
    if (chunkNumBins < 1) chunkNumBins = 1;

    // Do we actually do it in parallel?
    bool doParallel = getProperty("Parallel");
    if (!doParallel)
      chunkNumBins = int(binDimensions[chunkDimension]->getNBins());

    // Total number of steps
    size_t progNumSteps = 0;

    // Run the chunks in parallel. There is no overlap in the output workspace so it is
    // thread safe to write to it..
    PRAGMA_OMP( parallel for schedule(dynamic,1) if (doParallel) )
    for(int chunk=0; chunk < int(binDimensions[chunkDimension]->getNBins()); chunk += chunkNumBins)
    {
      // Region of interest for this chunk.
      size_t * chunkMin = new size_t[outD];
      size_t * chunkMax = new size_t[outD];
      for (size_t bd=0; bd<outD; bd++)
      {
        // Same limits in the other dimensions
        chunkMin[bd] = 0;
        chunkMax[bd] = binDimensions[bd]->getNBins();
      }
      // Parcel out a chunk in that single dimension dimension
      chunkMin[chunkDimension] = size_t(chunk);
      if (size_t(chunk+chunkNumBins) > binDimensions[chunkDimension]->getNBins())
        chunkMax[chunkDimension] = binDimensions[chunkDimension]->getNBins();
      else
        chunkMax[chunkDimension] = size_t(chunk+chunkNumBins);

      // Build an implicit function (it needs to be in the space of the MDEventWorkspace)
      MDImplicitFunction * function = this->getImplicitFunctionForChunk<MDE,nd>(ws, chunkMin, chunkMax);

      // Use getBoxes() to get an array with a pointer to each box
      std::vector<IMDBox<MDE,nd>*> boxes;
      // Leaf-only; no depth limit; with the implicit function passed to it.
      ws->getBox()->getBoxes(boxes, 1000, true, function);

      // For progress reporting, the # of boxes
      if (prog)
      {
        PARALLEL_CRITICAL(BinToMDHistoWorkspace_progress)
        {
          std::cout << "Chunk " << chunk << ": found " << boxes.size() << " boxes within the implicit function." << std::endl;
          progNumSteps += boxes.size();
          prog->setNumSteps( progNumSteps );
        }
      }

      // Go through every box for this chunk.
      for (size_t i=0; i<boxes.size(); i++)
      {
        MDBox<MDE,nd> * box = dynamic_cast<MDBox<MDE,nd> *>(boxes[i]);
        // Perform the binning in this separate method.
        if (box)
          this->binMDBox(box, chunkMin, chunkMax);

        // Progress reporting
        if (prog) prog->report();

      }// for each box in the vector
    } // for each chunk in parallel



    // Now the implicit function
    if (implicitFunction)
    {
      prog->report("Applying implicit function.");
      signal_t nan = std::numeric_limits<signal_t>::quiet_NaN();
      outWS->applyImplicitFunction(implicitFunction, nan, nan);
    }
  }

  //----------------------------------------------------------------------------------------------
  /** Templated method to apply the binning operation to the particular
   * MDEventWorkspace passed in.
   *
   * @param ws :: MDEventWorkspace of the given type
   */
  template<typename MDE, size_t nd>
  void BinToMDHistoWorkspace::do_centerpointBin(typename MDEventWorkspace<MDE, nd>::sptr ws)
  {
    bool DODEBUG = true;

    CPUTimer tim;

    // Number of output binning dimensions found
    size_t outD = binDimensions.size();

    //Since the costs are not known ahead of time, use a simple FIFO buffer.
    ThreadScheduler * ts = new ThreadSchedulerFIFO();

    // Create the threadpool with: all CPUs, a progress reporter
    ThreadPool tp(ts, 0, prog);

    // Big efficiency gain is obtained by grouping a few bins per task.
    size_t binsPerTask = 100;

    // For progress reporting, the approx  # of tasks
    if (prog)
      prog->setNumSteps( int(outWS->getNPoints()/100) );

    // The root-level box.
    IMDBox<MDE,nd> * rootBox = ws->getBox();

    // This is the limit to loop over in each dimension
    size_t * index_max = new size_t[outD];
    for (size_t bd=0; bd<outD; bd++) index_max[bd] = binDimensions[bd]->getNBins();

    // Cache a calculation to convert indices x,y,z,t into a linear index.
    size_t * index_maker = new size_t[outD];
    Utils::NestedForLoop::SetUpIndexMaker(outD, index_maker, index_max);

    int numPoints = int(outWS->getNPoints());

    // Run in OpenMP with dynamic scheduling and a smallish chunk size (binsPerTask)
    // Right now, not parallel for file-backed systems.
    bool fileBacked = (ws->getBoxController()->getFile() != NULL);
    PRAGMA_OMP(parallel for schedule(dynamic, binsPerTask) if (!fileBacked)  )
    for (int i=0; i < numPoints; i++)
    {
      PARALLEL_START_INTERUPT_REGION

      size_t linear_index = size_t(i);
      // nd >= outD in all cases so this is safe.
      size_t index[nd];

      // Get the index at each dimension for this bin.
      Utils::NestedForLoop::GetIndicesFromLinearIndex(outD, linear_index, index_maker, index_max, index);

      // Construct the bin and its coordinates
      MDBin<MDE,nd> bin;
      for (size_t bd=0; bd<outD; bd++)
      {
        // Index in this binning dimension (i_x, i_y, etc.)
        size_t idx = index[bd];
        // Dimension in the MDEventWorkspace
        size_t d = dimensionToBinFrom[bd];
        // Corresponding extents
        bin.m_min[d] = binDimensions[bd]->getX(idx);
        bin.m_max[d] = binDimensions[bd]->getX(idx+1);
      }
      bin.m_index = linear_index;

      bool dimensionsUsed[nd];
      for (size_t d=0; d<nd; d++)
        dimensionsUsed[d] = (d<3);

      // Check if the bin is in the ImplicitFunction (if any)
      bool binContained = true;
      if (implicitFunction)
      {
        binContained = implicitFunction->isPointContained(bin.m_min); //TODO. Correct argument passed to this method?
      }

      if (binContained)
      {
        // Array of bools set to true when a dimension is fully contained (binary splitting only)
        bool fullyContained[nd];
        for (size_t d=0; d<nd; d++)
          fullyContained[d] = false;

        // This will recursively bin into the sub grids
        rootBox->centerpointBin(bin, fullyContained);

        // Save the data into the dense histogram
        outWS->setSignalAt(linear_index, bin.m_signal);
        outWS->setErrorAt(linear_index, bin.m_errorSquared);
      }

      // Report progress but not too often.
      if (((linear_index % 100) == 0) && prog ) prog->report();

      PARALLEL_END_INTERUPT_REGION
    } // (for each linear index)
    PARALLEL_CHECK_INTERUPT_REGION

    if (DODEBUG) std::cout << tim << " to run the openmp loop.\n";

    delete [] index_max;
    delete [] index_maker;
  }




  //----------------------------------------------------------------------------------------------
  /** Load and check the inputs in the bin dimensions parameters
   */
  void BinToMDHistoWorkspace::checkBinDimensions()
  {
    // Create the dimensions based on the strings from the user
    binDimensionsIn.push_back( makeMDHistoDimensionFromString( getPropertyValue("DimX")) );
    binDimensionsIn.push_back( makeMDHistoDimensionFromString( getPropertyValue("DimY")) );
    binDimensionsIn.push_back( makeMDHistoDimensionFromString( getPropertyValue("DimZ")) );
    binDimensionsIn.push_back( makeMDHistoDimensionFromString( getPropertyValue("DimT")) );

    // Thin down the input dimensions for any invalid ones
    for (size_t i = 0; i < binDimensionsIn.size(); ++i)
    {
      if (binDimensionsIn[i]) // (valid pointer?)
      {
        if (binDimensionsIn[i]->getNBins() == 0)
          throw std::runtime_error("Dimension " + binDimensionsIn[i]->getName() + " was set to have 0 bins. Cannot continue.");

        try {
          size_t dim_index = in_ws->getDimensionIndexByName(binDimensionsIn[i]->getName());
          dimensionToBinFrom.push_back(dim_index);
          binDimensions.push_back(binDimensionsIn[i]);
        }
        catch (std::runtime_error &)
        {
          // The dimension was not found, so we are not binning across it.
          if (binDimensionsIn[i]->getNBins() > 1)
            throw std::runtime_error("Dimension " + binDimensionsIn[i]->getName() + " was not found in the MDEventWorkspace and has more than one bin! Cannot continue.");
        }
      }
    }
    // Number of output binning dimensions found
    outD = binDimensions.size();

    if (outD == 0)
      throw std::runtime_error("No output dimensions were found in the MDEventWorkspace. Cannot bin!");
    if (outD > in_ws->getNumDims())
      throw std::runtime_error("More output dimensions were specified than input dimensions exist in the MDEventWorkspace. Cannot bin!");
  }


  //----------------------------------------------------------------------------------------------
  /** Using the parameters, create a coordinate transformation
   * for aligned cuts
   */
  void BinToMDHistoWorkspace::createAlignedTransform()
  {
    std::vector<coord_t> origin(outD), scaling(outD);
    for (size_t d=0; d<outD; d++)
    {
      origin[d] = binDimensions[d]->getMinimum();
      scaling[d] = coord_t(1.0) / binDimensions[d]->getBinWidth();
    }

    m_transform = new CoordTransformAligned(in_ws->getNumDims(), outD,
        dimensionToBinFrom, origin, scaling);
  }


  //----------------------------------------------------------------------------------------------
  /** Execute the algorithm.
   */
  void BinToMDHistoWorkspace::exec()
  {
    // Input MDEventWorkspace
    in_ws = getProperty("InputWorkspace");

    // Treat the input dimensions.
    checkBinDimensions();


    // De serialize the implicit function
    std::string ImplicitFunctionXML = getPropertyValue("ImplicitFunctionXML");
    implicitFunction = NULL;
    if (!ImplicitFunctionXML.empty())
      implicitFunction = Mantid::API::ImplicitFunctionFactory::Instance().createUnwrapped(ImplicitFunctionXML);

    // De-serialize the coordinate transformation XML string
    std::string TransformationXML = getPropertyValue("TransformationXML");
    m_transform = NULL;
    if (!TransformationXML.empty())
    {
      Poco::XML::DOMParser pParser;
      Poco::XML::Document* pDoc = pParser.parseString(TransformationXML);
      Poco::XML::Element* pRootElem = pDoc->documentElement();
      CoordTransformAffineParser parser;
      m_transform = parser.createTransform(pRootElem);
      m_axisAligned = false;
    }
    else
    {
      // Make an aligned transform out of the parameters
      this->createAlignedTransform();
      m_axisAligned = true;
    }

    prog = new Progress(this, 0, 1.0, 1); // This gets deleted by the thread pool; don't delete it in here.

    // Create the dense histogram. This allocates the memory
    outWS = MDHistoWorkspace_sptr(new MDHistoWorkspace(binDimensions));

    // Wrapper to cast to MDEventWorkspace then call the function
    bool IterateEvents = getProperty("IterateEvents");
    if (IterateEvents)
    {
      CALL_MDEVENT_FUNCTION(this->binByIterating, in_ws);
    }
    else
    {
      CALL_MDEVENT_FUNCTION(this->do_centerpointBin, in_ws);
    }

    // Save the output
    setProperty("OutputWorkspace", boost::dynamic_pointer_cast<Workspace>(outWS));
  }



} // namespace Mantid
} // namespace MDEvents
