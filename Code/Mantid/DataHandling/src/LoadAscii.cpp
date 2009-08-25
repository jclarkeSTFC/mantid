//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
#include "MantidDataHandling/LoadAscii.h"
#include "MantidDataObjects/Workspace2D.h"
#include "MantidKernel/UnitFactory.h"

#include <fstream>
#include <boost/tokenizer.hpp>

namespace Mantid
{
  namespace DataHandling
  {
    // Register the algorithm into the algorithm factory
    DECLARE_ALGORITHM(LoadAscii)

    using namespace Kernel;
    using namespace API;

    // Initialise the logger
    Logger& LoadAscii::g_log = Logger::get("LoadAscii");

    /// Initialisation method.
    void LoadAscii::init()
    {
      std::vector<std::string> exts;
      exts.push_back("dat");
      exts.push_back("txt");
      exts.push_back("csv");

      declareProperty("Filename","",new FileValidator(exts),"A comma separated Ascii file");
      declareProperty(new WorkspaceProperty<DataObjects::Workspace2D>("Workspace",
        "",Direction::Output), "The name of the workspace that will be created.");

	  m_seperator_options.push_back("CSV");
	  m_seperator_options.push_back("Tab");
	  m_seperator_options.push_back("Space");
	  declareProperty("Separator", "CSV", new ListValidator(m_seperator_options));

	  m_separatormap.insert(separator_pair("CSV",","));
	  m_separatormap.insert(separator_pair("Tab","\t"));
	  m_separatormap.insert(separator_pair("Space"," "));

    }

    /** 
     *   Executes the algorithm.
     */
    void LoadAscii::exec()
    {
        std::string filename = getProperty("Filename");
		std::string separator =getProperty("Separator");
        std::ifstream file(filename.c_str());

        file.seekg (0, std::ios::end);
        Progress progress(this,0,1,file.tellg());
        file.seekg (0, std::ios::beg);

        std::string str;

        std::vector<DataObjects::Histogram1D> spectra;
        size_t iLine=0;    // line number
        size_t ncols = 0;  // number of comma-separated columns
        size_t nSpectra = 0;
        size_t nBins = 0;
		std::map<std::string,const char*>::const_iterator it;
		it=m_separatormap.find(separator);
		const char* ch;
		if(it!=m_separatormap.end())
		{  //get the key for the selected separator string
			ch=it->second;
		}
	    while(getline(file,str))
        {   ++iLine;
            progress.report(file.tellg());
            if (str.empty()) continue;
            typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
			boost::char_separator<char> sep(ch);
            boost::tokenizer<boost::char_separator<char> > values(str, sep);

            // define sizes
            if (ncols == 0)
            {
                for (tokenizer::iterator it = values.begin(); it != values.end(); ++it)
                    ++ncols;
                if (ncols % 2 == 0 || ncols < 3)
                    throw std::runtime_error("Input file must contain odd (> 2) number of comma-separated columns");
                nSpectra = (ncols-1) / 2;
                spectra.resize( nSpectra );
            }

            // read in the numbers
            bool numeric = true;
            std::vector<double> input;
            for (tokenizer::iterator it = values.begin(); it != values.end(); ++it)
            {
                std::istringstream istr(*it);
                double d;
                istr >> d;
                if (istr.fail())  // non-numeric data
                {
                    numeric = false;
                    break;
                }
                input.push_back(d);
            }
            if (!numeric) continue;
            if (ncols != input.size())
            {
                std::ostringstream ostr;
                ostr << "Number of columns changed in line " << iLine;
                throw std::runtime_error(ostr.str());
            }
            for(unsigned int i=0;i<nSpectra;i++)
            {
                spectra[i].dataX().push_back(input[0]);
                spectra[i].dataY().push_back(input[i*2+1]);
                spectra[i].dataE().push_back(input[i*2+2]);
            }
            ++nBins;
        } // read the file

        DataObjects::Workspace2D_sptr localWorkspace = boost::dynamic_pointer_cast<DataObjects::Workspace2D>
            (WorkspaceFactory::Instance().create("Workspace2D",nSpectra,nBins,nBins));
//        localWorkspace->setTitle(title);
        localWorkspace->getAxis(0)->unit() = UnitFactory::Instance().create("Energy");

        for(unsigned int i=0;i<nSpectra;i++)
        {
            localWorkspace->dataX(i) = spectra[i].dataX();
            localWorkspace->dataY(i) = spectra[i].dataY();
            localWorkspace->dataE(i) = spectra[i].dataE();
        }

        setProperty("Workspace",localWorkspace);
    }

  } // namespace DataHandling
} // namespace Mantid
