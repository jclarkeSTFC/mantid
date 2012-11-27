#ifndef MANTID_SLICEVIEWER_PEAKTRANSFORM_H_
#define MANTID_SLICEVIEWER_PEAKTRANSFORM_H_

#include "MantidKernel/V3D.h"

namespace MantidQt
{
  namespace SliceViewer
  {
    class DLLExport PeakTransform 
    {
    public:
      PeakTransform(const std::string& xPlotLabel, const std::string& yPlotLabel);
      virtual ~PeakTransform();
      PeakTransform(const PeakTransform& other);
      PeakTransform & operator=(const PeakTransform & other);
      Mantid::Kernel::V3D transform(const Mantid::Kernel::V3D& original) const;
    private:
      std::string m_xPlotLabel;
      std::string m_yPlotLabel;
      int m_indexOfPlotX;
      int m_indexOfPlotY;
      int m_indexOfPlotZ;
    };

  }
}

#endif /* MANTID_SLICEVIEWER_CONCRETEPEAKSPRESENTER_H_ */