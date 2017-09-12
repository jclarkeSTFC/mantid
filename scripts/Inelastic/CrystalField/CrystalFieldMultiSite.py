import numpy as np
from six import string_types


def makeWorkspace(xArray, yArray):
    """Create a workspace that doesn't appear in the ADS"""
    from mantid.api import AlgorithmManager
    alg = AlgorithmManager.createUnmanaged('CreateWorkspace')
    alg.initialize()
    alg.setChild(True)
    alg.setProperty('DataX', xArray)
    alg.setProperty('DataY', yArray)
    alg.setProperty('OutputWorkspace', 'dummy')
    alg.execute()
    return alg.getProperty('OutputWorkspace').value


class CrystalFieldMultiSite(object):

    field_parameter_names = ['BmolX', 'BmolY', 'BmolZ', 'BextX', 'BextY', 'BextZ',
                             'B20', 'B21', 'B22', 'B40', 'B41', 'B42', 'B43', 'B44', 'B60', 'B61', 'B62', 'B63', 'B64',
                             'B65', 'B66',
                             'IB21', 'IB22', 'IB41', 'IB42', 'IB43', 'IB44', 'IB61', 'IB62', 'IB63', 'IB64', 'IB65',
                             'IB66']

    def __init__(self, Ions, Symmetries, **kwargs):

        self._makeFunction(Ions, Symmetries)
        self.Ions = Ions
        self.Symmetries = Symmetries
        self._plot_window = {}
        self.chi2 = None
        parameter_dict = None
        attribute_dict = None

        free_parameters = []
        if 'parameters' in kwargs:
            parameter_dict = kwargs['parameters']
            del kwargs['parameters']
        if 'attributes' in kwargs:
            attribute_dict = kwargs['attributes']
            del kwargs['attributes']
        if 'Temperatures' in kwargs:
            self.Temperatures = kwargs['Temperatures']
            del kwargs['Temperatures']
            if 'FWHMs' in kwargs:
                self.FWHMs = kwargs['FWHMs']
                del kwargs['FWHMs']
            elif 'ResolutionModel' in kwargs:
                self.ResolutionModel = kwargs['ResolutionModel']
                del kwargs['ResolutionModel']
            else:
                raise RuntimeError("If temperatures are set, must also set FWHMs or ResolutionModel")
        for key in kwargs:
            if key == 'ToleranceEnergy':
                self.ToleranceEnergy = kwargs[key]
            elif key == 'ToleranceIntensity':
                self.ToleranceIntensity = kwargs[key]
            elif key == 'NPeaks':
                self.NPeaks = kwargs[key]
            elif key == 'FWHMVariation':
                self.FWHMVariation = kwargs[key]
            elif key == 'FixAllPeaks':
                self.FixAllPeaks = kwargs[key]
            elif key == 'PeakShape':
                self.PeakShape = kwargs[key]
            elif key == 'PhysicalProperty':
                self.PhysicalProperty = kwargs[key]
            else:
                # Crystal field parameters
                self.function.setParameter(key, kwargs[key])
                free_parameters.append(key)
        if not self._isMultiSite():
            for param in CrystalFieldMultiSite.field_parameter_names:
                if param not in free_parameters:
                    self.function.fixParameter(param)

        if attribute_dict is not None:
            for name, value in attribute_dict.items():
                self.function.setAttributeValue(name, value)
        if parameter_dict is not None:
            for name, value in parameter_dict.items():
                self.function.setParameter(name, value)

    @staticmethod
    def iterable_to_string(iterable):
        values_as_string = ""
        for element in iterable:
            values_as_string += ","
            values_as_string += element
        values_as_string = values_as_string[1:]
        return values_as_string

    def _isMultiSite(self):
        return len(self.Ions) > 1

    def _makeFunction(self, ion, symmetry):
        from mantid.simpleapi import FunctionFactory
        self.function = FunctionFactory.createFunction('CrystalFieldFunction')

    def getParameter(self, param):
        print self.function.numParams()
        self.function.getParameterValue(param)

    def getSpectrum(self, workspace, i=0, ws_index=None):
        """
        Get the i-th spectrum calculated with the current field and peak parameters.

        Alternatively can be called getSpectrum(workspace, ws_index). Spectrum index i is assumed zero.

        Examples:
            cf.getSpectrum(1, ws, 5) # Calculate the second spectrum using the x-values from the 6th spectrum
                                     # in workspace ws.
            cf.getSpectrum(ws) # Calculate the first spectrum using the x-values from the 1st spectrum
                               # in workspace ws.
            cf.getSpectrum(ws, 3) # Calculate the first spectrum using the x-values from the 4th spectrum
                                  # in workspace ws.

        @param i: Index of a spectrum to get.
        @param workspace: A workspace to base on.
        @param ws_index:  An index of a spectrum from workspace to use.
        @return: A tuple of (x, y) arrays
        """
        wksp = workspace
        if isinstance(wksp, int): # allow spectrum index to be passed as first argument
            wksp = i
            i = workspace
        elif not isinstance(i, int):
            raise RuntimeError('Spectrum index is expected to be int. Got %s' % i.__class__.__name__)
        elif ws_index is None: # else allow ws_index to be second argument
            ws_index = i
            i = 0
        if ws_index is None: # if ws_index not specified, set to default
            ws_index = 0

        if self.Temperatures[i] < 0:
            raise RuntimeError('You must first define a temperature for the spectrum')

        if isinstance(wksp, list) or isinstance(wksp, np.ndarray):
            xArray = wksp
            yArray = np.zeros_like(xArray)
            wksp = makeWorkspace(xArray, yArray)
            ws_index = 0
        return self._calcSpectrum(i, wksp, ws_index)

    def _calcSpectrum(self, i, workspace, ws_index, funstr=None):
        """Calculate i-th spectrum.

        @param i: Index of a spectrum or function string
        @param workspace: A workspace used to evaluate the spectrum function.
        @param ws_index:  An index of a spectrum in workspace to use.
        """
        from mantid.api import AlgorithmManager
        alg = AlgorithmManager.createUnmanaged('EvaluateFunction')
        alg.initialize()
        alg.setChild(True)
        alg.setProperty('Function', self.makeSpectrumFunction(i))
        alg.setProperty("InputWorkspace", workspace)
        alg.setProperty('WorkspaceIndex', ws_index)
        alg.setProperty('OutputWorkspace', 'dummy')
        alg.execute()
        out = alg.getProperty('OutputWorkspace').value
        # Create copies of the x and y because `out` goes out of scope when this method returns
        # and x and y get deallocated
        return np.array(out.readX(0)), np.array(out.readY(1))

    def makeSpectrumFunction(self, i=0):
        """Form a definition string for the CrystalFieldSpectrum function
        @param i: Index of a spectrum.
        """
        if self.NumberOfSpectra == 1:
            return str(self.function)
        else:
            funs = self.function.createEquivalentFunctions()
            return str(funs[i])

    def update(self, func):
        """
        Update values of the fitting parameters.
        @param func: A IFunction object containing new parameter values.
        """
        self.function = func

    def plot(self, i=0, workspace=None, ws_index=0, name=None):
        """Plot a spectrum. Parameters are the same as in getSpectrum(...)"""
        from mantidplot import plotSpectrum
        from mantid.api import AlgorithmManager
        createWS = AlgorithmManager.createUnmanaged('CreateWorkspace')
        createWS.initialize()

        xArray, yArray = self.getSpectrum(i, workspace, ws_index)
        ws_name = name if name is not None else 'CrystalFieldMultiSite_%s' % self.Ions

        if isinstance(i, int):
            if workspace is None:
                if i > 0:
                    ws_name += '_%s' % i
                createWS.setProperty('DataX', xArray)
                createWS.setProperty('DataY', yArray)
                createWS.setProperty('OutputWorkspace', ws_name)
                createWS.execute()
                plot_window = self._plot_window[i] if i in self._plot_window else None
                self._plot_window[i] = plotSpectrum(ws_name, 0, window=plot_window, clearWindow=True)
            else:
                ws_name += '_%s' % workspace
                if i > 0:
                    ws_name += '_%s' % i
                createWS.setProperty('DataX', xArray)
                createWS.setProperty('DataY', yArray)
                createWS.setProperty('OutputWorkspace', ws_name)
                createWS.execute()
                plotSpectrum(ws_name, 0)
        else:
            ws_name += '_%s' % i
            createWS.setProperty('DataX', xArray)
            createWS.setProperty('DataY', yArray)
            createWS.setProperty('OutputWorkspace', ws_name)
            createWS.execute()
            plotSpectrum(ws_name, 0)

    @property
    def Ions(self):
        string_ions = self.function.getAttributeValue('Ions')
        string_ions = string_ions[1:-1]
        return string_ions.split(",")

    @Ions.setter
    def Ions(self, value):
        if isinstance(value, basestring):
            self.function.setAttributeValue('Ions', value)
        else:
            self.function.setAttributeValue('Ions', self.iterable_to_string(value))

    @property
    def Symmetries(self):
        string_symmetries = self.function.getAttributeValue('Symmetries')
        string_symmetries = string_symmetries[1:-1]
        return string_symmetries.split(",")

    @Symmetries.setter
    def Symmetries(self, value):
        if isinstance(value, basestring):
            self.function.setAttributeValue('Symmetries', value)
        else:
            self.function.setAttributeValue('Symmetries', self.iterable_to_string(value))

    @property
    def ToleranceEnergy(self):
        """Get energy tolerance"""
        return self.function.getAttributeValue('ToleranceEnergy')

    @ToleranceEnergy.setter
    def ToleranceEnergy(self, value):
        """Set energy tolerance"""
        self.function.setAttributeValue('ToleranceEnergy', float(value))

    @property
    def ToleranceIntensity(self):
        """Get intensity tolerance"""
        return self.function.getAttributeValue('ToleranceIntensity')

    @ToleranceIntensity.setter
    def ToleranceIntensity(self, value):
        """Set intensity tolerance"""
        self.function.setAttributeValue('ToleranceIntensity', float(value))

    @property
    def Temperatures(self):
        return list(self.function.getAttributeValue("Temperatures"))

    @Temperatures.setter
    def Temperatures(self, value):
        self.function.setAttributeValue('Temperatures', value)

    @property
    def FWHMs(self):
        fwhm = self.function.getAttributeValue('FWHMs')
        nDatasets = len(self.Temperatures)
        if len(fwhm) != nDatasets:
            return list(fwhm) * nDatasets
        return list(fwhm)

    @FWHMs.setter
    def FWHMs(self, value):
        if len(value) == 1:
            value = value[0]
            value = [value] * len(self.Temperatures)
        self.function.setAttributeValue('FWHMs', value)

    @property
    def FWHMVariation(self):
        return self.function.getAttributeValue('FWHMVariation')

    @FWHMVariation.setter
    def FWHMVariation(self, value):
        self.function.setAttributeValue('FWHMVariation', float(value))

    @property
    def FixAllPeaks(self):
        return self.function.getAttributeValue('FixAllPeaks')

    @FixAllPeaks.setter
    def FixAllPeaks(self, value):
        self.function.setAttributeValue('FixAllPeaks', value)

    @property
    def PeakShape(self):
        return self.function.getAttributeValue('PeakShape')

    @PeakShape.setter
    def PeakShape(self, value):
        self.function.setAttributeValue('PeakShape', value)

    @property
    def NumberOfSpectra(self):
        return self.function.getNumberDomains()

    @property
    def NPeaks(self):
        return self.function.getAttributeValue('NPeaks')

    @NPeaks.setter
    def NPeaks(self, value):
        self.function.setAttributeValue('NPeaks', value)

    def fix(self, *args):
        for a in args:
            self.function.fixParameter(a)

    def __getitem__(self, item):
        if self.function.hasAttribute(item):
            return self.function.getAttributeValue(item)
        else:
            return self.function.getParameterValue(item)


    def setBackground(self, *args, peak=None, other=None):

        if len(args) > 0:
            bg = FunctionFactory.createFunction(args[0])
            if isComposite(bg):
                f0 = bg[0]
                if isPeak(f0):
                    peak = f0
            return

        self._background = Function(self.function, prefix='bg.')
        if peak and other:
            self._background.peak = Function(self.function, prefix='bg.f0.')
            self._background.background = Function(self.function, prefix='bg.f1.')
            self.function.setAttributeValue('Background', '%s;%s' % (peak, other))
        elif peak:
            self.function.setAttributeValue('Background', '%s' % peak)
        elif other:
            self.function.setAttributeValue('Background', '%s' % other)
        else:
            raise RuntimeError('!!!')

    @property
    def background(self):
        return self._background