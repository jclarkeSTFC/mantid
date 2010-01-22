#########################################################
# This module contains utility functions common to the 
# SANS data reduction scripts
########################################################
from mantidsimple import *
import math

# Return the details specific to the instrument and bank given
# as:
#   dimension, specmin, specmax, backmon_start, backmon_end
def GetInstrumentDetails(instr_name, detbank):
	if instr_name == 'LOQ':
		if detbank == 'main-detector-bank':
			return 128, 3, 16386, 31000, 39000
		elif detbank == 'HAB':
			return 128, 16387, 17792, 31000, 39000
		else:
			return -1, -1, -1, -1, -1
	else:
		# This is the number of monitors before the first set of detectors 
		monstart = 8
		dim = 192
		if detbank == 'front-detector':
			smin = dim*dim + 1 + monstart
			smax = dim*dim*2 + monstart
			return dim, smin, smax, 85000, 100000
		elif detbank == 'rear-detector':
			smin = 1 + monstart
			smax = dim*dim + monstart
			return 192, smin, smax, 85000, 100000
		else:
			return -1, -1, -1, -1, -1
			
# Parse a log file containing run information and return the detector positions
def parseLogFile(logfile):
	logkeywords = {'Rear_Det_X':0.0, 'Rear_Det_Z':0.0, 'Front_Det_X':0.0, 'Front_Det_Z':0.0, \
		'Front_Det_Rot':0.0}
	if logfile == None:
		return tuple(logkeywords.values())
	file = open(logfile, 'rU')
	for line in file:
		entry = line.split()[1]
		if entry in logkeywords.keys():
			logkeywords[entry] = float(line.split()[2])
	
	return tuple(logkeywords.values())

def normalizePhi(phi):
	if phi > 90.0:
		phi -= 180.0
	elif phi < -90.0:
		phi += 180.0
	else:
		pass
	return phi

def InfinitePlaneXML(id, plane_pt, normal_pt):
	return '<infinite-plane id="' + str(id) + '">' + \
	    '<point-in-plane x="' + str(plane_pt[0]) + '" y="' + str(plane_pt[1]) + '" z="' + str(plane_pt[2]) + '" />' + \
	    '<normal-to-plane x="' + str(normal_pt[0]) + '" y="' + str(normal_pt[1]) + '" z="' + str(normal_pt[2]) + '" />'+ \
	    '</infinite-plane>\n'

def InfiniteCylinderXML(id, centre, radius, axis):
	return  '<infinite-cylinder id="' + str(id) + '">' + \
	'<centre x="' + str(centre[0]) + '" y="' + str(centre[1]) + '" z="' + str(centre[2]) + '" />' + \
	'<axis x="' + str(axis[0]) + '" y="' + str(axis[1]) + '" z="' + str(axis[2]) + '" />' + \
	'<radius val="' + str(radius) + '" />' + \
	'</infinite-cylinder>\n'

# Mask a cylinder, specifying the algebra to use
def MaskWithCylinder(workspace, radius, xcentre, ycentre, algebra):
    '''Mask a cylinder on the input workspace.'''
    xmldef = InfiniteCylinderXML('shape', [xcentre, ycentre, 0.0], radius, [0,0,1])
    xmldef += '<algebra val="' + algebra + 'shape" />'
    # Apply masking
    MaskDetectorsInShape(workspace,xmldef)

# Mask the inside of a cylinder
def MaskInsideCylinder(workspace, radius, xcentre = '0.0', ycentre = '0.0'):
    '''Mask out the inside of a cylinder or specified radius'''
    MaskWithCylinder(workspace, radius, xcentre, ycentre, '')

# Mask the outside of a cylinder
def MaskOutsideCylinder(workspace, radius, xcentre = '0.0', ycentre = '0.0'):
    '''Mask out the outside of a cylinder or specified radius'''
    MaskWithCylinder(workspace, radius, xcentre, ycentre, '#')

# Mask such that the remainder is that specified by the phi range
def LimitPhi(workspace, centre, phimin, phimax):
    #Convert to radians
    phimin = math.pi*phimin/180.0
    phimax = math.pi*phimax/180.0
    xmldef =  InfinitePlaneXML('pla',centre, [math.cos(phimin + math.pi/2.0),math.sin(phimin + math.pi/2.0),0]) + \
    InfinitePlaneXML('pla2',centre, [-math.cos(phimax + math.pi/2.0),-math.sin(phimax + math.pi/2.0),0]) + \
    InfinitePlaneXML('pla3',centre, [math.cos(phimax + math.pi/2.0),math.sin(phimax + math.pi/2.0),0]) + \
    InfinitePlaneXML('pla4',centre, [-math.cos(phimin + math.pi/2.0),-math.sin(phimin + math.pi/2.0),0]) + \
    '<algebra val="#((pla pla2):(pla3 pla4))" />'
    
    MaskDetectorsInShape(workspace, xmldef)	

# Essentially an enumeration
class Orientation(object):

    Horizontal = 1
    Vertical = 2
    Rotated = 3
    # This is for the empty instrument
    HorizontalFlipped = 4

# Work out the spectra IDs for block of detectors
def spectrumBlock(base, ylow, xlow, ydim, xdim, det_dimension, orientation):
    '''Compile a list of spectrum IDs for rectangular block of size xdim by ydim'''
    output = ''
    if orientation == Orientation.Horizontal:
        start_spec = base + ylow*det_dimension + xlow
        for y in range(0, ydim):
            for x in range(0, xdim):
                output += str(start_spec + x + (y*det_dimension)) + ','
    elif orientation == Orientation.Vertical:
        start_spec = base + xlow*det_dimension + ylow
        for x in range(det_dimension - 1, det_dimension - xdim-1,-1):
            for y in range(0, ydim):
    	        std_i = start_spec + y + ((det_dimension-x-1)*det_dimension)
		output += str(std_i ) + ','
    elif orientation == Orientation.Rotated:
        # This is the horizontal one rotated so need to map the xlow and vlow to their rotated versions
        start_spec = base + ylow*det_dimension + xlow
        max_spec = det_dimension*det_dimension + base - 1
        for y in range(0, ydim):
            for x in range(0, xdim):
                std_i = start_spec + x + (y*det_dimension)
                output += str(max_spec - (std_i - base)) + ','
    elif orientation == Orientation.HorizontalFlipped:
         start_spec = base + ylow*det_dimension + xlow
	 for y in range(0,ydim):
             max_row = base + (y+1)*det_dimension - 1
	     min_row = base + (y)*det_dimension
	     for x in range(0,xdim):
                 std_i = start_spec + x + (y*det_dimension)
		 diff_s = std_i - min_row
		 output += str(max_row - diff_s) + ','

    return output.rstrip(",")
    
# Convert a mask string to a spectra list
# 6/8/9 RKH attempt to add a box mask e.g.  h12+v34 (= one pixel at intersection), h10>h12+v101>v123 (=block 3 wide, 23 tall)
def ConvertToSpecList(maskstring, firstspec, dimension, orientation):
    '''Compile spectra ID list'''
    if maskstring == '':
        return ''
    masklist = maskstring.split(',')
    speclist = ''
    for x in masklist:
        x = x.lower()
        if '+' in x:
            bigPieces = x.split('+')
            if '>' in bigPieces[0]:
                pieces = bigPieces[0].split('>')
                low = int(pieces[0].lstrip('hv'))
                upp = int(pieces[1].lstrip('hv'))
            else:
                low = int(bigPieces[0].lstrip('hv'))
                upp = low
            if '>' in bigPieces[1]:
                pieces = bigPieces[1].split('>')
                low2 = int(pieces[0].lstrip('hv'))
                upp2 = int(pieces[1].lstrip('hv'))
            else:
                low2 = int(bigPieces[1].lstrip('hv'))
                upp2 = low2            
            if 'h' in bigPieces[0] and 'v' in bigPieces[1]:
                ydim=abs(upp-low)+1
                xdim=abs(upp2-low2)+1
                speclist += spectrumBlock(firstspec,low, low2,ydim, xdim, dimension,orientation) + ','
            elif 'v' in bigPieces[0] and 'h' in bigPieces[1]:
                xdim=abs(upp-low)+1
                ydim=abs(upp2-low2)+1
                speclist += spectrumBlock(firstspec,low2, low,nstrips, dimension, dimension,orientation)+ ','
            else:
                print "error in mask, ignored:  " + x
        elif '>' in x:
            pieces = x.split('>')
            low = int(pieces[0].lstrip('hvs'))
            upp = int(pieces[1].lstrip('hvs'))
            if 'h' in pieces[0]:
                nstrips = abs(upp - low) + 1
                speclist += spectrumBlock(firstspec,low, 0,nstrips, dimension, dimension,orientation)  + ','
            elif 'v' in pieces[0]:
                nstrips = abs(upp - low) + 1
                speclist += spectrumBlock(firstspec,0,low, dimension, nstrips, dimension,orientation)  + ','
            else:
                for i in range(low, upp + 1):
                    speclist += str(i) + ','
        elif 'h' in x:
            speclist += spectrumBlock(firstspec,int(x.lstrip('h')), 0,1, dimension, dimension,orientation) + ','
        elif 'v' in x:
            speclist += spectrumBlock(firstspec,0,int(x.lstrip('v')), dimension, 1, dimension,orientation) + ','
        else:
            speclist += x.lstrip('s') + ','
    
    return speclist

# Mask by detector number
def MaskBySpecNumber(workspace, speclist):
    speclist = speclist.rstrip(',')
    if speclist == '':
        return ''
    MaskDetectors(workspace, SpectraList = speclist)

# Mask by bin range
def MaskByBinRange(workspace, timemask):
	# timemask should be a ';' separated list of start/end values
	ranges = timemask.split(';')
	for r in ranges:
		limits = r.split()
		if len(limits) == 2:
			MaskBins(workspace, workspace, XMin= limits[0] ,XMax=limits[1])

# Setup the transmission workspace
def SetupTransmissionWorkspace(inputWS, spec_list, backmon_start, backmon_end, wavbining, loqremovebins):
	tmpWS = inputWS + '_tmp'
	if loqremovebins == True:
		RemoveBins(inputWS,tmpWS, 19900, 20500, Interpolation='Linear')
		FlatBackground(tmpWS, tmpWS, StartX = backmon_start, EndX = backmon_end, WorkspaceIndexList = spec_list)
	else:
		FlatBackground(inputWS, tmpWS, StartX = backmon_start, EndX = backmon_end, WorkspaceIndexList = spec_list)
	# Convert and rebin
	ConvertUnits(tmpWS,tmpWS,"Wavelength")
	Rebin(tmpWS, tmpWS, wavbining)
	return tmpWS

# Correct of for the volume of the sample/can. Dimensions should be in order: width, height, thickness
def ScaleByVolume(inputWS, scalefactor, geomid, width, height, thickness):
	correction = scalefactor
	# Divide by the area
	if geomid == 1:
		# Volume = circle area * height
		# Factor of four comes from radius = width/2
		correction /= (height*math.pi*math.pow(width,2)/4.0)
	elif geomid == 2:
		correction /= (width*height*thickness)
	else:
		# Factor of four comes from radius = width/2
		correction /= (thickness*math.pi*math.pow(width, 2)/4.0)
	
	CreateSingleValuedWorkspace("scalar",str(correction),"0.0")
	Multiply(inputWS, "scalar", inputWS)
	mantid.deleteWorkspace("scalar")

def InfinitePlaneXML(id, planept, normalpt):
	return  '<infinite-plane id="' + str(id) + '">' + \
	    '<point-in-plane x="' + str(planept[0]) + '" y="' + str(planept[1]) + '" z="' + str(planept[2]) + '" />' + \
	    '<normal-to-plane x="' +str(normalpt[0]) + '" y="' + str(normalpt[1]) + '" z="' + str(normalpt[2]) + '" />' + \
	    '</infinite-plane>'
								     
def QuadrantXML(centre,rmin,rmax,quadrant):
	cin_id = 'cyl-in'
	xmlstring = InfiniteCylinderXML(cin_id, centre, rmin, [0,0,1])
	cout_id = 'cyl-out'
	xmlstring+= InfiniteCylinderXML(cout_id, centre, rmax, [0,0,1])
	plane1Axis=None
	plane2Axis=None
	if quadrant == 'Left':
		plane1Axis = [-1,1,0]
		plane2Axis = [-1,-1,0]
	elif quadrant == 'Right':
		plane1Axis = [1,-1,0]
		plane2Axis = [1,1,0]
	elif quadrant == 'Up':
		plane1Axis = [1,1,0]
		plane2Axis = [-1,1,0]
	elif quadrant == 'Down':
		plane1Axis = [-1,-1,0]
		plane2Axis = [1,-1,0]
	else:
		return ''
	p1id = 'pl-a'
	xmlstring += InfinitePlaneXML(p1id, centre, plane1Axis)
	p2id = 'pl-b'
	xmlstring += InfinitePlaneXML(p2id, centre, plane2Axis)
	xmlstring += '<algebra val="(#((#(' + cout_id + ':(#' + cin_id  + '))) ' + p1id + ' ' + p2id + '))"/>\n' 
	return xmlstring

def StripEndZeroes(workspace, flag_value = 0.0):
        result_ws = mantid.getMatrixWorkspace(workspace)
        y_vals = result_ws.readY(0)
        length = len(y_vals)
        # Find the first non-zero value
        start = 0
        for i in range(0, length):
                if ( y_vals[i] != flag_value ):
                        start = i
                        break
        # Now find the last non-zero value
        stop = 0
        length -= 1
        for j in range(length, 0,-1):
                if ( y_vals[j] != flag_value ):
                        stop = j
                        break
        # Find the appropriate X values and call CropWorkspace
        x_vals = result_ws.readX(0)
        startX = x_vals[start]
        # Make sure we're inside the bin that we want to crop
        endX = 1.001*x_vals[stop + 1]
        CropWorkspace(workspace,workspace,startX,endX)

##
# A small class to collect together run information, used to save passing tuples around
##
class RunDetails(object):

	def __init__(self, raw_ws, final_ws, trans_raw, direct_raw, maskpt_rmin, maskpt_rmax, suffix):
		self._rawworkspace = raw_ws
		self._finalws = final_ws
		self._trans_raw = trans_raw
		self._direct_raw = direct_raw
		self._maskrmin = maskpt_rmin
		self._maskrmax = maskpt_rmax
		self._suffix = suffix

	def getRawWorkspace(self):
		return self._rawworkspace

	def getReducedWorkspace(self):
		return self._finalws

	def setReducedWorkspace(self, final_ws):
		self._finalws = final_ws

	def getTransRaw(self):
		return self._trans_raw

	def getDirectRaw(self):
		return self._direct_raw

 	def getMaskPtMin(self):
		return self._maskrmin

 	def setMaskPtMin(self, rmin):
		self._maskrmin = rmin
		
 	def getMaskPtMax(self):
		return self._maskrmax

 	def setMaskPtMax(self, rmax):
		self._maskrmax = rmax
	
	def getSuffix(self):
		return self._suffix
		
		
