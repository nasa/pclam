'''
Copyright 2023 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.

This software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only: Python, https://docs.python.org/3/license.html; numpy, https://numpy.org/doc/stable/license.html; numba, https://github.com/numba/numba/blob/main/LICENSE; pytecplot, https://pypi.org/project/pytecplot/; https://projects.scipy.org/scipylib/license.html.

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
'''

import datetime
import numpy as np
from os import path, makedirs
from . import io as LL_io
from . import calc as LL_calc
from . import util as LL_util


def run_pclam(input_file, data_name, print_file=False, base_name=None, debug=False): # run_pclam
    '''
    FUNCTION TO RUN STANDARD LINELOAD SINGLE LINELOAD CALCULATION PROCEDURE
     arguments: [input_file](str) input .json file name ('' prompts for all defaults)
                [data_name](str) data file name
                [print_file](Bool) if true, outputs a sample input file (default False)
                [base_name](str) run name (default None takes base_name from Config)
                [debug](Bool) flag for debug outputs (default False)
       returns: [lineloads](np array) 3D array of lineloads
                [fandm](np array) 2D array of integrated f&m
                [config](object) object
        author: thomas.j.wignall@nasa.gov
    '''
    config = LL_io.get_input(input_file, print_file)
    
    if base_name is not None:
        config.base_name = base_name
    else:
        config.base_name = data_name.split('/')[-1].split('.')[0]
    
    if debug: print(str(datetime.datetime.now())+' - loading')
    
    data_by_node, data_by_elem, connectivity = LL_io.load_surf_data(data_name, config)
    
    if config.absolute_pressure_integration == True:
        adjust_cp = config.pinf_by_qref
    else:
        adjust_cp = 0
    
    data_by_node = LL_util.call_adjust_cp(data_by_node, adjust_cp)
    lineloads = make_lineload(config, data_by_node, connectivity, data_by_elem, debug)
    LL_util.call_write_lineloads(lineloads, config)
    fandm = LL_util.get_and_write_fandm(lineloads, config)
    
    return lineloads, fandm, config
#end run_pclam

def make_lineload(config, nodeData, nodeConn, elemData=None, debug=False):                    # make_lineload
    '''
    FUNCTION TO COMPUTE AND SAVE COMPONENTWISE LINELOADS GIVEN SURFACE CFD DATA
     arguments: [config](object) object
                [nodeData](np array) 2D array of node data
                [nodeConn](np array) 2D array of node connectivity
                [debug](Boolean) flag for debug outputs
       returns: [lineloads](np array) 3D array of lineloads [x-stns,f&m cmps,variables[p,fx,fy,fz]]
                [fandm](np array) 2D array of integrated f&m [p,fx,fy,fz,all][FX,FY,FZ,MX,MY,MZ]
        author: thomas.j.wignall@nasa.gov
                michael.w.lee@nasa.gov
    '''
    
    # ensure data in is in the right format
    if debug: print(str(datetime.datetime.now())+' - organizing')

    nodeData, elemData, config = LL_util.pad_data(nodeData, elemData, config)
    if nodeData is None:
        print('nodeData is None')
        return None
    
    # compute element-centered info
    elemData = LL_util.node_to_element(nodeData, nodeConn, elemData, debug)
    
    # configure user-defined lineload axes
    match config.axis.lower():
        case 'x':
            llIndex = 0
        case 'y':
            llIndex = 1
        case 'z':
            llIndex = 2
        case _:
            print('lineload axis '+config.axis+' not recognized - defaulting to x')
            llIndex = 0
    
    match config.profile_axis.lower():
        case 'x':
            profileIndex = 0
        case 'y':
            profileIndex = 1
        case 'z':
            profileIndex = 2
        case _:
            print('profile axis '+config.profile_axis+' not recognized - defaulting to z')
            profileIndex = 2

    # prepare for lineload integration
    nLLpoints = config.nll_points
    data_max = nodeData[:,llIndex].max() 
    data_min = nodeData[:,llIndex].min()
    data_range = data_max - data_min
    if config.bin_edges_on_minmax:
        db = (data_range) / nLLpoints
        offset = 0
    else:
        db = data_range/ (nLLpoints - 1)
        offset = db/2
    eps = data_range * 1e-8
    
    # if config.station_file == "0":
    bins = np.linspace(data_min-offset-eps, data_max+offset+eps, num=nLLpoints+1)
    '''
    else:
        bins = LL_io.read_stations(config)
        db = bins[1]-bins[0]
        if bins is None:
            print('station file incompatible - exiting')
            return None
    '''

    # reload existing grid reference data if wanted and possible
    doCompute = True
    if config.use_bin_mapping_file:
        if debug: print(str(datetime.datetime.now())+' - loading grid weights')
        binAreas, binWeights = LL_io.load_bin_weights(config)
        if binAreas is not None:
            if binWeights.shape != (elemData.shape[0],nLLpoints,3):
                if debug: print('Loaded bin weights incompatible with surface data. Recomputing.')
            else:
                doCompute = False
                if debug: print(str(datetime.datetime.now())+' - computing from reference grid')
                binMeans = LL_util.reget_elem_bin_means(elemData, binWeights)
    
    # bin data if necessary
    if doCompute:
        if debug: print(str(datetime.datetime.now())+' - element areas')
        elemAreas = LL_calc.triangle_areas(elemData[:,0,:3],elemData[:,1,:3],elemData[:,2,:3])
        binAreas, binMeans, binWeights = LL_util.get_elem_bin_means(elemData,elemAreas,bins,db,llIndex,debug)
        if config.use_bin_mapping_file:
            if debug: print(str(datetime.datetime.now())+' - saving weights')
            LL_io.save_bin_weights(config, binAreas, binWeights)
    
    # compute lineloads
    '''
    lineloadComponent array structure
    [ fx, fy, fz, mx, my, mz ][ p, cfx, cfy, cfz ]
    '''
    if debug: print(str(datetime.datetime.now())+' - integrating')
    mrp = np.array(config.mrp)
    lineloadComponents, profileVals = LL_calc.calc_lineload(mrp,nLLpoints,binMeans,binAreas,profileIndex)
    
    # clean data
    if debug: print(str(datetime.datetime.now())+' - saving')
    lineloadComponents = lineloadComponents / config.Sref # account for integrated area
    lineloadComponents = lineloadComponents / db * config.Lref # normalize by bin size
    lineloadComponents[:,3:,:] = lineloadComponents[:,3:,:] / config.Lref # account for moment arm
    bins = bins[1:] - db/2 # center bin locations instead of setting at extremities
    lineloadComponents = np.dstack((lineloadComponents,np.sum(lineloadComponents,axis=2)))
    
    # output lineloads
    '''
    lineload array structure
    [ axis/L  fx  fy  fz  mx  my  mz  profile/L  cumsum(fx  fy  fz  mx  my  mz)*db/L ]
    '''
    lineloads = np.zeros((nLLpoints, 14, 5))
    for i in range(5):
        lineloads[:,0,i] = bins #/ config.Lref
        lineloads[:,7,i] = profileVals #/ config.Lref
    lineloads[:,1:7,:] = lineloadComponents[:,:,:]
    lineloads[:,8:,:] = np.cumsum(lineloads[:,1:7,:], axis=0)*db/config.Lref

    return lineloads
#end make_lineload





if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 6:
        print('PCLAM.MAIN DIRECT CALL USAGE')
        print('   pclam/main.py [ data file ] [ input file   ] [ make input file flag ] [ base name  ] [ debug flag ]')
        print('      (defaults) ( required  ) ( pclam.Config ) ( False                ) ( input file ) ( False      )')
        sys.exit()

    run_name = sys.argv[1]

    match len(sys.argv):
        case 2:
            input_file = ''
            print_file = False
            base_name = None
            debug = False
        case 3:
            input_file = sys.argv[2]
            print_file = False
            base_name = None
            debug = False
        case 4 | 5:
            input_file = sys.argv[2]
            print_file = sys.argv[3] == 'True' or sys.argv[3] == 'true'
            base_name = None
            debug = False
        case 5:
            input_file = sys.argv[2]
            print_file = sys.argv[3] == 'True' or sys.argv[3] == 'true'
            base_name = sys.argv[4]
            debug = False
        case _:
            input_file = sys.argv[2]
            print_file = sys.argv[3] == 'True' or sys.argv[3] == 'true'
            base_name = sys.argv[4]
            debug = sys.argv[5] == 'True' or sys.argv[5] == 'true'
        
    _,_,_ = run_pclam(input_file, base_name, print_file=print_file, base_name=base_name, debug=debug)

