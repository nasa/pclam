import datetime
import os
import numpy as np
from . import io as LL_io
from . import calc as LL_calc

def get_elem_bin_means(elemData, elemAreas, bins, db, llIndex, debug=False):        # get_elem_bin_means
    '''
    FUNCTION TO COMPUTE BIN-ORGANIZED ELEMENT SUB-AREAS AND VALUE MEANS
     arguments: [elemData](np array) 3D array of element-organized nodal data
                                        [nElem, 3 nodes, [x,y,z,px,py,pz,fx,fy,fz]]
                [elemAreas](np array) 1D array of element areas
                [bins](np array) 1D vector of int-dir-stations for bins, increasing
                [db](scalar) uniform distance between bin stations
                [llIndex](int) column index down which the bins are defined
                [debug](opt, Bool) flag to write log of bin decompositions
       returns: [binAreas](np array) 2D array of per-element, per-bin areas
                                                [elem, area in bin]
                [binMeans](np array) 3D array of sub-element variable values (numElem, bins, xyz+6 variables)
                                                [elem, bin, mean value]
                [binWeights](np array) 3D array of bin weights for mean recreation
        author: michael.w.lee@nasa.gov, thomas.j.wignall@nasa.gov
    last write: 05/2022
    '''
    
    if debug:
        print(str(datetime.datetime.now())+' - binning')
        fid = open('debug.log','w')

    nElem = elemData.shape[0]
    nBins = len(bins)-1
    nPts = elemData.shape[2] + 3
    
    fillWeights = np.zeros((nElem,3,3))
    fillWeights[:,0,0] = 1
    fillWeights[:,1,1] = 1
    fillWeights[:,2,2] = 1
    elemData = np.dstack((elemData,fillWeights))

    binAreas = np.zeros((nElem, nBins))
    unprocessedElems = np.zeros(nElem) == 0
    binMeans = np.zeros((nElem, nBins, nPts))
    
    nodesInBin = np.zeros((nElem,3)) == 1
    procElems = np.zeros(nElem) == 1
    for i, b in enumerate(bins[1:]):
        
        # find unprocessed element indices that have any nodes in current bin
        elemsIn = (unprocessedElems) & (np.any(elemData[:,:,llIndex] < b, axis=1))
        unprocessedElems[elemsIn] = False

        if not np.any(elemsIn):
            continue
        if debug: fid.write(str(elemsIn.sum())+' elems in bin '+str(i)+'\n')

        # find which nodes are in this bin
        nodesInBin[elemsIn,:] = elemData[elemsIn,:,llIndex] <= b

        # process elements that have three nodes in this bin
        procElems[:] = False
        procElems[elemsIn] = np.all(nodesInBin[elemsIn,:], axis=1)
        pElem = np.sum(procElems)
        if pElem > 0:
            binAreas, binMeans = LL_calc.three_in_one_elems(elemData,elemAreas,procElems,binAreas,binMeans,i)
        if debug: fid.write('    '+str(pElem)+' 3-nodes\n')

        elemsIn[procElems] = False
        if not np.any(elemsIn):
            continue
        
        # process elements that have only two nodes in this bin
        procElems[:] = False
        procElems[elemsIn] = np.sum(nodesInBin[elemsIn,:], axis=1) == 2
        pElem = np.sum(procElems)
        if pElem > 0:
            binAreas, binMeans = LL_calc.two_in_one_elems(elemData,elemAreas,procElems,binAreas,binMeans,\
                                                          b,db,llIndex,nodesInBin,i)
            if debug: fid.write('    '+str(pElem)+' 2+1-nodes\n')
            
            elemsIn[procElems] = False
            if not np.any(elemsIn):
                continue
        
        # process elements that have only one node in this bin
        pElem = np.sum(elemsIn)
        binAreas, binMeans, num12nodes, num111nodes = LL_calc.one_in_one_elems(elemData, elemAreas, elemsIn,
                                                                        binAreas, binMeans,
                                                                        b, db, llIndex, nodesInBin, i)
        if num12nodes > 0 and debug: fid.write('    '+str(num12nodes) +' 1+2-nodes\n')
        if num111nodes> 0 and debug: fid.write('    '+str(num111nodes)+' 1+1+1-nodes\n')
    
    if np.any(np.invert(np.isclose(np.sum(binAreas,axis=1), elemAreas))) \
            or np.any(unprocessedElems):
        header = ','.join(str('bin1 '+str(np.arange(binAreas.shape[1]-1,dtype=int)+2)[1:-1]+\
                          ' trueArea sumArea error').split())
        np.savetxt('binAreas.csv',np.hstack((binAreas,elemAreas[None].T,
                                             np.sum(binAreas,axis=1)[None].T,
                                             (elemAreas-np.sum(binAreas,axis=1))[None].T)),
                                             header=header,delimiter=',')
        raise Exception('ERROR IN BIN AREA DECOMPOSITION - see binAreas.csv')
    
    binWeights = binMeans[:,:,nPts-3:]
    binMeans = binMeans[:,:,:nPts-3]

    if debug:
        fid.close()
    
    return binAreas, binMeans, binWeights
#end get_elem_bin_means

def reget_elem_bin_means(elemData, elemWeights):                               # reget_elem_bin_means
    '''
    FUNCTION TO COMPUTE ELEM BIN MEANS GIVEN PRECOMPUTED ELEMENT INTERPOLATION WEIGHTS
     arguments: [elemData](np array) 3D array of element data organized by nodes (numElem,3 nodes,9 vars)
                                        data order per-node: [x, y, z, px, py, pz, fx, fy, fz]
                [elemWeights](np array) 3D array of element node weights [elem, bin, node weights]
       returns: [binMeans](np array) 3D array of sub-element variable values (numElem, bins, xyz+6 variables)
                                                [elem, bin, mean value]
        author: michael.w.lee@nasa.gov
    last write: 10/2021
    '''

    numElem, numBins, _ = elemWeights.shape

    elemData = np.swapaxes(elemData, 1, 2)
    elemWeights = np.swapaxes(elemWeights, 1, 2)
    binMeans = np.swapaxes(elemData @ elemWeights, 1, 2)

    return binMeans
#end reget_elem_bin_means

def node_to_element(nodeData, nodeConn, elemData=None, debug=False):                            # node_to_element
    '''
    FUNCTION TO CONVERT NODE-CENTERED DATA TO ELEMENT-CENTERED DATA
    ASSUMES NODE DATA STORED IN COLUMNS: [x, y, z, px, py, pz, fx, fy, fz]
    ALSO COMPUTES CELL NORMALS AND DECOMPOSES PRESSURE INTO X, Y, Z COMPONENTS
     arguments: [nodeData](np array) 2D array of node data
                [nodeConn](np array) 2D array of node connectivity
                [elemData](np array) 3D array of element data organzied by nodes.
                    Optional argument incase element data is already available
                [debug](Boolean) debug output flag
       returns: [elemData](np array) 3D array of element data organized by nodes (numElem,3 nodes,9 vars)
                                        data order per-node: [x, y, z, px, py, pz, fx, fy, fz]
        author: thomas.j.wignall@nasa.gov
    '''
    
    # move data to element centers
    if debug: print(str(datetime.datetime.now())+'    - connecting')
    if elemData is None:
        nElem = nodeConn.shape[0]
        temp = nodeData[nodeConn[:, :3], :]
        elemData = np.zeros((temp.shape[0], 3, 9))
        elemData[:, :, (0, 1, 2, 3, 6, 7, 8)] = temp
    else:
        elemData2 = np.zeros((elemData.shape[0], 3, 9))
        elemData2[:, :, (0, 1, 2, 3, 6, 7, 8)] = elemData
        elemData = elemData2
    
    # compute element inward-facing normals
    if debug: print(str(datetime.datetime.now())+'    - normals')
    elemNormals = LL_calc.element_normals(elemData)
    
    # decompose pressure into x,y,z components
    if debug: print(str(datetime.datetime.now())+'    - pressure')
    for i in range(3):
        elemData[:,i,3:6] = (elemData[:,i,3] * elemNormals.T).T

    return elemData

def call_adjust_cp(node_data, pinf):
    node_data[:,3] = node_data[:,3] + pinf
    return node_data

def pad_data(nodeData, elemData, config):
    if len(nodeData.shape) != 2:
        print("nodeData wrong shape")
        return None, None, config

    newNodeData = np.zeros((nodeData.shape[0],7))
    newNodeData[:, :3] = nodeData[:, :3]
    if elemData is not None:
        newElemData = np.zeros((elemData.shape[0], 3, 7))
        newElemData[:, :, :3] = elemData[:, :, :3]
    else:
        newElemData = None
    
    if nodeData.shape[1] == 4: # only pressure data given
        newNodeData[:, 3] = nodeData[:, 3]
        if elemData is not None:
            newElemData[:, :, 3] = elemData[:, :, 3]
        max_variables = ['all', 'cp']
    elif nodeData.shape[1] == 6:  # only shear data given
        newNodeData[:, 4:] = nodeData[:,3:]
        if elemData is not None:
            newElemData[:, :, 4:] = elemData[:, :, 3:]
        max_variables = ['all', 'cfx', 'cfy', 'cfz']
    elif nodeData.shape[1] == 7:
        newNodeData[:, 3:] = nodeData[:,3:]
        if elemData is not None:
            newElemData[:, :, 3:] = elemData[:, :, 3:]
        max_variables = ['all', 'cp', 'cfx', 'cfy', 'cfz']
    else: # making sure the only alternative is pressure and shear given
        return None, None, config

    new_variables = []
    for var in max_variables:
        if var in config.variables_saved:
            new_variables.append(var)
    config.variables_saved = new_variables
    
    return newNodeData, newElemData, config

def call_write_lineloads(lineloads, config):
    os.makedirs(config.output_dir, exist_ok=True)
    basename = os.path.join(config.output_dir, config.base_name+'_')
    var_index = {'cp':0, 'cfx':1, 'cfy':2, 'cfz':3, 'all':4}
    for var in config.variables_saved:
        i = var_index[var]
        outFile = basename + var + '_LL' + config.output_type
        LL_io.write_lineload(lineloads[:,:,i], outFile, config.axis.lower(), config.profile_axis.lower())    

def get_and_write_fandm(lineloads, config):
    dxOverL = (lineloads[1,0,-1] - lineloads[0,0,-1])/config.Lref # hardcoded for constant bin distribution
    fandm = np.zeros((5,6))
    os.makedirs(config.output_dir, exist_ok=True)
    basename = os.path.join(config.output_dir,config.base_name+'_')
    var_index = {'cp':0, 'cfx':1, 'cfy':2, 'cfz':3, 'all':4}
    for  var in config.variables_saved:
        i = var_index[var]
        fandm[i,:] = np.sum(lineloads[:,1:7,i]*dxOverL,axis=0)
        LL_io.write_fandm(fandm[i,:], basename + var + '_fandm.csv')

    return fandm
