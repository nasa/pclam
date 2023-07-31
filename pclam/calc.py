import numpy as np

try:
    from numba import jit_module
    #error(wrench)
    jit = True
except:
    # dummy function that allows numba to be optional
    # also prints out info telling user about numba
    jit = False
    def jit_module(nopython=True): 
        print('INFO: installation of the numba python module greatly improves computation speed')

def calc_lineload(mrp,nLLpoints,binMeans,binAreas, profileIndex):              # calc_lineload
    '''
    FUNCTION TO CALCULATE THE PER-VARIABLE LINELOADS GIVEN THE ELEMENT DATA
     arguments: [mrp](np array) 1D list of moment reference position [x,y,z]
                [nLLpoints](int) number of lineload points (number of bins)
                [binMeans](np array) 3D array of mean element data [numElem][numBins][x,y,z,px,py,pz,fx,fy,fz]
                [binAreas](np array) 2D array of bined element areas [numElem][numBins]
                [profileIndex](int) index that defines the reference profile
       returns: [lineloads](np array) 3D array of per-bin lineload components [fx,fy,fz,mx,my,mz][p,f]
                [profileVals](np array) 1D array of visually convenient OML widths in LL-direction
        author: michael.w.lee@nasa.gov
    last write: 03/2021
    '''
    
    profileVals = np.zeros(binMeans.shape[1])
    for i in range(binMeans.shape[1]):
        profileVals[i] = np.max(np.abs(binMeans[:,i,profileIndex]))

    lineloads = np.zeros((nLLpoints, 6, 4))  # 6 force/moment components, separate pressure/shear component info

    for b in range(nLLpoints):
        
        bindices = binAreas[:,b] != 0

        lineloads[b,:3,0] = np.sum(binMeans[bindices,b,3:6].T * binAreas[bindices,b], axis=1)
        lineloads[b,0,1]  = np.sum(binMeans[bindices,b,6].T   * binAreas[bindices,b])
        lineloads[b,1,2]  = np.sum(binMeans[bindices,b,7].T   * binAreas[bindices,b])
        lineloads[b,2,3]  = np.sum(binMeans[bindices,b,8].T   * binAreas[bindices,b])

        momArms = binMeans[bindices,b,:3] - mrp
        momTemplate = binMeans[bindices,b,3:6]
        lineloads[b,3:,0] = np.sum(np.cross(momArms,momTemplate).T * binAreas[bindices,b], axis=1)
        for i in range(3):
            momTemplate[:,:] = 0
            momTemplate[:,i] = binMeans[bindices,b,i+6]
            lineloads[b,3:,i+1] = np.sum(np.cross(momArms,momTemplate).T * binAreas[bindices,b], axis=1)

    return lineloads, profileVals

def organize_nodes(nodesInBin):                                                # organize_nodes
    '''
    FUNCTION TO ORGANIZE NODE NUMBERS BASED ON WHICH NODE IS ABOVE/BELOW THE BIN BOUNDARY
     arguments: [nodesInBin](np array) 2D array with each row being two Trues and one False
       returns: [falseNodes](np array) indices for which node has False in each row
                [oneTrueNodes](np array indices for one of the two True nodes in each row
                [twoTrueNodes](np array) indices for the other True node in each row
        author: michael.w.lee@nasa.gov
    last write: 05/2022
    '''
    
    nElem = nodesInBin.shape[0]
    falseNodes = np.zeros(nElem,dtype=np.intc)
    oneTrueNodes = np.zeros(nElem,dtype=np.intc)+1
    twoTrueNodes = np.zeros(nElem,dtype=np.intc)+2
    
    indices = ~nodesInBin[:,1]
    falseNodes[indices] = 1
    oneTrueNodes[indices] = 0

    indices = ~nodesInBin[:,2]
    falseNodes[indices] = 2
    twoTrueNodes[indices] = 0
    
    return falseNodes, oneTrueNodes, twoTrueNodes
#end organize_nodes

def interp_to_bin_bounds(bottomNodes, topNodes, b, db, llIndex):
    '''
    FUNCTION TO INTERPOLATE VALUES AT BIN BOUNDS BETWEEN THE BOTTOM NODES TO THE TOP NODES
     arguments: [bottomNodes](np array) 2D array of values at the bottom nodes [nElem,nPts]
                [topNodes](np array) 2D array of values at the top nodes [nElem,nPts]
                [b](np array) values for each element of first upper bin boundary
                [db](float) constant bin width
                [llIndex](int) lineload axis (x,y,z) = (0,1,2)
       returns: [interpData](np array) 3D array of interpolated values [nElem,maxInterpPts,nPts]
                [nInterpPts](np.array) array of the number of interpolation points
        author: michael.w.lee@nasa.gov
    last write: 05/2022
    '''

    (nElem, nPts) = topNodes.shape[:2]

    # build vectors along lineload axis for interpolation points of interest
    binVectors = [np.arange(1,dtype=np.float64)]*nElem
    nInterpPts = [0]*nElem
    for i in range(nElem):
        #top = topNodes[i,llIndex]  #for numba this is being seen as an array as opposed to just a float?????
        binVectors[i] = np.arange(b[i], topNodes[i,llIndex], db, dtype=np.float64)
        if len(binVectors[i]) == 0:
            continue
        nInterpPts[i] = len(binVectors[i])
    maxInterpPts = max(nInterpPts)

    # perform interpolations
    interpData = np.zeros((nElem,maxInterpPts+2,nPts))
    d = topNodes[:,:3] - bottomNodes[:,:3]
    dNorms = axis_1_norm_2D(d)
    for i in range(nElem):
        interpData[i,0,:] = bottomNodes[i,:]
        interpData[i,nInterpPts[i]+1:,:] = topNodes[i,:]
        
        if len(binVectors[i]) == 0:
            continue
        # p is shape [nInterpPts,3]
        p = d[i,:]*(binVectors[i].reshape((-1,1))-bottomNodes[i,llIndex])/d[i,llIndex] + bottomNodes[i,:3]
        interpData[i,1:nInterpPts[i]+1,:] = (topNodes[i,:]-bottomNodes[i,:])\
                                            *axis_1_norm_2D(p-bottomNodes[i,:3]).reshape((-1,1))/dNorms[i]\
                                            +bottomNodes[i,:]
        
    return interpData, np.array(nInterpPts)
#end interpolate_to_bin_boundaries

def trap_areas(topPoints, botPoints, llIndex):
    '''
    FUNCTION TO COMPUTE TRAPEZOID AREAS GIVEN TWO VECTORS DEFINING THE PARALLEL SIDES
     arguments: [topPoints](np array) if trapezoids are running left to right, this is top line
                                        [nElem,maxInterpPts,3(xyz)]
                [botPoints](np array) if trapezoids are running left to right, this is bottom line
                                        [nElem,maxInterpPts,3(xyz)]
                [llIndex](int) lineloads axis index
       returns: [trapAreas](np array) trapezoid areas
                                        [nElem,maxInterpPts-1]
        author: michael.w.lee@nasa.gov
    last write: 05/2022
    '''

    h = np.abs(topPoints[:,1:,llIndex] - topPoints[:,:-1,llIndex])
    b1 = axis_1_norm_3D(np.transpose(topPoints[:,:-1,:]-botPoints[:,:-1,:],(0,2,1)))
    b2 = axis_1_norm_3D(np.transpose(topPoints[:,1:,:]-botPoints[:,1:,:],(0,2,1)))
    
    trapAreas = 0.5 * h * (b1 + b2)
    return trapAreas
#end trap_areas

def trap_areas_2D(topPoints, botPoints, llIndex):
    '''
    FUNCTION TO COMPUTE TRAPEZOID AREAS FOR ONE ELEMENT GIVEN TWO VECTORS DEFINING THE PARALLEL SIDES
     arguments: [topPoints](np array) if trapezoids are running left to right, this is top line
                                        [maxInterpPts,3(xyz)]
                [botPoints](np array) if trapezoids are running left to right, this is bottom line
                                        [maxInterpPts,3(xyz)]
                [llIndex](int) lineloads axis index
       returns: [trapAreas](np array) trapezoid areas
                                        [maxInterpPts-1]
        author: michael.w.lee@nasa.gov
    last write: 03/2023
    '''

    h = np.abs(topPoints[1:,llIndex] - topPoints[:-1,llIndex])
    b1 = axis_1_norm_2D(topPoints[:-1,:]-botPoints[:-1,:])
    b2 = axis_1_norm_2D(topPoints[1:,:]-botPoints[1:,:])
    
    trapAreas = 0.5 * h * (b1 + b2)
    return trapAreas
#end trap_areas_2D

def triangle_areas(p1,p2,p3):                                                  # triangle_areas
    '''
    FUNCTION TO COMPUTE THE AREAS OF TRIANGLES USING HERON'S FORMULA
     arguments: [p1](np array) 2D array for point 1 [nElem,3(xyz)]
                [p2](np array) 2D array for point 2
                [p3](np array) 2D array for point 3
       returns: [areas](np array) 1D array of areas of triangles
        author: michael.w.lee@nasa.gov
    last write: 10/2020
    '''

    u = p2 - p1
    v = p3 - p1
    w = p3 - p2
    # using manual norms instead of np.linalg.norm due to jit compatability
    u = axis_1_norm_2D(u)
    v = axis_1_norm_2D(v)
    w = axis_1_norm_2D(w)
    halfPerim = (u+v+w) / 2
    areas = np.sqrt(halfPerim * (halfPerim-u) * (halfPerim-v) * (halfPerim-w))

    return areas
#end triangle_areas

def axis_1_norm_2D(a):                                                         # axis_1_norm_2D
    '''
    NUMBA-COMPATIBLE FUNCTION TO REPLACE np.linalg.norm(a,axis=1) FOR 2D ARRAY
     arguments: [a](np array) 2D array of interest
       returns: [aNorm](np array) array, normed over axis 1
        author: michael.w.lee@nasa.gov
    last write: 10/2020
    '''

    I = a.shape[0]
    aNorm = np.zeros(I)
    for i in range(I):
        aNorm[i] = np.sqrt(np.sum(a[i,:]**2))

    return aNorm
#end axis_1_norm_2D

def axis_1_norm_3D(a):                                                         # axis_1_norm_3D
    '''
    NUMBA-COMPATIBLE FUNCTION TO REPLACE np.linalg.norm(a,axis=1) FOR 3D ARRAY
     arguments: [a](np array) 3D array of interest
       returns: [aNorm](np array) array, normed over axis 1
        author: michael.w.lee@nasa.gov
    last write: 05/2022
    '''

    I,_,J = a.shape
    aNorm = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            aNorm[i,j] = np.sqrt(np.sum(a[i,:,j]**2))

    return aNorm
#end axis_1_norm_3D

def axis_1_mean_2D(a):                                                            # axis_1_mean_2D
    '''
    NUMBA-COMPATIBLE FUNCTION TO REPLACE np.mean(a,axis=0) FOR 2D ARRAY
     arguments: [a](numpy array) 2D array of interest
       returns: [aMean](np array) array, meaned over axis 0
        author: michael.w.lee@nasa.gov
    last write: 03/2023
    '''
     
    I = a.shape[0]
    aMean = np.zeros(I)
    for i in range(I):
        aMean[i] = np.mean(a[i,:])

    return aMean
#end axis_1_mean_2D

def axis_1_mean_3D(a):                                                            # axis_1_mean_3D
    '''
    NUMBA-COMPATIBLE FUNCTION TO REPLACE np.mean(a,axis=1) FOR 3D ARRAY
     arguments: [a](numpy array) 3D array of interest
       returns: [aMean](np array) array, meaned over axis 1
        author: michael.w.lee@nasa.gov
    last write: 06/2022
    '''

    I,_,J = a.shape
    aMean = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            aMean[i,j] = np.mean(a[i,:,j])

    return aMean
#end axis_1_mean_3D

def element_normals(elemData):                                                 # element_normals
    '''
    FUNCTION TO COMPUTE ELEMENT INWARD-FACING NORMALS
     arguments: [elemData](np array) 3D element data array
       returns: [elemNormals](np array) 2D unit normal vector array
        author: thomas.j.wignall@nasa.gov
    last write: 12/2020
    '''

    u = elemData[:,1,:3] - elemData[:,0,:3]
    v = elemData[:,2,:3] - elemData[:,0,:3]
    elemNormals = np.cross(v,u) # v x u so that the normal faces inward
    elemNormals = (elemNormals.T / axis_1_norm_2D(elemNormals)).T

    return elemNormals
#end element_normals

def three_in_one_elems(elemData,elemAreas,procElems,binAreas,binMeans,i):      # three_in_one_elems
    '''
    FUNCTION TO FIND BIN AREAS AND MEANS FOR ELEMENTS WITH THREE NODES IN THE SAME BIN
     arguments: [elemData](np array) 3D element data array
                [elemAreas](np array) 1D array of element areas
                [procElems](np array) 1D Bool array of which elements apply
                [binAreas](np array) 2D array of bin sub-areas, to be updated
                [binMeans](np array) 3D array of bin sub-means, to be updated
                [i](int) bin index
       returns: [binAreas](np array) updated with processed elements
                [binMeans](np array) updated with processed elements
        author: michael.w.lee@nasa.gov
    last write: 06/2022
    '''

    binAreas[procElems,i] = elemAreas[procElems]
    binMeans[procElems,i,:] = axis_1_mean_3D(elemData[procElems,:,:])

    return binAreas, binMeans
#end three_in_one_elems

def two_in_one_elems(elemData, elemAreas, procElems, binAreas, binMeans, b, db, llIndex, nodesInBin, i):
    '''
    FUNCTION TO FIND BIN AREAS AND MEANS FOR ELEMENTS WITH TWO NODES IN THE BOTTOM BIN
     arguments: [elemData](np array) 3D element data array
                [elemAreas](np array) 1D array of element areas
                [procElems](np array) 1D Bool array of which elements apply
                [binAreas](np array) 2D array of bin sub-areas, to be updated
                [binMeans](np array) 3D array of bin sub-means, to be updated
                [b](float) first upper bin boundary
                [db](float) uniform bin width
                [llIndex](int) linelaod axis index (0,1,2)=(x,y,z)
                [nodesInBin](np array) 2D Bool array of which nodes are in this bin
                [i](int) bin index
       returns: [binAreas](np array) updated with processed elements
                [binMeans](np array) updated with processed elements
        author: michael.w.lee@nasa.gov
    last write: 06/2022
    '''

    pElem = procElems.sum()
    # size [pElem]
    outNodes, oneInNodes, twoInNodes = organize_nodes(nodesInBin[procElems,:])
    # size [pElem,maxIntPts+2,nPts], [nElem]

    if jit: #numba's jit can handle only one fancy slice at a time so we fall back to iterative method when using jit 
        one_in_node = fancy_slice(elemData, procElems, oneInNodes)
        two_in_node = fancy_slice(elemData, procElems, twoInNodes)
        top_nodes = fancy_slice(elemData, procElems, outNodes)
    else:
        one_in_node = elemData[procElems,oneInNodes,:]
        two_in_node = elemData[procElems,twoInNodes,:]
        top_nodes = elemData[procElems,outNodes,:]

    boundaries = np.array([b]*pElem)

    oneInterp, nInterpPts = interp_to_bin_bounds(one_in_node,
                                                top_nodes,
                                                boundaries, db, llIndex)
    twoInterp, _          = interp_to_bin_bounds(two_in_node,
                                                top_nodes,
                                                boundaries, db, llIndex)
    # size [pElem,maxIntPts]
    trapAreas = trap_areas(oneInterp[:,1:,:3], twoInterp[:,1:,:3], llIndex)

    binAreas[procElems,i+1:i+1+np.max(nInterpPts)] = trapAreas
    binAreas[procElems,i] = elemAreas[procElems] - binAreas[procElems,:].sum(axis=1)

    # size [pElem,maxIntPts+1,nPts,4]
    stackedData = np.stack((oneInterp[:,:-1,:],oneInterp[:,1:,:],\
                            twoInterp[:,:-1,:],twoInterp[:,1:,:]),axis=-1)
    allElems = np.where(procElems)[0]
    for j in range(len(allElems)):
        for k in range(nInterpPts[j]):
            binMeans[allElems[j],i+k,:] = axis_1_mean_2D(stackedData[j,k,:,:])
        binMeans[allElems[j],i+nInterpPts[j],:] = axis_1_mean_2D(stackedData[j,nInterpPts[j],:,:-1]) # last element is triangular
        if np.all(np.abs(stackedData[j,0,:3,0]-stackedData[j,0,:3,1])<1e-8): # if first element is triangular
            binMeans[allElems[j],i,:] = axis_1_mean_2D(stackedData[j,0,:,1:])
        elif np.all(np.abs(stackedData[j,0,:3,-2]-stackedData[j,0,:3,-1])<1e-8):
            binMeans[allElems[j],i,:] = axis_1_mean_2D(stackedData[j,0,:,:-1])

    return binAreas, binMeans
#end two_in_one_elems

def one_in_one_elems(elemData, elemAreas, elemsIn, binAreas, binMeans, b, db, llIndex, nodesInBin, i):
    '''
    FUNCTION TO FIND BIN AREAS AND MEANS FOR ELEMENTS WITH ONE NODE IN THE BOTTOM BIN
     arguments: [elemData](np array) 3D element data array
                [elemAreas](np array) 1D array of element areas
                [elemsIn](np array) 1D Bool array of which elements apply
                [binAreas](np array) 2D array of bin sub-areas, to be updated
                [binMeans](np array) 3D array of bin sub-means, to be updated
                [b](float) first upper bin boundary
                [db](float) uniform bin width
                [llIndex](int) linelaod axis index (0,1,2)=(x,y,z)
                [nodesInBin](np array) 2D Bool array of which nodes are in this bin
                [i](int) bin index
       returns: [binAreas](np array) updated with processed elements
                [binMeans](np array) updated with processed elements
                [num12nodes](int) number of nodes which had two nodes in the same top bin
                [num111nodes](int) number of nodes which had each node in a separate bin
        author: michael.w.lee@nasa.gov
    last write: 06/2022
    '''
    num12nodes = 0
    num111nodes = 0
    pElem = np.sum(elemsIn)
    inNodes, oneOutNodes, twoOutNodes = organize_nodes(~nodesInBin[elemsIn,:])
    
    # this move from one/two to max/min referencing streamlines the 1+1+1 cases
    #oneMaxNodes = elemData[elemsIn,oneOutNodes,llIndex] >= elemData[elemsIn,twoOutNodes,llIndex]
    compare1 = fancy_slice(elemData,elemsIn,oneOutNodes)[:,llIndex]
    compare2 = fancy_slice(elemData,elemsIn,twoOutNodes)[:,llIndex]
    oneMaxNodes = compare1 >= compare2
    twoMaxNodes = np.array([not a for a in oneMaxNodes])
    maxOutNodes = oneOutNodes.copy()
    maxOutNodes[twoMaxNodes] = twoOutNodes[twoMaxNodes]
    minOutNodes = oneOutNodes.copy()
    minOutNodes[oneMaxNodes] = twoOutNodes[oneMaxNodes]

    maxInterp, nInterpPtsMax = interp_to_bin_bounds(fancy_slice(elemData,elemsIn,inNodes),
                                                    fancy_slice(elemData,elemsIn,maxOutNodes),
                                                    np.array([b]*pElem), db, llIndex)
    minInterp, nInterpPtsMin = interp_to_bin_bounds(fancy_slice(elemData,elemsIn,inNodes),
                                                    fancy_slice(elemData,elemsIn,minOutNodes),
                                                    np.array([b]*pElem), db, llIndex)
    
    # process elements that have two nodes in the same upper bin
    procElems = np.zeros(len(elemsIn)) == 1
    whichElems = nInterpPtsMax == nInterpPtsMin # to access elemsIn-size arrays
    procElems[elemsIn] = whichElems # to access full-size arrays
    pElem = np.sum(procElems)

    if pElem > 0:
        someElems = np.where(whichElems)[0]
        allElems = np.where(procElems)[0]
        for j in range(len(allElems)):
            areas = trap_areas_2D( maxInterp[someElems[j], :nInterpPtsMin[someElems[j]]+1, :3],
                                   minInterp[someElems[j], :nInterpPtsMin[someElems[j]]+1, :3], llIndex)
            binAreas[allElems[j], i:i+nInterpPtsMin[someElems[j]]] = areas
            binAreas[allElems[j], i+nInterpPtsMin[someElems[j]]] = elemAreas[allElems[j]] - \
                                                                   binAreas[allElems[j],:].sum()

        minDex = min(maxInterp.shape[1], minInterp.shape[1])
        stackedData = np.stack((maxInterp[whichElems,:minDex-1,:], 
                                maxInterp[whichElems,1:minDex,:],
                                minInterp[whichElems,:minDex-1,:], 
                                minInterp[whichElems,1:minDex,:]),
                                axis=-1)

        for j in range(len(allElems)):
            for k in range(1,nInterpPtsMin[someElems[j]]+1):
                binMeans[allElems[j],i+k,:] = axis_1_mean_2D(stackedData[j,k,:,:])
            binMeans[allElems[j],i,:] = axis_1_mean_2D(stackedData[j,0,:,1:]) # first element is triangular
            if np.all(np.abs(stackedData[j,-1,:3,0]-stackedData[j,-1,:3,1])<1e-8): # if last element is triangular
                binMeans[allElems[j],i+nInterpPtsMin[someElems[j]],:] = axis_1_mean_2D(stackedData[j,-1,:,1:])
            elif np.all(np.abs(stackedData[j,-1,:3,-2]-stackedData[j,-1,:3,-1])<1e-8):
                binMeans[allElems[j],i+nInterpPtsMin[someElems[j]],:] = axis_1_mean_2D(stackedData[j,-1,:,:-1])
        num12nodes = procElems.sum()

        elemsIn[procElems] = False
        if not np.any(elemsIn):
            return binAreas, binMeans, num12nodes, num111nodes
    
    # process elements that have upper two nodes in different bins
    whichElems = ~whichElems
    pElem = np.sum(elemsIn)
    threeInterp,nInterpPtsThree = interp_to_bin_bounds(fancy_slice(elemData,elemsIn,minOutNodes[whichElems]),
                                                       fancy_slice(elemData,elemsIn,maxOutNodes[whichElems]),
                                                       b+db*nInterpPtsMin[whichElems],db,llIndex)

    someElems = np.where(whichElems)[0]
    allElems = np.where(elemsIn)[0]
    for j in range(pElem):
        mindex = nInterpPtsMin[someElems[j]]
        lowerTrapAreas = trap_areas_2D(maxInterp[someElems[j], :mindex+1, :3],
                                       minInterp[someElems[j], :mindex+1, :3], llIndex)
        upperTrapAreas = trap_areas_2D(maxInterp[someElems[j], mindex+1:mindex+2+nInterpPtsThree[j], :3],
                                       threeInterp[j,1:2+nInterpPtsThree[j],:3], llIndex)
        if upperTrapAreas.shape[0] > 1 and upperTrapAreas[-1] < 1e-10*np.mean(upperTrapAreas[:-1]): # catch for floating point slicing
            upperTrapAreas = upperTrapAreas[:-1]
            nInterpPtsThree[j] -= 1
        
        binAreas[allElems[j],i:i+mindex] = lowerTrapAreas
        binAreas[allElems[j],i+1+mindex:\
                         i+1+mindex+len(upperTrapAreas)] = upperTrapAreas
        binAreas[allElems[j],i+mindex] = elemAreas[allElems[j]] - \
                                         binAreas[allElems[j],:].sum()
        
        stackedData = np.stack((minInterp[someElems[j], :mindex+1, :], 
                                minInterp[someElems[j], 1:mindex+2, :],
                                maxInterp[someElems[j], :mindex+1, :],
                                maxInterp[someElems[j], 1:mindex+2, :],
                                minInterp[someElems[j], :mindex+1, :]),
                                axis=-1)
        stackedData[-1,:,-1] = threeInterp[j, 1, :] 
        for k in range(1,nInterpPtsMin[someElems[j]]):
            binMeans[allElems[j],i+k,:] = axis_1_mean_2D(stackedData[k,:,:-1])
        binMeans[allElems[j],i,:] = axis_1_mean_2D(stackedData[0,:,1:-1]) # first element is triangular
        if np.all(np.abs(stackedData[-1,:3,1]-stackedData[-1,:3,-1])<1e-8): # if central element is quadrilateral
            binMeans[allElems[j],i+nInterpPtsMin[someElems[j]],:] = axis_1_mean_2D(stackedData[-1,:,:-1])
        else: # else central element is pentagonal
            binMeans[allElems[j],i+nInterpPtsMin[someElems[j]],:] = axis_1_mean_2D(stackedData[-1,:,:])

        stackedData = np.stack((threeInterp[j,1:1+nInterpPtsThree[j],:],\
                                threeInterp[j,2:2+nInterpPtsThree[j],:],\
                                maxInterp[someElems[j],mindex+1:mindex+1+nInterpPtsThree[j],:],\
                                maxInterp[someElems[j],mindex+2:mindex+2+nInterpPtsThree[j],:]),axis=-1)
        for k in range(nInterpPtsThree[j]-1):
            binMeans[allElems[j],i+mindex+1+k,:] = axis_1_mean_2D(stackedData[k,:,:])
        binMeans[allElems[j],i+mindex+nInterpPtsThree[j],:] = axis_1_mean_2D(stackedData[-1,:,:-1]) # last element is triangular
    
    num111nodes = pElem
    return binAreas, binMeans, num12nodes, num111nodes
#end one_in_one_elems

def fancy_slice(array, condition1, condition2):
    array1 = np.empty((condition2.shape[0], array.shape[-1]))

    j=0
    for i in range(condition1.shape[0]):
        if condition1[i]:
            array1[j,:] = array[i, condition2[j], :]
            j+=1
    return array1      


jit_module(nopython=True)

