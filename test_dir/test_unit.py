import os
import unittest
import pytest
import numpy as np
import pclam
import pclam.calc as calc

# # # # # IO TESTING                                                                     # # # # #

@pytest.mark.parametrize('surf_data_file',['unit_test_data.dat','unit_test_data.plt'])
def test_load_surf_data(surf_data_file):
    config_info = pclam.Config({'variables_to_load':['X','Y','Z','CP','CFX','CFY','CFZ']})

    _, _, _ = pclam.io.load_surf_data(surf_data_file, config_info=config_info)

@pytest.mark.parametrize('inputfile',['','unit_test_input.json'])
@pytest.mark.parametrize('print_sample_input',[True,False])
def test_get_input(inputfile,print_sample_input):
    _ = pclam.io.get_input(inputfile,print_sample_input)
    if print_sample_input:
        assert os.path.exists('sample_lineload_input.json')
        os.remove('sample_lineload_input.json')

@pytest.mark.parametrize('out_file',['temp.dat','temp.npy','temp.plt'])
def test_write_lineload(out_file):
    lineload = np.zeros((101,14))
    axis = 'axis'
    profile = 'profile'
    _ = pclam.io.write_lineload(lineload, out_file, axis, profile)
    assert os.path.exists(out_file)
    os.remove(out_file)

def test_bin_weights_io():
    config = pclam.Config({'mapping_file_dir':'.','mapping_file_name':'temp'})
    binAreas = np.zeros((101,11))
    binWeights = np.zeros((11,101,3))

    pclam.io.save_bin_weights(config,binAreas,binWeights)
    testAreas,testWeights = pclam.io.load_bin_weights(config)
    assert np.all(binAreas==testAreas)
    assert np.all(binWeights==testWeights)
    assert os.path.exists('temp_bin_weights.npz')
    os.remove('temp_bin_weights.npz')


# # # # # UTIL TESTING                                                                   # # # # #

def test_calc_lineload():
    mrp = np.array([-1,0,1])
    nLLpoints = 17
    binMeans = np.random.rand(11,17,9)
    binAreas = np.random.rand(11,17)
    profileIndex = 2

    _, _ = calc.calc_lineload(mrp, nLLpoints, binMeans, binAreas, profileIndex)

def test_organize_nodes():
    falseNodes   = [0,0,1,1,2,2]
    oneTrueNodes = [1,2,0,2,0,1]
    twoTrueNodes = [2,1,2,0,1,0]
    nodesInBin = np.zeros((6,3)) == 0
    for i in range(nodesInBin.shape[0]):
        nodesInBin[i,falseNodes[i]] = False
    oFalseNodes,oOneTrueNodes,oTwoTrueNodes = calc.organize_nodes(nodesInBin)
    assert np.all(falseNodes == oFalseNodes)
    assert np.all(oFalseNodes+oOneTrueNodes+oTwoTrueNodes == 3)

def test_interp_to_bin_bounds():
    botNodes = np.array([[0,0,0,0,0],\
                         [0,0,0,0,0],\
                         [0,0,0,0,0],\
                         [0.5,0,0,0.5,1],\
                         [0,0,0,0,0],\
                         [0,0,0,0,0]])
    topNodes = np.array([[0,0,0,0,0],\
                         [1,0,0,1,2],\
                         [1.5,0,0,1.5,3],\
                         [2,0,0,2,4],\
                         [2.5,0,0,2.5,5],\
                         [3,0,0,3,6]])
    b = np.ones(6)
    db = 1
    llIndex = 0
    
    interpData = np.array([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],\
                           [[0,0,0,0,0],[1,0,0,1,2],[1,0,0,1,2],[1,0,0,1,2]],\
                           [[0,0,0,0,0],[1,0,0,1,2],[1.5,0,0,1.5,3],[1.5,0,0,1.5,3]],\
                           [[0.5,0,0,0.5,1],[1,0,0,1,2],[2,0,0,2,4],[2,0,0,2,4]],\
                           [[0,0,0,0,0],[1,0,0,1,2],[2,0,0,2,4],[2.5,0,0,2.5,5]],\
                           [[0,0,0,0,0],[1,0,0,1,2],[2,0,0,2,4],[3,0,0,3,6]]])
    nInterpPts = [0,0,1,1,2,2]
    oInterpData,oNInterpPts = calc.interp_to_bin_bounds(botNodes,topNodes,b,db,llIndex)
    assert np.all(np.isclose(interpData,oInterpData))
    assert np.all(np.isclose(nInterpPts,oNInterpPts))

def test_trap_areas():
    topPoints = np.array([[[0,1,0],[1,.5,0],[0,0,0]],\
                          [[.5,0,0],[1,.5,0],[2,1,0]]])
    botPoints = np.array([[[0,0,0],[1,0,0],[0,0,0]],\
                          [[.5,0,0],[1,0,0],[2,0,0]]])
    llIndex = 0
    
    trapAreas = np.array([[.75,.25],[.125,.75]])
    oTrapAreas = calc.trap_areas(topPoints,botPoints,llIndex)
    assert np.all(np.isclose(trapAreas,oTrapAreas))

def test_triangle_areas():
    p1 = np.array([[0,0,0],[0,0,0]])
    p2 = np.array([[1,0,0],[0,-1,0]])
    p3 = np.array([[1,1,0],[0,-1,-1]])

    areas = np.array([.5,.5])
    oAreas = calc.triangle_areas(p1,p2,p3)
    assert np.all(np.isclose(areas,oAreas))

@pytest.mark.parametrize('dim',[(11,7,3),(5,17,11),(23,29,111)])
def test_axis_1_norms(dim):
    a = np.random.rand(dim[0],dim[1],dim[2])
    aNorm = np.linalg.norm(a,axis=1)
    oANorm = calc.axis_1_norm_2D(a[:,:,0])
    assert np.all(np.isclose(aNorm[:,0],oANorm))
    oANorm = calc.axis_1_norm_3D(a)
    assert np.all(np.isclose(aNorm,oANorm))

def test_element_normals():
    elemData = np.array([[[0,0,0],[1,0,0],[1,1,0]],\
                         [[0,0,0],[0,1,0],[1,1,0]],\
                         [[0,0,0],[0,1,0],[0,1,1]]])
    elemNormals = np.array([[0,0,-1],[0,0,1],[-1,0,0]]) # negative because inverted
    oElemNormals = calc.element_normals(elemData)
    assert np.all(np.isclose(elemNormals,oElemNormals))

@pytest.mark.parametrize('procElems',[np.array([False,True,False]),np.array([True,True,False]),\
                                      np.array([True,True,True])])
def test_three_in_one_elems(procElems):
    elemData = np.array([[[0,0,0,0,0,0,0,0,0],[1,0,0,1,2,3,4,5,6],[1,1,0,2,3,4,5,6,7]],\
                         [[0,0,0,0,0,0,0,0,0],[0,1,0,3,4,5,6,7,8],[1,0,0,4,5,6,7,8,9]],\
                         [[0,0,0,0,0,0,0,0,0],[1,1,0,5,6,7,8,9,0],[1,0,1,6,7,8,9,0,1]]])
    elemAreas = np.array([1/3,1/3,1/3])
    binAreas = np.zeros((3,2))
    binMeans = np.zeros((3,2,9))
    i = 0
    oBinAreas, oBinMeans = calc.three_in_one_elems(elemData,elemAreas,procElems,binAreas,binMeans,i)
    binAreas = np.array([[1/3,0],[1/3,0],[1/3,0]])
    binMeans = np.array([[[2/3,1/3,0,1,5/3,7/3,3,11/3,13/3],[0,0,0,0,0,0,0,0,0]],\
                         [[1/3,1/3,0,7/3,3,11/3,13/3,5,17/3],[0,0,0,0,0,0,0,0,0]],\
                         [[2/3,1/3,1/3,11/3,13/3,5,17/3,3,1/3],[0,0,0,0,0,0,0,0,0]]])
    assert(np.all(oBinAreas[procElems,:] == binAreas[procElems,:]))
    assert(np.all(oBinMeans[procElems,:,:] == binMeans[procElems,:,:]))

@pytest.mark.parametrize('theseElems',[[0],[0,1],[0,4],[0,1,2,3],[0,1,2,3,4,5,6,7]])
def test_two_in_one_elems(theseElems):
    procElems = np.zeros(8) == 1
    procElems[theseElems] = True
    elemData = np.array([[[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[1.5,0,0,1,2,3,4,5,6]],\
                         [[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[2,0,0,2,3,4,5,6,7]],\
                         [[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[2.5,0,0,3,4,5,6,7,8]],\
                         [[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[3,0,0,4,5,6,7,8,9]],\
                         [[0,1,0,0,0,0,0,0,0],[1,0,0,1,1,1,1,1,1],[1.5,0,0,1,2,3,4,5,6]],\
                         [[0,1,0,0,0,0,0,0,0],[1,0,0,1,1,1,1,1,1],[2,0,0,2,3,4,5,6,7]],\
                         [[0,1,0,0,0,0,0,0,0],[1,0,0,1,1,1,1,1,1],[2.5,0,0,3,4,5,6,7,8]],\
                         [[0,1,0,0,0,0,0,0,0],[1,0,0,1,1,1,1,1,1],[3,0,0,4,5,6,7,8,9]]])
    elemAreas = np.array([1.5/2,1,2.5/2,3/2,.5/2,1/2,1.5/2,1])
    binAreas = np.zeros((8,4))
    binMeans = np.zeros((8,4,9))
    i = 0
    b = 1
    db = 1
    llIndex = 0
    nodesInBin = elemData[:,:,llIndex] <= b
    oBinAreas, oBinMeans = calc.two_in_one_elems(elemData,elemAreas,procElems,binAreas,binMeans,\
                                        b,db,llIndex,nodesInBin,i)
    binAreas = np.array([[2/3,1/12,0,0],[3/4,1/4,0,0],[4/5,2/5,1/20,0],[5/6,1/2,1/6,0],\
                         [1/6,1/12,0,0],[1/4,1/4,0,0],[3/10,2/5,1/20,0],[1/3,1/2,1/6,0]])
    binMeansX = np.array([[2/4,3.5/3,0,0],[2/4,4/3,0,0],[2/4,6/4,6.5/3,0],[2/4,6/4,7/3,0],\
                          [2/3,3.5/3,0,0],[2/3,4/3,0,0],[2/3,6/4,6.5/3,0],[2/3,6/4,7/3,0]])
    assert(np.all(np.isclose(oBinAreas[procElems,:], binAreas[procElems,:])))
    assert(np.all(np.isclose(oBinMeans[procElems,:,0], binMeansX[procElems,:])))

@pytest.mark.parametrize('theseElems',[[0],[4],[0,4],\
                                       [1],[0,1],[1,5],\
                                       [0,1,2,3],[0,1,2,3,4,5,6,7]])
def test_one_in_one_elems(theseElems):
    procElems = np.zeros(8) == 1
    procElems[theseElems] = True
    elemData = np.array([[[0,0,0,0,0,0,0,0,0],[1.5,0,0,0,0,0,0,0,0],[2,1,0,1,2,3,4,5,6]],\
                         [[0,0,0,0,0,0,0,0,0],[1.5,0,0,0,0,0,0,0,0],[2.5,1,0,2,3,4,5,6,7]],\
                         [[0,0,0,0,0,0,0,0,0],[1.5,0,0,0,0,0,0,0,0],[3,1,0,3,4,5,6,7,8]],\
                         [[0,0,0,0,0,0,0,0,0],[1.5,0,0,0,0,0,0,0,0],[3.5,1,0,4,5,6,7,8,9]],\
                         [[0,0,0,0,0,0,0,0,0],[2,0,0,1,1,1,1,1,1],[2,1,0,1,2,3,4,5,6]],\
                         [[0,0,0,0,0,0,0,0,0],[2,0,0,1,1,1,1,1,1],[2.5,1,0,2,3,4,5,6,7]],\
                         [[0,0,0,0,0,0,0,0,0],[2,0,0,1,1,1,1,1,1],[3,1,0,3,4,5,6,7,8]],\
                         [[0,0,0,0,0,0,0,0,0],[2,0,0,1,1,1,1,1,1],[3.5,1,0,4,5,6,7,8,9]]])
    elemAreas = np.array([3/4,3/4,3/4,3/4,1,1,1,1])
    binAreas = np.zeros((8,4))
    binMeans = np.zeros((8,4,9))
    i = 0
    b = 1
    db = 1
    llIndex = 0
    nodesInBin = elemData[:,:,llIndex] <= b
    oBinAreas, oBinMeans, _, _ = calc.one_in_one_elems(elemData,elemAreas,procElems,binAreas,\
                                        binMeans,b,db,llIndex,nodesInBin,i)
    binAreas = np.array([[1/4,1/2,0,0],[1/5,19/40,3/40,0],[1/6,5/12,1/6,0],[1/7,41/112,3/14,3/112],\
                         [1/4,3/4,0,0],[1/5,3/5,1/5,0],[1/6,1/2,1/3,0],[1/7,3/7,8/21,1/21]])
    binMeansX = np.array([[2/3,5.5/4,0,0],[2/3,7.5/5,6.5/3,0],[2/3,7.5/5,7/3,0],[2/3,7.5/5,10/4,9.5/3],\
                          [2/3,6/4,0,0],[2/3,6/4,6.5/3,0],[2/3,6/4,7/3,0],[2/3,6/4,10/4,9.5/3]])
    assert(np.all(np.isclose(oBinAreas[procElems,:], binAreas[procElems,:])))
    assert(np.all(np.isclose(oBinMeans[procElems,:,0], binMeansX[procElems,:])))


# # # # # MAIN TESTING                                                                   # # # # #
@pytest.mark.parametrize('numPts',[4,5,6,7,44])
def test_pad_data(numPts):
    config = pclam.Config({})
    if numPts == 44:
        nodeData = np.ones((11,4,4))
        elemData = np.ones((11,4,4))
    else:
        nodeData = np.ones((11,numPts))
        elemData = np.ones((11,3,numPts))
    newData,newElem,config = pclam.util.pad_data(nodeData,elemData,config)
    
    if numPts == 4:
        assert np.all(nodeData == newData[:,:4])
    elif numPts == 6:
        assert np.all(nodeData == newData[:,[0,1,2,4,5,6]])
    elif numPts == 7:
        assert np.all(nodeData == newData)
    else:
        assert newData == None

def test_writes():
    config = pclam.Config({'output_dir':'unittest','base_name':'test',\
                           'variables_saved':['cp','cfx','cfy','cfz','all'],'output_type':'.npy'})
    lineloads = np.random.rand(11,14,5)
    
    pclam.util.call_write_lineloads(lineloads,config)
    assert os.path.exists('unittest/')
    for i,var in enumerate(config.variables_saved):
        f = 'unittest/test_'+var+'_LL.npy'
        assert os.path.exists(f)
        compare = np.load(f)
        assert np.all(compare == lineloads[:,:,i])
        os.remove(f)
    os.rmdir('unittest')
    
    fandm = pclam.util.get_and_write_fandm(lineloads,config)
    assert os.path.exists('unittest/')
    for i,var in enumerate(config.variables_saved):
        f = 'unittest/test_'+var+'_fandm.csv'
        assert os.path.exists(f)
        compare = np.loadtxt(f)
        assert np.all(compare == fandm[i,:])
        os.remove(f)
    os.rmdir('unittest')

def test_node_to_element():
    config = pclam.Config({'variables_to_load':['X','Y','Z','CP','CFX','CFY','CFZ']})
    nodeData, elemData, conn = pclam.io.load_surf_data('unit_test_data.dat',config)
    nodeData, elemData, config = pclam.util.pad_data(nodeData,elemData,config)

    elemCompare = np.zeros((19,3,7))
    nodes = [[2,5,3],[3,5,7],[7,5,6],[7,6,9],[9,11,7],[9,10,11],[11,10,14],[14,10,13],\
             [2,1,5],[5,1,4],[4,6,5],[4,8,6],[6,8,9],[9,8,10],[10,8,13],\
             [0,4,1],[0,8,4],[0,12,8],[8,12,13]]
    for i in range(19):
        elemCompare[i,:,:] = np.stack((nodeData[nodes[i][0],:],\
                                       nodeData[nodes[i][1],:],\
                                       nodeData[nodes[i][2],:]),axis=0)
    elemCompare = np.dstack((elemCompare[:,:,:3],np.zeros((19,3,2)),-elemCompare[:,:,3],elemCompare[:,:,4:]))
    elemData = pclam.util.node_to_element(nodeData, conn, debug=False)
    assert np.all(np.isclose(elemData, elemCompare))

def test_elem_bin_means():
    config = pclam.Config({'variables_to_load':['X','Y','Z','CP','CFX','CFY','CFZ']})
    nodeData, elemData, conn = pclam.io.load_surf_data('unit_test_data.dat', config)
    nodeData, elemData, config = pclam.util.pad_data(nodeData, elemData, config)

    elemData = pclam.util.node_to_element(nodeData, conn, debug=False)

    bins = np.array([0,1,2,3,4,5])
    elemAreas = calc.triangle_areas(elemData[:,0,:3],elemData[:,1,:3],elemData[:,2,:3])
    binAreasCompare = np.array([[0.5, 0.000, 0.0000, 0.0, 0.0000],\
                                [0.5, 0.250, 0.0000, 0.0, 0.0000],\
                                [0.0, 0.250, 0.0000, 0.0, 0.0000],\
                                [0.0, 0.375, 0.1250, 0.0, 0.0000],\
                                [0.0, 0.125, 0.8125, 0.5, 0.0625],\
                                [0.0, 0.000, 0.0625, 0.5, 0.4375],\
                                [0.0, 0.000, 0.0000, 0.0, 0.2500],\
                                [0.0, 0.000, 0.0000, 0.0, 0.5000],\
                                [0.5, 0.000, 0.0000, 0.0, 0.0000],\
                                [0.5, 0.000, 0.0000, 0.0, 0.0000],\
                                [0.0, 0.250, 0.0000, 0.0, 0.0000],\
                                [0.0, 0.625, 0.1250, 0.0, 0.0000],\
                                [0.0, 0.125, 0.3750, 0.0, 0.0000],\
                                [0.0, 0.000, 0.4375, 0.5, 0.0625],\
                                [0.0, 0.000, 0.0625, 0.5, 0.6875],\
                                [0.5, 0.000, 0.0000, 0.0, 0.0000],\
                                [0.3, 0.400, 0.0500, 0.0, 0.0000],\
                                [0.2, 0.600, 0.9000, 0.6, 0.2000],\
                                [0.0, 0.000, 0.0500, 0.4, 0.8000]])
    binMeansXCompare = np.array([[  1/3,      0,      0,      0,      0],\
                                 [  2/3,  3.5/3,      0,      0,      0],\
                                 [    0,    4/3,      0,      0,      0],\
                                 [    0,    7/4,  6.5/3,      0,      0],\
                                 [    0,  5.5/3, 12.5/5,   14/4, 12.5/3],\
                                 [    0,      0,  8.5/3,   14/4,   17/4],\
                                 [    0,      0,      0,      0,   14/3],\
                                 [    0,      0,      0,      0, 14.5/3],\
                                 [  1/3,      0,      0,      0,      0],\
                                 [  2/3,      0,      0,      0,      0],\
                                 [    0,  3.5/3,      0,      0,      0],\
                                 [    0,  6.5/4,  6.5/3,      0,      0],\
                                 [    0,  5.5/3,    9/4,      0,      0],\
                                 [    0,      0,   11/4,   14/4, 12.5/3],\
                                 [    0,      0,  8.5/3,   14/4, 17.5/4],\
                                 [  1/3,      0,      0,      0,      0],\
                                 [  2/3,    6/4,  6.5/3,      0,      0],\
                                 [  2/3,    6/4, 12.5/5,   14/4,   13/3],\
                                 [    0,      0,  8.5/3,   14/4,   18/4]])
    binAreas, binMeans, binWeights = pclam.util.get_elem_bin_means(elemData,elemAreas,bins,1,0)
    # check 3-nodes
    indices = [0,2,6,7,8,9,10,15]
    assert np.all(np.isclose(binAreas[indices,:],binAreasCompare[indices,:]))
    assert np.all(np.isclose(binMeans[indices,:,0],binMeansXCompare[indices,:]))
    # check 2+1-nodes
    indices = [1,3,11,13,16]
    assert np.all(np.isclose(binAreas[indices,:],binAreasCompare[indices,:]))
    assert np.all(np.isclose(binMeans[indices,:,0],binMeansXCompare[indices,:]))
    # check 1+2-nodes
    indices = [5,12,14,18]
    assert np.all(np.isclose(binAreas[indices,:],binAreasCompare[indices,:]))
    assert np.all(np.isclose(binMeans[indices,:,0],binMeansXCompare[indices,:]))
    # check 1+1+1-nodes
    indices = [4,17]
    assert np.all(np.isclose(binAreas[indices,:],binAreasCompare[indices,:]))
    assert np.all(np.isclose(binMeans[indices,:,0],binMeansXCompare[indices,:]))

    regotBinMeans = pclam.util.reget_elem_bin_means(elemData,binWeights)
    assert np.all(np.isclose(binMeans,regotBinMeans))
'''
def test_make_lineload():
    config = pclam.io.get_input('unit_test_input.json',False)
    nodeData,_,conn = pclam.io.load_surf_data('unit_test_data.dat',config)
    _ = pclam.make_lineload(config,nodeData,conn)
'''
if __name__ == '__main__':
    unittest.main()
