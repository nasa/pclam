import os
import sys
import numpy as np
from scipy.sparse import csc_matrix
import json
from .config import Config

def load_surf_data(surf_data_file, config_info=None, node_only=False):

    if surf_data_file.endswith('.npy'):
        data_by_node = np.load(surf_data_file)
        if not node_only:
            node_to_elem = np.load(surf_data_file.replace('_data', '_conn'))
            data_by_elem = get_data_by_elem(data_by_node, node_to_elem)

    elif surf_data_file.endswith('.dat'):
        if node_only:
            data_by_node = load_ascii_tp0(surf_data_file, node_only, config_info)
        else:
            data_by_node, data_by_elem, node_to_elem = load_ascii_tp0(surf_data_file, node_only, config_info)

    elif surf_data_file.endswith('.plt') or surf_data_file.endswith('.szplt'):
        data_by_node, data_by_elem, node_to_elem = load_binary_tp(surf_data_file, config_info)

    if node_only:
        return data_by_node
    else:
        return data_by_node, data_by_elem, node_to_elem

def get_input(inputfile, print_sample_input):
    input_info = {}
    if len(inputfile) > 0:
        if os.path.isfile(inputfile):
            with open(inputfile, 'r') as fo:
                input_info = json.load(fo)
        else:
            print('Inputfile: ' + inputfile + ' cannot be found')
            ignore_error = input('Would you like to continue with default values? y/[n]')
            if not ignore_error.startswith('y'):
                sys.exit()

    input_info = Config(input_info)

    if len(input_info.defaults_used) > 0:
        print('Default values used for:')
        print(input_info.defaults_used)

    if print_sample_input:
        print("Printing 'sample_lineload_input.json'")
        with open('sample_lineload_input.json', 'w') as fo:
            json.dump(input_info.__dict__, fo, indent=4)

    return input_info

def read_stations(config):
    '''
    FUNCTION TO READ A CUSTOM BIN STATION FILE
     arguments: [config](object) user settings object
       returns: [bins](np array) 1D vector of bin stations
        author: michael.w.lee@nasa.gov
    '''
    try:
        bins = np.loadtxt(config.station_file)
        if len(bins.shape) != 1:
            bins = None
    except:
        bins = None

    return bins

def write_lineload(lineload, out_file, axis, profile): 
    '''
    FUNCTION TO WRITE PROVIDED LINELOAD IN USER-DEFINED FORMAT
     arguments: [lineload](np array) lineload array[nLLpoints,14]
                [out_file](String or Path) used to determine file type and outputname
                [axis](string) lineload axis label
                [profile](string) profile axis label
       returns: 0
        author: thomas.j.wignall@nasa.gov
    '''

    #done to confirm that we don't have the leading period in the extention 
    output_type = out_file[-3:]

    #using a caller dispatcher instead of if else statements
    # I (tjw) like this format better than tons of if, elif and else statements
    function_dict = {}
    function_dict['dat'] = write_ascii_tecplot
    function_dict['npy'] = write_numpy
    function_dict['plt'] = write_binary_tecplot

    function_dict[output_type](lineload, out_file, axis, profile)

    return 0

def write_fandm(fandm, out_file):
    '''
    FUNCTION TO WRITE PROVIDED INTEGRATED FORCE & MOMENT VECTOR
     arguments: [fandm](np array) force and moment vector [CX, CY, CZ, CMX, CMY, CMZ]
                [out_file](string or path) uesd to determine output name
       returns: 0
       author: michael.w.lee@nasa.gov
    '''

    np.savetxt(out_file, fandm, header='CX,CY,CZ,CMX,CMY,CMZ',delimiter=',')
    return 0

def write_ascii_tecplot(lineload, out_file, axis, profile):

    header = 'VARIABLES= '+axis.upper()+'       Cx            Cy            Cz           Cmx           Cmy           Cmz       '+profile.upper()
    if lineload.shape[1] == 14:
        header += "       CCx            CCy            CCz           CCmx           CCmy           CCmz"
    header += '\nZONE'
    np.savetxt(out_file, lineload, header=header, comments='')
    return 0

def write_numpy(lineload, out_file, axis, profile): # unused 'profile' needed for write_lineload logic
    np.save(out_file, lineload)
    return 0

def write_binary_tecplot(lineload, out_file, axis, profile):
    import tecplot as tp
    print('saving binary tecplot not yet implemented - saving ascii')
    write_ascii_tecplot(lineload, out_file[:-3]+'dat', axis, profile)

    return 0

def load_ascii_tp0(data_file, node_only, config):
    try:
        f = open(data_file,'r')
        counting = False
        for i,line in enumerate(f):
            if 'variables' in line.lower():
                file_vars = line.upper().split('"')[1::2]
                varIds = [j for j in range(len(file_vars)) if file_vars[j].upper() in [k.upper() for k in config.variables_to_load]]
                print('loaded variables: ',end='')
                print([config.variables_to_load[k] for k in varIds],flush=True)

            if counting:
                if len(line.split()) == num_vars:
                    num_nodes += 1
                else:
                    break
            else:
                try:
                    test = np.float64(line.split()[0])
                    num_header_rows = i
                    num_vars = len(line.split())
                    num_nodes = 1
                    counting = True
                except:
                    continue
        f.close()
        data_by_node = np.loadtxt(data_file, usecols=varIds,\
                                  skiprows=num_header_rows, max_rows=num_nodes)

        if not node_only:
            node_to_elem = np.loadtxt(data_file, skiprows=num_header_rows+num_nodes, dtype=np.int64)
            node_to_elem = np.searchsorted(np.unique(node_to_elem.flatten()), node_to_elem)
            data_by_elem = get_data_by_elem(data_by_node, node_to_elem)
            return data_by_node, data_by_elem, node_to_elem
        else:
            return data_by_node
    except: #catches cases with multiple zones or other idiosyncrasies
        return load_binary_tp(data_file,config)

def load_binary_tp(surf_data_file, config_info):
    import tecplot as tp
    from tecplot.constant import ZoneType

    if surf_data_file.endswith('.plt') or surf_data_file.endswith('.dat'):
        dataset = tp.data.load_tecplot(surf_data_file, read_data_option=tp.constant.ReadDataOption.Replace)
    else:
        dataset = tp.data.load_tecplot_szl(surf_data_file)

    vars_to_save = config_info.variables_to_load
    if len(list(dataset.zones())) > 1:
        print('Warning! Code only outputs a single lineload and combines all surfaces')
    for i_zone, zone in enumerate(dataset.zones()):
        #print(zone.name)
        num_points = zone.num_points
        data_by_node = np.zeros((num_points, len(vars_to_save)))
        varobjstosave = []
        for i, variables in enumerate(vars_to_save):
            data_by_node[:, i] = zone.values(variables).as_numpy_array()
            varobjstosave.append(dataset.variable(variables))

        connectivity = np.asarray(zone.nodemap.array[:]).astype(int)
        if zone.zone_type == ZoneType.FETetra or zone.zone_type == ZoneType.FEQuad:
            #logger.debug('Tetrahedral or Quad zone type detected.')
            connectivity = connectivity.reshape((-1, 4))
        elif zone.zone_type == ZoneType.FETriangle:
            #logger.debug('Triangular zone type detected.')
            connectivity = connectivity.reshape((-1, 3))
        else:
            #logger.error('Unknown Zone type. Defaulting to 4 points per element')
            #logger.error(zone.zone_type)
            connectivity = connectivity.reshape((-1, 4))

        # have encoutred zones that include unused points
        # this is a way to removethose
        points_used = np.unique(connectivity.flatten())
        data_by_node = data_by_node[points_used, :]
        node_to_elem = np.searchsorted(points_used, connectivity)

        data_by_elem = get_data_by_elem(data_by_node, node_to_elem)


        if i_zone == 0:
            data_by_node_full = data_by_node.copy()
            data_by_elem_full = data_by_elem.copy()
            node_to_elem_full = node_to_elem.copy()
        else:
            node_to_elem_full = np.vstack((node_to_elem_full, node_to_elem + data_by_node_full.shape[0]))
            data_by_node_full = np.vstack((data_by_node_full, data_by_node))
            data_by_elem_full = np.vstack((data_by_elem_full, data_by_elem))

    return data_by_node_full, data_by_elem_full, node_to_elem_full

def get_data_by_elem(data_by_node, node_to_elem):
    num_elem = node_to_elem.shape[0]
    data_by_elem = np.zeros([num_elem, 3, data_by_node.shape[1]])

    # this is help deal with the case where connectivity that is read in
    # is in 1-based index
    offset = node_to_elem.min()

    for i in range(num_elem):
        for j in range(3):
            ipoint = int(node_to_elem[i, j] - offset)  # -1 for 0 based index
            data_by_elem[i, j, :] = data_by_node[ipoint, :]
    return data_by_elem

def load_bin_weights(config):
    try:
        with np.load(os.path.join(config.mapping_file_dir,config.mapping_file_name+'_bin_weights.npz')) as data:
            binAreas = data['binAreas']
            binWeightsData = data['binWeightsData']
            binWeightsIndices = data['binWeightsIndices']
            binWeightsIndptr = data['binWeightsIndptr']
            binWeightsShape = data['binWeightsShape']
        binWeights1 = csc_matrix((binWeightsData[0],binWeightsIndices[0],\
                                  binWeightsIndptr[0]),shape=binWeightsShape[0])
        binWeights2 = csc_matrix((binWeightsData[1],binWeightsIndices[1],\
                                  binWeightsIndptr[1]),shape=binWeightsShape[1])
        binWeights3 = csc_matrix((binWeightsData[2],binWeightsIndices[2],\
                                  binWeightsIndptr[2]),shape=binWeightsShape[2])
        binWeights = np.dstack((binWeights1.A,binWeights2.A,binWeights3.A))
    except:
        binAreas = None
        binWeights = None
        
    return binAreas, binWeights

def save_bin_weights(config, binAreas, binWeights):
    binWeights1 = csc_matrix(binWeights[:,:,0])
    binWeights2 = csc_matrix(binWeights[:,:,1])
    binWeights3 = csc_matrix(binWeights[:,:,2])
    np.savez(os.path.join(config.mapping_file_dir,config.mapping_file_name+'_bin_weights.npz'), \
             binAreas=binAreas, \
             binWeightsData   =[binWeights1.data,   binWeights2.data,   binWeights3.data], \
             binWeightsIndices=[binWeights1.indices,binWeights2.indices,binWeights3.indices], \
             binWeightsIndptr =[binWeights1.indptr, binWeights2.indptr, binWeights3.indptr], \
             binWeightsShape  =[binWeights1.shape,  binWeights2.shape,  binWeights3.shape])
    return
