import numpy as np
import tecplot as tp
from tecplot.constant import ZoneType


def extract_surfaces_and_save_as_npy(dataset, zone, vars_to_save, out_fn):
    print(zone.name)
    print('Extracting surface file...')

    num_points = zone.num_points
    data = np.zeros((num_points, len(vars_to_save)))
    varobjstosave = []
    for i, variables in enumerate(vars_to_save):
        data[:, i] = zone.values(variables).as_numpy_array()
        varobjstosave.append(dataset.variable(variables))

    connectivity = np.asarray(zone.nodemap.array[:]).astype(int)
    if zone.zone_type == ZoneType.FETetra or zone.zone_type == ZoneType.FEQuad:
        print('Tetrahedral zone type detected.')
        connectivity = connectivity.reshape((-1, 4))
    elif zone.zone_type == ZoneType.FETriangle:
        print('Triangular zone type detected.')
        connectivity = connectivity.reshape((-1, 3))

    points_used = np.unique(connectivity.flatten())

    data_to_save = data[points_used, :]
    new_connectivity = np.searchsorted(points_used, connectivity)
    np.save(out_fn + '_data.npy', data_to_save)
    np.save(out_fn + '_conn.npy', new_connectivity)
    return 0


def convert_numpy_to_tecplot(nodal_data, conn_data, vars_to_save, sfcfilename, new_ds=True):
    # this takes the outputted numpy formatted files and recombines them in a way tecplot can use


    num_points = nodal_data.shape[0]
    num_elements = conn_data.shape[0]

    frame = tp.active_frame()
    if new_ds:
        dataset = frame.create_dataset('surf', vars_to_save, reset_style=True)
    else:
        dataset = frame.dataset
    if conn_data.shape[1] == 4:
        zone = dataset.add_fe_zone(tp.constant.ZoneType.FEQuad, sfcfilename, num_points, num_elements)
    elif conn_data.shape[1] == 3:
        zone = dataset.add_fe_zone(tp.constant.ZoneType.FETriangle, sfcfilename, num_points, num_elements)
    for i, variables in enumerate(vars_to_save):
        zone.values(variables)[:] = nodal_data[:, i]
    zone.nodemap.array[:] = conn_data

    return dataset

if __name__ == '__main__':
    import os
    import dfrom
    import sys
    
    if len(sys.argv) > 1:
        vehicle = sys.argv[1]
    else:
        vehicle = ''

    working_dir = os.path.join('..', dfrom.config,'surface_files')
    vars_to_save = ["X", "Y", "Z", "CP", "CFX", "CFY", "CFZ"]
    surfaces_to_proces = [fn for fn in os.listdir(working_dir) if '_data.npy' in fn if vehicle in fn]
    surfaces_to_proces.sort()
    for sfcfilename in surfaces_to_proces:
        print(sfcfilename)
        sfcfilename = sfcfilename.replace('_data.npy','')
        print(sfcfilename)
        working_file = os.path.join(working_dir, sfcfilename)
        dataset = convert_numpy_to_tecplot(working_file, vars_to_save)

        new_fn = os.path.join(working_dir, sfcfilename + '.plt')
        tp.data.save_tecplot_plt(new_fn, dataset=dataset)

    print('end')


