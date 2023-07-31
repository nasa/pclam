import sys
import os
import unittest
import pytest
import numpy as np
from itertools import product
from pclam import make_lineload, Config
from pclam.util import call_write_lineloads
from pclam.io import write_lineload

import geometry.create_test_surface as cts


#probs a better way to do this but oh well
if '--save' in sys.argv:
    save_exact = True
else:
    save_exact =False
rocket_grids = ['smooth', 'bumpy']
rocket_fields = ['smooth', 'bumpy']
rocket_orientation = ['xy']

flat_grids = ['smooth', 'cubic', 'step']
flat_fields= ['step', 'wavy']
flat_orientation = ['xy', 'xz', 'yz']
rocket_node_to_elem = {}
rocket_data_by_node = {}
flat_node_to_elem = {}
flat_data_by_node = {}

for grid, field, orientation in product(rocket_grids, rocket_fields, rocket_orientation):
    case = (grid, field, orientation)
    node_to_elem, data_by_node = cts.create_sample_surfaces('rocket', grid, field, orientation)
    rocket_node_to_elem[case] = node_to_elem
    rocket_data_by_node[case] = data_by_node

for grid, field, orientation in product(flat_grids, flat_fields, flat_orientation):
    case = (grid, field, orientation)
    node_to_elem, data_by_node = cts.create_sample_surfaces('flat', grid, field, orientation, save_exact)
    flat_node_to_elem[case] = node_to_elem
    flat_data_by_node[case] = data_by_node


@pytest.mark.parametrize("grid", flat_grids)
@pytest.mark.parametrize("field", flat_fields)
@pytest.mark.parametrize("orientation", flat_orientation)
@pytest.mark.parametrize("nll", (33, 67))
@pytest.mark.parametrize("bin_edge",(True, False)) 
def test_flat_plate(grid, field, orientation, nll, bin_edge, save):
    config = Config({})
    case = (grid, field, orientation)
    for axis in orientation:
        config.base_name = '_'.join(('flat',) + case + (str(nll),))
        config.axis = axis
        config.nll_points = nll
        config.bin_edges_on_minmax = bin_edge
        data_by_node = flat_data_by_node[case]
        node_to_elem = flat_node_to_elem[case]
        ll = make_lineload(config, data_by_node, node_to_elem)
        if save:
            if bin_edge:
                edge = 'b1'
            else:
                edge = 'b0'
            output_dir = os.path.join('lineloads', edge, axis+'_axis')
            os.makedirs(output_dir, exist_ok=True)
            config.output_dir = output_dir
            config.variables_saved = ['all']
            call_write_lineloads(ll, config)

def test_large_bin_number(save):
    config = Config({})
    case = ('smooth', 'wavy', 'xy')
    config.base_name = 'large_bin_number'
    config.nll_points = 1000
    config.output_dir = os.path.join('lineloads', 'x_axis')
    data_by_node = flat_data_by_node[case]
    node_to_elem = flat_node_to_elem[case]
    ll = make_lineload(config, data_by_node, node_to_elem)
    if save:
        output_dir = os.path.join('lineloads', 'other')
        os.makedirs(output_dir, exist_ok=True)
        out_fn = config.base_name + '.dat'
        write_lineload(ll[:,:,4], os.path.join(output_dir, out_fn), config.axis, config.profile_axis)

def test_small_bin_number(save):
    config = Config({})
    case = ('smooth', 'wavy', 'xy')
    config.base_name = 'small_bin_number'
    config.nll_points = 20
    config.output_dir = os.path.join('lineloads', 'x_axis')
    data_by_node = flat_data_by_node[case]
    node_to_elem = flat_node_to_elem[case]
    ll = make_lineload(config, data_by_node, node_to_elem)
    if save:
        output_dir = os.path.join('lineloads', 'other')
        os.makedirs(output_dir, exist_ok=True)
        out_fn = config.base_name + '.dat'
        write_lineload(ll[:,:,4], os.path.join(output_dir, out_fn), config.axis, config.profile_axis)


@pytest.mark.parametrize("grid",rocket_grids)
@pytest.mark.parametrize("field", rocket_fields)
@pytest.mark.parametrize("orientation", rocket_orientation)
@pytest.mark.parametrize("bin_edge",(True, False)) 
@pytest.mark.parametrize("nll",(33, 67)) 
def test_rocket(grid, field, orientation, bin_edge, nll, save):
    #something up with this grid definition which causes some divide by 0 warnings
    config = Config({})
    config.nll_points = nll
    case = (grid, field, orientation)
    for axis in orientation:
        config.base_name = '_'.join(('rocket',) + case + (str(nll),))
        config.axis = axis
        config.output_dir = os.path.join('lineloads', axis+'_axis')
        data_by_node = rocket_data_by_node[case]
        node_to_elem = rocket_node_to_elem[case]
        ll = make_lineload(config, data_by_node, node_to_elem)

        if save:
            if bin_edge:
                edge = 'b1'
            else:
                edge = 'b0'
            output_dir = os.path.join('lineloads', edge, axis+'_axis')
            os.makedirs(output_dir, exist_ok=True)
            config.output_dir = output_dir
            config.variables_saved = ['cp', 'cfx', 'all']
            call_write_lineloads(ll, config)


if __name__ == '__main__':
    unittest.main()
