import sys
import os
import unittest
import pytest

import numpy as np
import tecplot as tp
from pathlib import Path
from itertools import product

import geometry.create_test_surface as cts


class MyTestCase(unittest.TestCase):
    rocket_grids = ['smooth', 'bumpy']
    rocket_fields = ['smooth', 'bumpy']
    rocket_orientation = ['xy']

    flat_grids = ['smooth', 'cubic', 'step']
    flat_fields= ['step', 'wavy']
    flat_orientation = ['xy', 'xz', 'yz']
    
    os.makedirs('ref_surfaces',exist_ok=True)
    os.makedirs('plots',exist_ok=True)
    os.makedirs('lineloads',exist_ok=True)
    os.makedirs('lineloads',exist_ok = True)
    os.makedirs('lineloads/x_axis',exist_ok=True)
    os.makedirs('lineloads/y_axis',exist_ok=True)
    os.makedirs('lineloads/z_axis',exist_ok=True)
    rocket_node_to_elem = {}
    rocket_data_by_node = {}
    flat_node_to_elem = {}
    flat_data_by_node = {}

    for grid, field, orientation in product(rocket_grids, rocket_fields, rocket_orientation):
        case = (grid, field, orientation)
        node_to_elem, data_by_node = cts.create_sample_surfaces('rocket', grid, field, orientation)
        rocket_node_to_elem[case] = node_to_elem
        rocket_data_by_node[case]= data_by_node

    for grid, field, orientation in product(flat_grids, flat_fields, flat_orientation):
        case = (grid, field, orientation)
        node_to_elem, data_by_node = cts.create_sample_surfaces('flat', grid, field, orientation)
        flat_node_to_elem[case] = node_to_elem
        flat_data_by_node[case] = data_by_node

    def test_save_sample_surface(self):
        from geometry import tp_related 
        vars_to_save = ["X", "Y", "Z", "CP", "CFX", "CFY", "CFZ"]
        #for i,case in enumerate(self.rocket_node_to_elem.items()):
        for case, node_to_elem in self.rocket_node_to_elem.items():
            dataset = tp_related.convert_numpy_to_tecplot(self.rocket_data_by_node[case],
                                                          node_to_elem, vars_to_save, 'sample')
            case_name = '_'.join(case)
            new_fn = Path('ref_surfaces', 'rocket_'+case_name+'.plt')
            tp.data.save_tecplot_plt(new_fn, dataset=dataset)
        #for i,case in enumerate(self.flat_cases):
        for case, node_to_elem in self.flat_node_to_elem.items():
            dataset = tp_related.convert_numpy_to_tecplot(self.flat_data_by_node[case],
                                                          node_to_elem, vars_to_save, 'sample')
            case_name = '_'.join(case)
            new_fn = Path('ref_surfaces', 'flat_'+case_name+'.plt')
            tp.data.save_tecplot_plt(new_fn, dataset=dataset)



if __name__ == '__main__':
    unittest.main()
