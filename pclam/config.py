class Config(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        self.defaults_used = []

        if not hasattr(self, 'nll_points'):
            self.nll_points = 100
            self.defaults_used.append('nll_points')
        if not hasattr(self, 'use_bin_mapping_file'):
            self.use_bin_mapping_file = False
            self.defaults_used.append('use_bin_mapping_file')
        if not hasattr(self, 'axis'):
            self.axis = 'x'
            self.defaults_used.append('axis')
        if not hasattr(self, 'profile_axis'):
            self.profile_axis = 'z'
            self.defaults_used.append('profile_axis')
        if not hasattr(self, 'bin_edges_on_minmax'):
            self.bin_edges_on_minmax = True
            self.defaults_used.append('bin_edges_on_minmax')
        if not hasattr(self, 'output_dir'):
            self.output_dir = '.'
            self.defaults_used.append('output_dir')
        if not hasattr(self, 'mapping_file_dir'):
            self.mapping_file_dir = '.'
            self.defaults_used.append('mapping_file_dir')
        if not hasattr(self, 'base_name'):
            self.base_name = 'placeholder'
            self.defaults_used.append('base_name')
        if not hasattr(self, 'mapping_file_name'):
            self.mapping_file_name = 'placeholder'
            self.defaults_used.append('mapping_file_name')
        if not hasattr(self, 'mrp'):
            self.mrp = (0, 0, 0)
            self.defaults_used.append('mrp')
        else: #numba's jit compiler likes tuples more than lists
            self.mrp = tuple(self.mrp)
        if not hasattr(self, 'Lref'):
            self.Lref = 1
            self.defaults_used.append('Lref')
        if not hasattr(self, 'Sref'):
            self.Sref = 1
            self.defaults_used.append('Sref')

        '''
        if not hasattr(self, 'station_file'):
            self.use_station_file = False
            self.station_file = '0'
            self.defaults_used.append('station_file')
        elif self.station_file == '0':
            self.use_station_file = False
        else:
            self.use_station_file = True
        '''

        if not hasattr(self, 'print_fandm'):
            self.print_fandm = True
            self.defaults_used.append('print_fandm')

        if not hasattr(self, 'output_type'):
            self.output_type = '.dat'
            self.defaults_used.append('output_type')
        if not self.output_type.startswith('.'):
            self.output_type = '.' + self.output_type

        if not hasattr(self, 'variables_saved'):
            self.variables_saved = ['all']
            #input_info.variables = ['cp', 'cfx', 'cfy', 'cfz', 'all']
            self.defaults_used.append('variables_saved')
        if not hasattr(self, 'variables_to_load'):
            self.variables_to_load = ['x', 'y', 'z', 'cp']
            self.defaults_used.append('variables_to_load')
        if not hasattr(self, 'absolute_pressure_integration'):
            self.absolute_pressure_integration = False
            self.defaults_used.append('absolute_pressure_integration')
        if not hasattr(self, 'pinf_by_qref'):
            self.pinf_by_qref = 0
            self.defaults_used.append('pinf_by_qref')

