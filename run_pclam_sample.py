import datetime
import argparse

import pclam

parser = argparse.ArgumentParser()

parser.add_argument('surf_data_file',
                    help='Determines what file to read')
parser.add_argument('-n','--name', type=str,
                    help='Name to be used for outputting files. If not provided, name will be generated from the data file name.')
parser.add_argument('-V','--verbose', action="store_true",
                    help='Increases verbosity of outputs')
parser.add_argument('-i','--input', type=str, default='',
                    help='Input file to use to calculate lineloads. If none provided default values will be used.')
parser.add_argument('-p','--print_sample', action="store_true",
                    help='Outputs a sample input file')

args=parser.parse_args()
runname = args.surf_data_file
base_name = args.name
if base_name == None:
    base_name = 'placeholder'
debug = args.verbose
input_file = args.input
print_file = args.print_sample
    
print(base_name)
lineloads, fandm, config = pclam.run_pclam(input_file, runname, print_file, base_name, debug)

