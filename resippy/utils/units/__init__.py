import os
from pint import UnitRegistry
ureg = UnitRegistry()

dir_path = os.path.dirname(os.path.realpath(__file__))
path_to_unit_defs_file = os.path.join(dir_path, 'unit_defs.txt')
ureg.load_definitions(path_to_unit_defs_file)
