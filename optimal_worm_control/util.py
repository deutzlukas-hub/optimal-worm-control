# From built-in
from typing import Optional
from argparse import ArgumentParser
# From rod-engine
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.parameter.physical_parameter import phyisical_parameter_parser
from worm_rod_engine.parameter.dimensionless_parameter import dimensionless_parameter_parser
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.parameter.util import convert_to_dimensionless
# From optimal-worm-control
from optimal_worm_control.parameter.control_parameter import control_param_parser
from optimal_worm_control.parameter.optimizer_parameter import optimizer_param_parser

def parameters_from_args(args: list = []):

    params = {}
    params['numerical_param'] = numerical_argument_parser.parse_known_args(args)[0]
    params['output_param'] = output_parameter_parser.parse_known_args(args)[0]
    params['control_param'] = control_param_parser.parse_known_args(args)[0]
    params['optimizer_param'] = optimizer_param_parser.parse_known_args(args)[0]

    phyiscal_param = phyisical_parameter_parser.parse_known_args(args)[0]
    if phyiscal_param.physical_to_dimensionless:
        params['physical_param'] = phyiscal_param
        params['dimensionless_param'] = convert_to_dimensionless(phyiscal_param)
    else:
        params['dimensionless_param'] = dimensionless_parameter_parser.parse_known_args(args)[0]

    return params

def linear_norm(min: float, max: float):
    def norm(obj_value: float):
        return (obj_value - min) / (max - min)
    return norm