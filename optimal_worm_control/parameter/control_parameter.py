from argparse import ArgumentParser, BooleanOptionalAction

control_param_parser = ArgumentParser()
control_param_parser.add_argument('--T_sim', type=float, default=3, help='Simulation time')
# Muscle parameter
control_param_parser.add_argument('--gsmo', action=BooleanOptionalAction, default=True, help='If true, muscles have a gradual spatial onset at the head and tale')
control_param_parser.add_argument('--Ds_h', type=float, default=0.01, help='Sigmoid slope at the head')
control_param_parser.add_argument('--Ds_t', type=float, default=0.01, help='Sigmoid slope at the tale')
control_param_parser.add_argument('--s0_h', type=float, default=5 * 0.01, help='Sigmoid midpoint at the head')
control_param_parser.add_argument('--s0_t', type=float, default=1 - 5 * 0.01, help='Sigmoid midpoint at the tale')
# Muscle timescale
control_param_parser.add_argument('--fmts', action=BooleanOptionalAction, default=True, help='If true, muscles switch on on a finite time scale')
control_param_parser.add_argument('--tau_on', type=float, default=0.1, help='Muscle time scale')
control_param_parser.add_argument('--t_on', type=float, default=5 * 0.1, help='Sigmoid midpoint')

