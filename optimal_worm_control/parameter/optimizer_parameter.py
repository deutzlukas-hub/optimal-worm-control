from argparse import ArgumentParser, BooleanOptionalAction

optimizer_param_parser = ArgumentParser()
optimizer_param_parser.add_argument('--T_sim', type=float, default=3, help='Simulation time')

optimizer_param_parser.add_argument('--minimize', action=BooleanOptionalAction, default=True, help='If true, minimize objective otherwise maximize')
optimizer_param_parser.add_argument('--normalize', action=BooleanOptionalAction, default=False, help='If true, normalize objetive')

