"""
=====================
Probabilistic Wrapper
=====================
"""
import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.process import Process
from vivarium_probabilistic.processes.ode_process import ODE, arrays_from, get_repressilator
from vivarium.core.composition import simulate_process


class ProbabilisticWrapper(Process):
    defaults = {
        'process': None,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']

    def ports_schema(self):
        ports_schema = self.process.ports_schema()
        return ports_schema

    def next_update(self, timestep, states):
        update = self.process.next_update(timestep, states)
        return update


def main(total_time=100.0):

    # make a repressilator ODE function
    repressilator = get_repressilator(
        beta=0.5,
        alpha0=0,
        alpha=100,
        n=2,
    )

    # make the ode process
    variables = [
        'm0', 'm1', 'm2', 'p0', 'p1', 'p2']
    ode_config = {
        'system': repressilator,
        'variables': variables,
        'time_step': 0.1,
    }
    process = ODE(ode_config)

    # make the probabilistic wrapper process
    probabilistic_config = {
        'process': process,
    }
    probabilistic_process = ProbabilisticWrapper(probabilistic_config)

    # declare the initial state
    initial_state = {
        'm0': 1.0,
        'm1': 4.0,
        'm2': 1.0,
        'p0': 2.0,
        'p1': 1.0,
        'p2': 1.0,
    }

    # run simulation
    sim_config = {
        'total_time': total_time,
        'initial_state': {
            'variables': initial_state,
        },
        'return_raw_data': True,
    }
    output = simulate_process(probabilistic_process, sim_config)

    # transform and plot
    results = None
    time = np.array([t for t in output.keys()])
    for state in output.values():
        array = arrays_from(state['variables'], variables)
        if results is None:
            results = array
        else:
            results = np.vstack((results, array))

    # plot
    plt.plot(time, results[:, 0], time, results[:, 1], time, results[:, 2])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()


# python vivarium_probabilistic/processes/probabilistic_wrapper.py
if __name__ == '__main__':
    main()
