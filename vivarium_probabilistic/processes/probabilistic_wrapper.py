"""
=====================
Probabilistic Wrapper
=====================
"""
import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.process import Process
from vivarium_probabilistic.processes.ode_process import (
    ODE, arrays_from, get_repressilator_config)
from vivarium.core.composition import simulate_process


class ProbabilisticWrapper(Process):
    defaults = {
        'process': None,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']
        self.process_ports = self.process.ports_schema()

        # TODO -- get process parameters?

    def ports_schema(self):
        # TODO -- replace the process' variables with probability distributions
        return self.process_ports

    def next_update(self, timestep, states):
        # TODO - get update from probabilistic model
        update = self.process.next_update(timestep, states)
        return update

    def sample(self):
        return None

    def observe(self):
        return None


def test_probwrapper(total_time=100.0):

    # make a repressilator ODE process
    repressilator_config, initial_state = get_repressilator_config()
    process = ODE(repressilator_config)

    # make the probabilistic wrapper process
    probabilistic_config = {
        'process': process,
    }
    probabilistic_process = ProbabilisticWrapper(probabilistic_config)

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
    variable_ids = list(initial_state.keys())
    results = None
    time = np.array([t for t in output.keys()])
    for state in output.values():
        array = arrays_from(state['variables'], variable_ids)
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
    test_probwrapper()
