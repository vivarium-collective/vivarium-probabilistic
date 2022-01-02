"""
=====================
Probabilistic Wrapper
=====================
"""
import numpy as np
import matplotlib.pyplot as plt

# from torch.distributions.normal import Normal
import torch
from torch import Tensor
from vivarium.core.process import Process
from vivarium.core.store import Store
from vivarium_probabilistic.processes.ode_process import (
    ODE, arrays_from, get_repressilator_config)
from vivarium.core.composition import simulate_process


def convert_schema_probabilistic(schema):
    """convert port schema to pytorch distributions"""
    probabilistic_schema = {}
    for k, v in schema.items():
        if isinstance(v, dict):
            if set(Store.schema_keys).intersection(v.keys()):
                default = v.get('_default', 0.0)
                probabilistic_schema[k] = v
                probabilistic_schema[k]['_default'] = \
                    torch.empty(10).normal_(mean=default, std=1.0)
            else:
                probabilistic_schema[k] = convert_schema_probabilistic(v)
        else:
            probabilistic_schema[k] = v

    return probabilistic_schema


class ProbabilisticWrapper(Process):
    defaults = {
        'process': None,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']
        # TODO -- get process parameters?

    def ports_schema(self):
        schema = self.process.ports_schema()
        probabilistic_schema = convert_schema_probabilistic(schema)
        return {
            'process': schema,
            'prior': probabilistic_schema,
        }

    def next_update(self, timestep, states):
        input_sample = self.sample(states['prior'])
        update = self.process.next_update(timestep, input_sample)
        return {
            'process': update,
            'prior': {}  # TODO -- update the prior?
        }

    def sample(self, states):
        sample = {}
        for k, v in states.items():
            if isinstance(v, dict):
                sample[k] = self.sample(v)
            elif isinstance(v, Tensor):
                sample[k] = np.random.choice(v)
            else:
                Exception(f"value {v} unexpected")
        return sample

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
        # 'initial_state': {
        #     'variables': initial_state,
        # },
        'return_raw_data': True,
    }
    output = simulate_process(probabilistic_process, sim_config)

    import ipdb; ipdb.set_trace()

    # # transform and plot
    # variable_ids = list(initial_state.keys())
    # results = None
    # time = np.array([t for t in output.keys()])
    # for state in output.values():
    #     array = arrays_from(state['variables'], variable_ids)
    #     if results is None:
    #         results = array
    #     else:
    #         results = np.vstack((results, array))
    # # plot
    # plt.plot(time, results[:, 0], time, results[:, 1], time, results[:, 2])
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.show()


# python vivarium_probabilistic/processes/probabilistic_wrapper.py
if __name__ == '__main__':
    test_probwrapper()
