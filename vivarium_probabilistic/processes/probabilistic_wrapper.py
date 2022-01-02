"""
=====================
Probabilistic Wrapper
=====================
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

# from torch.distributions.normal import Normal
import torch
from torch import Tensor
from vivarium.core.process import Process
from vivarium.core.store import Store
from vivarium.core.engine import Engine
from vivarium_probabilistic.processes.ode_process import (
    ODE, arrays_from, get_repressilator_config)


def convert_schema_probabilistic(schema):
    """convert port schema to pytorch distributions"""
    probabilistic_schema = {}
    for k, v in schema.items():
        v = copy.deepcopy(v)
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
        'process_config': {}
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # make the input process object
        process_class = self.parameters['process']
        process_config = self.parameters['process_config']
        self.process = process_class(process_config)

        # TODO -- get process parameters?

    def ports_schema(self):
        schema = self.process.ports_schema()
        probabilistic_schema = convert_schema_probabilistic(schema)
        new_schema = {
            'process': schema,
            'priors': probabilistic_schema,
        }
        return new_schema

    def next_update(self, timestep, states):
        input_sample = self.sample(states['priors'])
        update = self.process.next_update(timestep, input_sample)
        return {
            'process': update,
            'priors': {}  # TODO -- update the prior?
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

    # make the probabilistic wrapper process
    probabilistic_config = {
        'process': ODE,
        'process_config': repressilator_config,
    }
    probabilistic_process = ProbabilisticWrapper(probabilistic_config)

    # make and run a simulation
    sim = Engine(
        processes={
            'probabilistic_ode': probabilistic_process},
        topology={
            'probabilistic_ode': {
                'process': ('process',),
                'priors': ('priors',),
            }
        },
        # initial_state=
    )
    sim.update(total_time)

    # retrieve data, transform, and plot
    data = sim.emitter.get_data()
    var_data = {t: s['process']['variables'] for t, s in data.items()}
    variable_ids = list(var_data[0.0].keys())
    time = np.array([t for t in data.keys()])
    results = None
    for state in var_data.values():
        array = arrays_from(state, variable_ids)
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
