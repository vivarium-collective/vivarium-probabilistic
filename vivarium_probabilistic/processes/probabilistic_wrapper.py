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
    """
    Reads the Store of samples
    Updates the the parameter weights
    """

    defaults = {
        'process': None,
        'process_config': {},
        'number_of_samples': 1,
        'observations': {},  # ground truth
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # make the input process object
        process_class = self.parameters['process']
        initial_parameters = self.parameters['process_config']

        self.processes = [
            process_class(initial_parameters)
            for _ in range(self.parameters['number_of_samples'])]

        # TODO -- get process parameters?

    def ports_schema(self):
        schema = self.process.ports_schema()
        probabilistic_schema = convert_schema_probabilistic(schema)
        new_schema = {
            'process_states': {
                idx: schema for idx in range(self.parameters['number_of_samples'])
            },
            # one parameter set for each process in self.processes
            'parameter_samples': {
                idx: {} for idx in range(self.parameters['number_of_samples'])
            },
            'parameter_weights': {
                idx: {} for idx in range(self.parameters['number_of_samples'])
            },
            'observations': {},
        }
        return new_schema

    def sample_parameters(self):
        # TODO put the parameters in a store
        return {}

    def update_weights(self, prediction, observation, weights):
        # TODO -- recursive on process state
        error = (prediction - observation) ** 2
        updated_weights = weights + error
        return updated_weights

    def next_update(self, timestep, states):

        # read parameters from the store
        process_states = states['process_states']
        parameter_samples = states['parameter_samples']
        parameter_weights = states['parameter_weights']
        observations = states['observations']

        # run multiple copies of the process,
        process_update = {}
        weights_update = {}
        for idx, process in enumerate(self.processes):
            # parameters = parameter_samples[idx] # TODO -- parameters only once in constructor
            weights = parameter_weights[idx]

            # update the process with the parameters. TODO -- need a ProbabilisticWrapper method for this.


            # run the process with the previous final state
            process_state = process_states[idx]
            update = process.next_update(timestep, process_state)
            process_update[idx] = update

            # update the weights
            # compare to observations
            new_weights = self.update_weights(update, observations, weights)
            weights_update[idx] = new_weights

        return {
            'process_states': process_update,
            'parameter_weights': weights_update,
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
        'observations': {}  # TODO -- put this in. array from underlying repressilator
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
