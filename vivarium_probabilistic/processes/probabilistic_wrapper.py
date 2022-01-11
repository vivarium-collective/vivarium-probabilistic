"""
=====================
Probabilistic Wrapper
=====================
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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

def sample_normal_parameters(parameters):
    new_parameters = {}
    for param_id, mean in parameters.items():
        new_parameters[param_id] = np.random.normal(mean, scale=1.0)
    return new_parameters

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

        self.observations = self.parameters['observations']

        # make the input process object
        process_class = self.parameters['process']
        self.process_ids = [idx for idx in range(self.parameters['number_of_samples'])]

        # make the processes
        self.processes = {}
        self.priors = {}
        for process_id in self.process_ids:
            process_config = copy.deepcopy(self.parameters['process_config'])

            # sample new parameters
            process_parameters = sample_normal_parameters(process_config['parameters'])
            process_config['parameters'] = process_parameters

            # save the sampled parameters, and make a process instance
            self.priors[process_id] = process_parameters
            self.processes[process_id] = process_class(process_config)

    def ports_schema(self):
        # schema = self.process.ports_schema()
        # probabilistic_schema = convert_schema_probabilistic(schema)

        new_schema = {
            'process_states': {
                process_id: process.ports_schema()
                for process_id, process in self.processes.items()
            },
            # one parameter set for each process
            'parameter_samples': {
                process_id: {} for process_id in self.process_ids
            },
            'parameter_weights': {
                process_id: {} for process_id in self.process_ids
            },
        }
        return new_schema

    def update_weights(self, prediction, observation, weights):
        import ipdb; ipdb.set_trace()

        # TODO -- recursive on process state
        error = (prediction - observation) ** 2
        updated_weights = weights + error
        return updated_weights

    def next_update(self, timestep, states):

        # read parameters from the store
        process_states = states['process_states']
        # parameter_samples = states['parameter_samples']
        parameter_weights = states['parameter_weights']

        # run multiple copies of the process,
        process_update = {}
        weights_update = {}
        for process_id, process in self.processes.items():
            weights = parameter_weights[process_id]

            # run the process with the previous final state
            process_state = process_states[process_id]
            update = process.next_update(timestep, process_state)
            process_update[process_id] = update

            # compare to observations and update the weights
            new_weights = self.update_weights(update['variables'], self.observations, weights)
            weights_update[process_id] = new_weights

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


def test_probwrapper(
        total_time=100.0,
        number_of_samples=10,
):

    # make a repressilator ODE process
    repressilator_config, initial_state = get_repressilator_config()

    # run repressilator directly and use results as observation data
    system_generator = repressilator_config['system_generator']
    parameters = repressilator_config['parameters']
    repressilator = system_generator(**parameters)
    time = np.linspace(0.0, total_time, 1000)
    minit = np.array(list(initial_state.values()))
    results = odeint(repressilator, minit, time)

    # make the probabilistic wrapper process
    probabilistic_config = {
        'process': ODE,
        'process_config': repressilator_config,
        'number_of_samples': number_of_samples,
        'observations': results,
    }
    probabilistic_process = ProbabilisticWrapper(probabilistic_config)

    # make and run a simulation
    sim = Engine(
        processes={
            'probabilistic_ode': probabilistic_process},
        topology={
            'probabilistic_ode': {
                'process_states': ('process_states',),
                'parameter_samples': ('parameter_samples',),
                'parameter_weights': ('parameter_weights',),
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
