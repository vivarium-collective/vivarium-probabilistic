"""
====================================
Probabilistic Wrapper and Subclasses
====================================

Subclasses:
 - ImportanceSampler
 - MarkovChainMonteCarlo
"""

import abc
import math
import numpy as np
import copy
import matplotlib.pyplot as plt

from vivarium.core.process import Process
from vivarium.core.store import Store
from vivarium.core.engine import Engine
from vivarium.plots.simulation_output import _save_fig_to_dir
from vivarium_probabilistic.processes.ode_process import (
    ODE, arrays_from, get_repressilator_config)


def get_variables_from_schema(schema, emit=True, updater=None):
    """return a map from port to variable ids"""
    variables = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            if set(Store.schema_keys).intersection(value.keys()):
                default = value.get('_default', 0.0)
                variables[key] = {
                    '_default': default,
                    '_emit': emit,
                }
                if updater:
                    variables[key]['_updater'] = updater
            else:
                variables[key] = get_variables_from_schema(
                    value, emit=emit, updater=updater)
        else:
            variables[key] = value
    # variable_values = list(variables.values())
    # if all(v is None for v in variable_values):
    #     variables = list(variables.keys())
    return variables


def sample_normal_parameters(parameters, std_dev=1.0):
    new_parameters = {}
    for param_id, mean in parameters.items():
        new_parameters[param_id] = np.random.normal(mean, scale=std_dev)
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
        'std_dev': 1.0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # make the input process
        process_class = self.parameters['process']
        process_config = copy.deepcopy(self.parameters['process_config'])
        self.input_process = process_class(process_config)

        # make variations of the input processes
        self.process_ids = [
            str(idx) for idx in range(self.parameters['number_of_samples'])]
        self.processes = {}
        self.priors = {}
        for process_id in self.process_ids:
            # sample new parameters
            process_parameters = sample_normal_parameters(
                process_config['parameters'],
                self.parameters['std_dev'])
            process_config['parameters'] = process_parameters

            # save the sampled parameters, and make a process instance
            self.priors[process_id] = process_parameters
            self.processes[process_id] = process_class(process_config)

        # get the variable ids from the process
        self.process_schema = self.input_process.ports_schema()
        self.observables = get_variables_from_schema(self.process_schema)

    def ports_schema(self):
        sample_schema = get_variables_from_schema(
            self.process_schema)
        weight_schema = get_variables_from_schema(
            self.process_schema, emit=True, updater='set')
        schema = {
            'process_states': {
                process_id: self.process_schema
                for process_id, process in self.processes.items()
            },
            'parameter_samples': {
                process_id: sample_schema for process_id in self.process_ids
            },
            'parameter_weights': {
                process_id: weight_schema for process_id in self.process_ids
            },
            'observations': self.observables,
        }
        return schema

    @abc.abstractmethod
    def next_update(self, timestep, states):
        return {}


class ImportanceSampler(ProbabilisticWrapper):
    """Performs Importance Sampling"""

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def update_weights(self, prediction, observation, weights):

        updated_weights = copy.deepcopy(weights)
        if isinstance(prediction, dict):
            for variable_id, variable_prediction in prediction.items():
                variable_observation = observation[variable_id]
                variable_weights = weights[variable_id]
                updated_weights[variable_id] = self.update_weights(
                    variable_prediction, variable_observation, variable_weights)
        else:
            if math.isnan(weights):
                weights = 0.0
            error = (prediction - observation) ** 2
            updated_weights = weights + error
        return updated_weights

    def next_update(self, timestep, states):

        # read variables through the ports
        process_states = states['process_states']
        parameter_weights = states['parameter_weights']
        observations = states['observations']

        # run multiple copies of the process
        # TODO -- this should utilize Vivarium parallelization
        process_update = {}
        weights_update = {}
        for process_id, process in self.processes.items():
            weights = parameter_weights[process_id]

            # run the process with the previous final state
            process_state = process_states[process_id]
            update = process.next_update(timestep, process_state)
            process_update[process_id] = update

            # compare to observations and update the weights
            new_weights = self.update_weights(update, observations, weights)
            weights_update[process_id] = new_weights

        return {
            'process_states': process_update,
            'parameter_weights': weights_update,
        }


class MarkovChainMonteCarlo(ProbabilisticWrapper):
    """
    TODO: Perform Markov Chain Monte Carlo
    """


def test_importance_sampling(
        total_time=100.0,
        time_step=1.0,
        number_of_samples=10,
):
    # make a "ground truth" repressilator ODE process
    repressilator_config, initial_state = get_repressilator_config(
        time_step=time_step)
    repressilator_process = ODE(repressilator_config)

    # make the probabilistic wrapper process
    probabilistic_config = {
        'process': ODE,
        'process_config': repressilator_config,
        'number_of_samples': number_of_samples,
        'time_step': time_step,
        'std_dev': 0.5,
    }
    probabilistic_process = ImportanceSampler(probabilistic_config)
    process_ids = probabilistic_process.process_ids

    # configure a simulation
    sim = Engine(
        processes={
            'probabilistic_process': probabilistic_process,
            'ground_truth_process': repressilator_process,
        },
        topology={
            'probabilistic_process': {
                'process_states': ('process_states',),
                'parameter_samples': ('parameter_samples',),
                'parameter_weights': ('parameter_weights',),
                'observations': ('expected',),
            },
            'ground_truth_process': {
                'variables':  ('expected', 'variables',),
            },
        },
        initial_state={
            'expected': initial_state,
            'process_states': {
                process_id: initial_state
                for process_id in process_ids}
        },
    )

    # run the simulation
    sim.update(total_time)

    # retrieve data, transform, and plot
    timeseries = sim.emitter.get_timeseries()
    plot_process_output(
        timeseries,
        process_ids,
        out_dir='out/',
        filename='importance_sampling')


def plot_process_output(
        timeseries,
        process_ids=[],
        out_dir=None,
        filename='plot'):
    time_vec = timeseries['time']

    # make figure and plot
    n_rows = len(process_ids)
    n_cols = 4
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 1.5))
    grid = plt.GridSpec(n_rows, n_cols)

    # plot observed process
    observed = timeseries['expected']['variables']
    for var_id, var_timeseries in observed.items():
        ax0 = fig.add_subplot(grid[0, 0])
        ax0.plot(time_vec, var_timeseries, label=var_id)
        ax0.set_title('expected output')

    # plot sampled processes
    row_idx = 0
    for process_id in process_ids:
        ax1 = fig.add_subplot(grid[row_idx, 1])
        ax2 = fig.add_subplot(grid[row_idx, 2])
        ax3 = fig.add_subplot(grid[row_idx, 3])

        process_timeseries = timeseries['process_states'][process_id]['variables']
        process_weights = timeseries['parameter_weights'][process_id]['variables']

        for var_id, var_timeseries in process_timeseries.items():
            var_weights = process_weights[var_id]
            ax2.plot(time_vec, var_timeseries, label=var_id)
            ax3.plot(time_vec, var_weights, label=var_id)

        # adjust
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

        # add text
        ax1.text(0.6, 0.5, f'process {process_id}')
        ax1.axis('off')
        if row_idx == 0:
            ax2.set_title('simulation output')
            ax3.set_title('parameter weights')

        row_idx += 1

    if out_dir:
        _save_fig_to_dir(fig, filename, out_dir)
    return fig


# python vivarium_probabilistic/processes/probabilistic_wrapper.py
if __name__ == '__main__':
    test_importance_sampling()
