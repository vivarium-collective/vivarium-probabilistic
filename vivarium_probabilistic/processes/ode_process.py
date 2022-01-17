"""
===========
ODE Process
===========
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process


def arrays_from(ds, keys):
    return np.array([ds[key] for key in keys])

def arrays_to(n, keys):
    return {key: n[idx] for idx, key in enumerate(keys)}


class ODE(Process):
    defaults = {
        'system_generator': None,
        'parameters': {},
        'variable_ids': [],
        'internal_dt': 1e-2,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # make the system from the declared parameters and generator
        system_generator = self.parameters['system_generator']
        parameters = self.parameters['parameters']
        self.system = system_generator(**parameters)
        self.variable_ids = self.parameters['variable_ids']
        self.dt = self.parameters['internal_dt']

    def set_parameters(self, parameters):
        # TODO -- expose parameters?
        pass

    def ports_schema(self):
        """
        put the variables under a "variables" port, declare their schema
        """
        return {
            'variables': {
                var_id: {
                    '_default': 0.0,
                    '_emit': True,
                    '_updater': 'set',
                } for var_id in self.variable_ids
            }
        }

    def next_update(self, timestep, states):

        # get the system, the inputs, and the internal timestep
        inputs = states['variables']

        # put variables in array
        state_array = arrays_from(inputs, self.variable_ids)
        time = np.linspace(0.0, timestep, int(timestep/self.dt))

        # solve odes
        output = odeint(self.system, state_array, time)

        # return results
        final_state = output[-1, :]
        results = arrays_to(final_state, self.variable_ids)
        return {'variables': results}



def get_repressilator(
        beta=0.5,
        alpha0=0,
        alpha=100,
        n=2,
):
    def repressilator(var, t):
        m = var[:3]
        p = var[3:]
        dm0 = - m[0] + alpha / (1 + p[2] ** n) + alpha0
        dm1 = - m[1] + alpha / (1 + p[0] ** n) + alpha0
        dm2 = - m[2] + alpha / (1 + p[1] ** n) + alpha0
        dp0 = - beta * (p[0] - m[0])
        dp1 = - beta * (p[1] - m[1])
        dp2 = - beta * (p[2] - m[2])
        return [dm0, dm1, dm2, dp0, dp1, dp2]
    return repressilator


def get_repressilator_config(time_step=1.0):
    # make the ode process configuration
    variable_ids = ['m0', 'm1', 'm2', 'p0', 'p1', 'p2']
    parameters = {
        'beta': 0.5,
        'alpha0': 0,
        'alpha': 100,
        'n': 2,
    }
    ode_config = {
        'system_generator': get_repressilator,
        'variable_ids': variable_ids,
        'parameters': parameters,
        'time_step': time_step,
    }
    # declare the initial state
    initial_state = {
        'variables': {
            'm0': 1.0,
            'm1': 4.0,
            'm2': 1.0,
            'p0': 2.0,
            'p1': 1.0,
            'p2': 1.0,
        }
    }
    return ode_config, initial_state


def main(total_time=100.0):
    ode_config, initial_state = get_repressilator_config()
    process = ODE(ode_config)

    # run simulation
    sim_config = {
        'total_time': total_time,
        'initial_state': initial_state,
        'return_raw_data': True,
    }
    output = simulate_process(process, sim_config)

    # transform and plot
    variable_ids = list(initial_state['variables'].keys())
    results = None
    time = np.array([t for t in output.keys()])
    for state in output.values():
        array = arrays_from(state['variables'], variable_ids)
        if results is None:
            results = array
        else:
            results = np.vstack((results, array))

    # # run repressilator directly
    # time = np.linspace(0.0, total_time, 1000)
    # minit = np.array(list(initial_state.values()))
    # results = odeint(repressilator, minit, time)

    # plot
    plt.plot(time, results[:, 0], time, results[:, 1], time, results[:, 2])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()


# python vivarium_probabilistic/processes/ode_process.py
if __name__ == '__main__':
    main()
