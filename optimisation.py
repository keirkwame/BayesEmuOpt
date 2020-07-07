import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy.optimize as spo

def map_from_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector from the unit cube to the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map. Should all be [0,1]
    param_limits - the maximal limits of the parameters to choose.
    #Credit to Simeon Bird
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all((param_vec >= 0)*(param_vec <= 1))
    assert np.all(param_limits[:,0] <= param_limits[:,1])
    new_params = param_limits[:,0] + param_vec*(param_limits[:,1] - param_limits[:,0])
    assert np.all(new_params <= param_limits[:,1])
    assert np.all(new_params >= param_limits[:,0])
    return new_params

def map_to_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector to the unit cube from the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map.
    param_limits - the limits of the allowed parameters.
    Returns:
    vector of parameters, all in [0,1].
    #Credit to Simeon Bird
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all(param_vec-1e-16 <= param_limits[:,1])
    assert np.all(param_vec+1e-16 >= param_limits[:,0])
    ii = np.where(param_vec > param_limits[:,1])
    param_vec[ii] = param_limits[ii,1]
    ii = np.where(param_vec < param_limits[:,0])
    param_vec[ii] = param_limits[ii,0]
    assert np.all(param_limits[:,0] <= param_limits[:,1])
    new_params = (param_vec-param_limits[:,0])/(param_limits[:,1] - param_limits[:,0])
    assert np.all((new_params >= 0)*(new_params <= 1))
    return new_params

def map_to_unit_cube_list(param_vec_list, param_limits):
    """Map multiple parameter vectors to the unit cube"""
    return np.array([map_to_unit_cube(param_vec, param_limits) for param_vec in param_vec_list])

def map_from_unit_cube_list(param_vec_list, param_limits):
    """Map multiple parameter vectors back from the unit cube"""
    return np.array([map_from_unit_cube(param_vec, param_limits) for param_vec in param_vec_list])


class OptimisationClass:
    """Class to contain Bayesian optimisation computations."""
    def __init__(self, get_objective, get_emulator_error, param_limits, inverse_data_covariance):
        self.get_objective = get_objective
        self.get_emulator_error = get_emulator_error
        self.param_limits = param_limits
        self.inverse_data_covariance = inverse_data_covariance

    def exploration_weight_GP_UCB(self, nu, delta=0.5):
        """Choose the exploration weight for the GP-UCB acquisition function."""
        assert 0. < delta < 1.
        return np.sqrt(nu * 2. * np.log((np.pi ** 2.) / 3. / delta))

    def exploration_GP_UCB(self, params, nu, **kwargs):
        """Evaluate the exploration term of the GP-UCB acquisition function."""
        emulator_error = self.get_emulator_error(params)
        return self.exploration_weight_GP_UCB(nu, **kwargs) * np.dot(emulator_error,
                                                                np.dot(self.inverse_data_covariance, emulator_error))

    def exploitation_GP_UCB(self, params):
        """Evaluate the exploitation term of the GP-UCB acquisition function."""
        return self.get_objective(params)

    def acquisition_GP_UCB(self, params, nu, **kwargs):
        """Evaluate the modified GP-UCB acquisition function."""
        return self.exploitation_GP_UCB(params) + self.exploration_GP_UCB(params, nu, **kwargs)

    def optimise_acquisition_function(self, params_start='default', nu=0.19, acquisition='GP_UCB', bounds='default',
                                      method='TNC', **kwargs):
        """Find parameter vector at maximum of acquisition function."""
        if acquisition == 'GP_UCB':
            acquisition = lambda params: -1. * self.acquisition_GP_UCB(map_from_unit_cube(params, self.param_limits),
                                                                       nu, **kwargs)
        else:
            raise ValueError('Unsupported acquisition function.')

        if bounds == 'default':
            bounds = [(0.05, 0.95) for i in range(params_start.shape[0])]

        if params_start == 'default':
            params_start = np.ones(self.param_limits.shape[0]) * 0.5
        else:
            params_start = map_to_unit_cube(params_start, self.param_limits)

        acquisition_max = spo.minimize(acquisition, params_start, method=method, bounds=bounds)
        return map_from_unit_cube(acquisition_max, self.param_limits)

    def make_proposal(self, params_start='default', nu=0.19, std_dev=None, acquisition='GP_UCB', bounds='default',
                      method='TNC', **kwargs):
        """Make a proposal for the next optimisation point."""
        if std_dev == None:
            displacement = np.zeros(self.param_limits.shape[0])
        else:
            displacement = npr.normal(scale=std_dev)
        acquisition_max = self.optimise_acquisition_function(params_start=params_start, nu=nu, acquisition=acquisition,
                                                             bounds=bounds, method=method, **kwargs)
        proposal = acquisition_max + displacement
        npt.assert_array_less(proposal, self.param_limits[:, 1],
                              err_msg='Proposal greater than parameter limits -- try again!')
        npt.assert_array_less(self.param_limits[:, 0], proposal,
                              err_msg='Proposal less than parameter limits -- try again!')
        return proposal
