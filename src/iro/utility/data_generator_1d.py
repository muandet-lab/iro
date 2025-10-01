import numpy as np

class data_generator_1D:
    
    """ This class generates the 1D simulation data. """
    

    def __init__(self, num_envs = 250, size = (100,1), theta="uniform", experiment="1D_linear"):
        
        """ Construct the aggregation function.
        
        Parameters
        -----------
        experiment : str, optional
            the name of the experimental setting (whether we want f to be linear n X or non-linear)
        num_envs : int, optional
            the number of environments
        size : tuple, optional
            the size of the datset (number of observations, number of features)
        theta : str, optional
            the distrubution from which we sample the true thetas for each environment
        """
        
        self.num_envs = num_envs
        self.size = size
        self.theta = theta
        self.experiment = experiment
        self.description = f"{theta}_{experiment}"
    
    def generate(self, f) -> dict:           
        """Aggregates a list of risks. 

        Returns
        ------
        env_dict : dict
        Examples
        --------
        >>> data_generator_1D(num_envs = 250, size = (100,1), theta="uniform", experiment="1D_linear").generate()
        """
        env_list = [f'e_{i}' for i in range(1,self.num_envs+1,1)]
        env_dict = dict(list(enumerate(env_list)))
        for e in env_dict.keys():

            theta_true=np.random.uniform(0,1) if self.theta == "uniform" else np.random.beta(0.1,0.2)

            x=np.random.normal(loc=1, scale=0.5, size=self.size)
            noise=np.random.normal(loc=0, scale=0.05, size=self.size)
            y = f(x,theta_true)+noise

            env_dict[e] = {'x': x,'y': y,'theta_true': theta_true, 'env': [e]*self.size[0]}

        return env_dict