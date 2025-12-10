import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from iro.aggregation.aggregators import AggregationFunction as aggregation_function

# this is a simple example script to visualize different risk aggregation functions

def generate_sample_risks(num_domains=100, distribution='lognormal'):
    """
    Generates a sample distribution of risks.
    """
    if distribution == 'normal':
        risks = np.random.normal(loc=1.5, scale=0.5, size=num_domains)
        risks = np.abs(risks) + np.random.uniform(0, 0.5, size=num_domains)
    elif distribution == 'lognormal': 
        risks = np.random.lognormal(mean=0.5, sigma=0.4, size=num_domains)
    else:
        raise ValueError("Unsupported distribution type.")
    
    return torch.from_numpy(risks).float()

def visualize_aggregations(risk_distribution, aggregation_functions):
    """
    Visualizes the distribution of risks and aggregated values by plotting
    each distribution separately in a loop.
    
    Parameters
    ----------
    risk_distribution : torch.Tensor
        The tensor of risk values from different domains.
    aggregation_functions : dict
        A dictionary where keys are function names and values are dicts
        containing the parameter name and a list of its values.
    """
    sns.set_style("whitegrid")
    
    for func_name, params in aggregation_functions.items():
        param_name = params['name']
        param_values = params['values']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Use a colormap to get colors for each parameter value
        cmap = plt.cm.get_cmap("RdBu_r")
        norm = plt.Normalize(vmin=min(param_values), vmax=max(param_values))

        # Loop to plot each KDE and vertical line
        for param_value in tqdm(param_values, desc=f"Plotting {func_name}"):
            color = cmap(norm(param_value))

            # Generate a new, slightly different set of risks for each iteration
            # This simulates the original notebook code from https://github.com/muandet-lab/dgil
            # where risks were calculated for a different model each time.
            current_risks = generate_sample_risks(num_domains=100, distribution='lognormal')
            
            # Plot the distribution
            sns.kdeplot(current_risks.numpy(), ax=ax, color=color, alpha=0.6,
                        label=f'{param_value:.2f}')
            
            # Calculate and plot the aggregated risk as a vertical line
            aggregator = aggregation_function(name=func_name)
            agg_kwargs = {param_name: param_value}
            
            if func_name == "soft_cvar":
                agg_kwargs = {'alpha': param_value, 'eta': 1.0}
            
            try:
                aggregated_risk = aggregator.aggregate(current_risks, **agg_kwargs)
                ax.axvline(aggregated_risk.item(), linestyle='--', color=color, 
                           zorder=2, alpha=0.8)
            except NotImplementedError as e:
                print(f"Skipping {func_name}: {e}")
                continue
        
        # Add a colorbar and remove the seaborn-generated legend
        ax.legend(title=f'${param_name}$', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.get_legend().remove()
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label=f'${param_name}$', size=12, labelpad=5)
        
        # Customize the plot
        ax.set_title(f'Distribution of Risks on Training Data for {func_name.upper()}')
        ax.set_xlabel('Risk Value')
        ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    risk_data = generate_sample_risks(num_domains=100, distribution='lognormal')

    aggregation_functions_to_plot = {
        'cvar': {'name': 'alpha', 'values': np.linspace(0.5, 0.95, 10)},
        'var': {'name': 'alpha', 'values': np.linspace(0.5, 0.95, 10)},
        'ph': {'name': 'xi', 'values': np.linspace(0.1, 2.0, 10)},
        'entropic': {'name': 'eta', 'values': np.linspace(0.1, 10.0, 10)},
        'soft_cvar': {'name': 'alpha', 'values': np.linspace(0.5, 0.95, 10)}
    }
    
    visualize_aggregations(risk_data, aggregation_functions_to_plot)