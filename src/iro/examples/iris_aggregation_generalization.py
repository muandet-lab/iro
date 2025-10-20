import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Direct Import (Assumes iro package is installed) ---
from iro.aggregation.aggregators import AggregationFunction as aggregation_function

# --- CONFIGURATION ---
NUM_DOMAINS = 100  # Large number of domains
SAMPLES_PER_DOMAIN = 10 # Small number of samples per domain
RISKS_TO_AGGREGATE = 100 # How many risks (from domains) to consider for aggregation
MAX_CLASSES = 3 # Hardcoding max number of classes for Iris (0, 1, 2) to fix IndexError

# --- 1. DATA GENERATION AND DOMAIN SIMULATION ---

def generate_domain_data(iris_data, n_samples, domain_id):
    """
    Simulates domain shift by applying a unique, subtle perturbation to features.
    
    Returns: X (features), y (targets) for a single domain.
    """
    X_original, y = iris_data.data, iris_data.target
    
    # Use a small subset of the total data
    X_sub, _, y_sub, _ = train_test_split(X_original, y, train_size=n_samples, random_state=domain_id)
    
    # Introduce domain shift (simulating covariance or mean shift)
    shift_factor = np.sin(domain_id / 10.0) * 0.1  # Unique shift based on domain_id
    noise = np.random.normal(loc=shift_factor, scale=0.05, size=X_sub.shape)
    X_shifted = X_sub + noise
    
    # Add a global domain property feature (not used for training, but useful conceptually)
    # X_shifted = np.concatenate([X_shifted, np.full((n_samples, 1), shift_factor)], axis=1)

    return X_shifted, y_sub

def load_and_split_data(n_domains, n_samples):
    """Loads Iris data and generates domain-specific splits."""
    iris = load_iris()
    domain_data = {}
    for i in range(n_domains):
        X, y = generate_domain_data(iris, n_samples, i)
        domain_data[f'D{i}'] = {'X': X, 'y': y}
    return domain_data

# --- 2. MODEL TRAINING AND RISK CALCULATION ---

def train_domain_models(domain_data):
    """Trains a simple Logistic Regression model for each domain."""
    domain_models = {}
    for domain_name, data in tqdm(domain_data.items(), desc="Training Domain Models"):
        model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
        model.fit(data['X'], data['y'])
        domain_models[domain_name] = model
    return domain_models

def calculate_risk_matrix(domain_data, domain_models):
    """
    Calculates the risk matrix R where R[i, j] is the risk (MSE)
    of model_i on data from domain_j.
    """
    domain_names = list(domain_data.keys())
    num_domains = len(domain_names)
    risk_matrix = np.zeros((num_domains, num_domains))
    
    for i in tqdm(range(num_domains), desc="Calculating Risk Matrix"):
        model_i = domain_models[domain_names[i]]
        for j in range(num_domains):
            data_j = domain_data[domain_names[j]]
            
            # Use probabilities for Logistic Regression
            y_pred_proba = model_i.predict_proba(data_j['X'])
            
            # --- FIXED: PADDING PREDICT_PROBA TO MATCH MAX_CLASSES ---
            num_samples_j = data_j['X'].shape[0]
            # Initialize a full prediction matrix of size (N_samples, 3) with zeros
            y_pred_full = np.zeros((num_samples_j, MAX_CLASSES))
            
            # Get the indices (0, 1, or 2) that the model actually predicted
            predicted_class_indices = model_i.classes_.astype(int)
            
            # Copy the predictions into the correct columns, leaving missing classes as 0
            y_pred_full[:, predicted_class_indices] = y_pred_proba
            
            # Calculate True One-Hot Labels (size 3)
            y_true_one_hot = np.eye(MAX_CLASSES)[data_j['y']]
            
            # Now both y_true_one_hot and y_pred_full have the same number of columns (MAX_CLASSES)
            risk_matrix[i, j] = mean_squared_error(y_true_one_hot, y_pred_full)
            
    # Convert to torch tensor for aggregation
    return torch.from_numpy(risk_matrix).float()

# --- 3. VISUALIZATION OF AGGREGATION ---

def visualize_aggregations(risk_matrix, aggregation_functions):
    """
    Visualizes the distribution of *a single model's* risks across all domains,
    and the effect of different parameter levels on the aggregation.
    
    We choose the risks for model_0 (R[0, :]) to visualize how aggregation works
    on a vector of risks.
    """
    sns.set_style("whitegrid")
    
    # Choose a single model's performance vector (risks from model 0 across all domains)
    # This represents the 'training domain risks' for a hypothetical model trained on D0
    base_risks = risk_matrix[0, :RISKS_TO_AGGREGATE] 
    
    for func_name, params in aggregation_functions.items():
        param_name = params['name']
        param_values = params['values']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Use a colormap to get colors for each parameter value
        cmap = plt.cm.get_cmap("RdBu_r")
        norm = plt.Normalize(vmin=min(param_values), vmax=max(param_values))

        # Loop to plot each KDE and vertical line
        
        # Plot the base KDE once for a clear, filled background
        sns.kdeplot(base_risks.numpy(), ax=ax, color='skyblue', fill=True, alpha=0.3, linewidth=0)


        for param_value in tqdm(param_values, desc=f"Plotting {func_name}"):
            color = cmap(norm(param_value))
            
            # Use the single risk vector for aggregation
            aggregator = aggregation_function(name=func_name) 
            agg_kwargs = {param_name: param_value}
            
            if func_name == "soft_cvar":
                agg_kwargs = {'alpha': param_value, 'eta': 1.0}
            
            try:
                aggregated_risk = aggregator.aggregate(base_risks, **agg_kwargs)
                
                # Plot the vertical line corresponding to the aggregated risk
                ax.axvline(aggregated_risk.item(), linestyle='--', color=color, 
                           label=f'{param_value:.2f}', zorder=2, alpha=0.8, linewidth=2)
            except NotImplementedError as e:
                print(f"Skipping {func_name}: {e}")
                continue
        
        # Add a colorbar and remove the seaborn-generated legend
        # We plot a dummy legend item to ensure the colorbar works
        ax.legend(title=f'${param_name}$', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.get_legend().remove()
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label=f'${param_name}$', size=12, labelpad=5)
        
        # Customize the plot
        ax.set_title(f'Aggregated Risk for {func_name.upper()} on Model 0 Risks')
        ax.set_xlabel('Risk Value (MSE on Target Domain)')
        ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # 1. Load data and simulate domains
    domain_data = load_and_split_data(NUM_DOMAINS, SAMPLES_PER_DOMAIN)
    
    # 2. Train models
    domain_models = train_domain_models(domain_data)
    
    # 3. Calculate risk matrix
    # This matrix is the core input for the aggregation experiment
    risk_matrix_tensor = calculate_risk_matrix(domain_data, domain_models)

    # 4. Define the aggregation functions and their parameter ranges
    aggregation_functions_to_plot = {
        'cvar': {'name': 'alpha', 'values': np.linspace(0.5, 0.99, 10)},
        'var': {'name': 'alpha', 'values': np.linspace(0.5, 0.99, 10)},
        'ph': {'name': 'xi', 'values': np.linspace(0.1, 5.0, 10)},
        'entropic': {'name': 'eta', 'values': np.linspace(0.1, 10.0, 10)},
        'soft_cvar': {'name': 'alpha', 'values': np.linspace(0.5, 0.99, 10)}
    }
    
    # 5. Visualize the aggregations on the risks generated from this experiment
    visualize_aggregations(risk_matrix_tensor, aggregation_functions_to_plot)
