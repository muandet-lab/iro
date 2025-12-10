import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 

# --- Direct Import (Assumes iro package is installed) ---
from iro.aggregation.aggregators import AggregationFunction as aggregation_function

# --- CONFIGURATION ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
NUM_DOMAINS = 10  # 10 domains defined by LIMIT_BAL quantiles, aligning with DomainBed structure
MAX_CLASSES = 2 # Binary classification (0: No Default, 1: Default)

# --- 1. DATA LOADING AND DOMAIN SPLITTING (DomainBed Style) ---

def load_credit_card_default_data():
    """
    Loads and preprocesses the UCI Credit Card Default dataset.
    """
    df = pd.read_excel(DATA_URL, header=1)
    df = df.drop(columns=['ID'])
    df.rename(columns={'default payment next month': 'TARGET'}, inplace=True) # Renamed to TARGET
    
    # Clean column names
    df.columns = [col.replace(' ', '_').upper() for col in df.columns]
    
    # Basic data cleaning: SEX, EDUCATION, MARRIAGE have undocumented/incorrect values
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4}) # Group unknown/other into 'other' (4)
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3}) # Group unknown into 'other' (3)
    
    return df

def create_domain_splits(df):
    """
    Splits the dataframe into 10 distinct domains (D0-D9) based on LIMIT_BAL quantiles.
    The resulting X is a DataFrame (not numpy array) for compatibility with ColumnTransformer.
    """
    # Define a new domain column based on LIMIT_BAL quantiles
    df['DOMAIN_ID'] = pd.qcut(df['LIMIT_BAL'], q=NUM_DOMAINS, labels=False, duplicates='drop')
    
    domain_data = {}
    domain_ids = df['DOMAIN_ID'].unique()
    
    for i, domain_id in enumerate(domain_ids):
        df_domain = df[df['DOMAIN_ID'] == domain_id].copy()
        
        # X is kept as a DataFrame for ColumnTransformer compatibility
        X = df_domain.drop(columns=['TARGET', 'DOMAIN_ID']) 
        y = df_domain['TARGET'].values # y remains a NumPy array
        
        domain_data[f'D{i}'] = {'X': X, 'y': y}
    
    return domain_data, domain_data

# --- 2. PREPROCESSING, MODEL TRAINING AND RISK CALCULATION ---

def setup_and_fit_preprocessor(df_full):
    """
    Creates and fits the preprocessor globally on the entire dataset.
    This ensures a consistent feature space (dimensionality) across all domains.
    """
    feature_names = df_full.drop(columns=['TARGET']).columns

    numerical_features = [
        'LIMIT_BAL', 'AGE', 
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    numerical_cols = [f for f in numerical_features if f in feature_names]
    categorical_cols = [f for f in categorical_features if f in feature_names]

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' ensures consistent column count even if a category is missing in a split
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit on the full feature data (excluding target)
    X_full = df_full.drop(columns=['TARGET'])
    preprocessor.fit(X_full)
    
    return preprocessor

def train_domain_models(train_domain_data, fitted_preprocessor):
    """
    Trains one LogisticRegression model for each domain (M_i trained on D_i) 
    after transforming the data with the globally fitted preprocessor.
    This avoids dimension issues caused by fitting preprocessors on subsets.
    """
    domain_models = {}
    
    for domain_name, data in tqdm(train_domain_data.items(), desc="Training Domain Models"):
        # 1. Transform the domain data using the globally fitted preprocessor
        X_transformed = fitted_preprocessor.transform(data['X'])
        y = data['y']
        
        # 2. Train a pure classifier on the transformed NumPy array
        model = LogisticRegression(max_iter=1000, solver='saga', random_state=42, penalty='l1')
        
        # NOTE: The ConvergenceWarning is common with L1 and SAGA on this dataset
        model.fit(X_transformed, y)
        domain_models[f'M{domain_name[1:]}'] = model # Store the fitted LogisticRegression
        
    return domain_models

def calculate_risk_matrix(target_domain_data, domain_models, fitted_preprocessor):
    """
    Calculates the risk matrix R where R[i, j] is the risk (MSE)
    of model_i on data from target_domain_j, after transforming the data.
    """
    model_names = list(domain_models.keys())
    target_names = list(target_domain_data.keys())
    num_models = len(model_names)
    num_targets = len(target_names)
    risk_matrix = np.zeros((num_models, num_targets))
    
    for i in tqdm(range(num_models), desc="Calculating Risk Matrix"):
        model_i = domain_models[model_names[i]]
        for j in range(num_targets):
            data_j = target_domain_data[target_names[j]]
            
            # 1. Transform the target data using the globally fitted preprocessor
            X_transformed = fitted_preprocessor.transform(data_j['X'])
            
            # 2. Predict probabilities using the pure classifier
            y_pred_proba = model_i.predict_proba(X_transformed) 
            
            # True one-hot labels
            y_true_one_hot = np.eye(MAX_CLASSES)[data_j['y']]
            
            # MSE of predicted probabilities vs one-hot targets (Expected Loss definition)
            risk_matrix[i, j] = mean_squared_error(y_true_one_hot, y_pred_proba)
            
    # Convert to torch tensor for aggregation
    return torch.from_numpy(risk_matrix).float()

def create_metrics_table(risk_matrix):
    """
    Calculates and prints a table summarizing the average performance of various 
    risk aggregation functions across all domain models.
    
    The performance metric used is the average aggregated risk (over the 9 out-of-domain targets)
    across all 10 trained models.
    """
    metrics = {
        "Mean Risk (ERM)": ('mean', {}),
        "Worst-Case Risk": ('worst_case', {}),
        "CVaR (alpha=0.9)": ('cvar', {'alpha': 0.9}),
        "Entropic Risk (eta=5.0)": ('entropic', {'eta': 5.0}),
    }

    results = {}
    num_models = risk_matrix.size(0)

    print("\n--- Calculating Aggregated Risk Metrics Across All Models ---")

    for metric_name, (func_name, kwargs) in metrics.items():
        aggregator = aggregation_function(name=func_name)
        
        # Calculate the aggregated risk for each model i using its 9 OOD risks
        model_risks = []
        for i in range(num_models):
            # Extract the risks R[i, j] excluding the self-risk R[i, i]
            base_risks = risk_matrix[i, :]
            
            # Create a mask to select out-of-domain risks
            ood_mask = torch.ones_like(base_risks, dtype=torch.bool)
            ood_mask[i] = False
            risks_for_aggregation = base_risks[ood_mask]

            try:
                # Use the out-of-domain risk vector for aggregation
                aggregated_risk = aggregator.aggregate(risks_for_aggregation, **kwargs)
                model_risks.append(aggregated_risk.item())
            except NotImplementedError:
                model_risks.append(np.nan)
        
        # Calculate the final metric: Mean of the aggregated risks across all 10 models
        if model_risks and not all(np.isnan(model_risks)):
            mean_agg_risk = np.nanmean(model_risks)
            results[metric_name] = mean_agg_risk

    # Format and print the table
    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Avg. Aggregated Risk (MSE)'])
    df_results['Avg. Aggregated Risk (MSE)'] = df_results['Avg. Aggregated Risk (MSE)'].map('{:.5f}'.format)
    
    print("\n--- Aggregation Performance Table ---")
    print("Metric: Mean OOD Aggregated Risk (lower is better)")
    # Using to_markdown() for clean, printable output in the terminal
    print(df_results.to_markdown())
    print("-" * 40)

def plot_cvar_tradeoff(risk_matrix):
    """
    Calculates and plots the average CVaR risk across all 10 models 
    as the risk parameter (alpha) varies from risk-averse (0.5) to mean (1.0).
    """
    alphas = np.linspace(0.5, 0.999, 20)
    avg_cvar_risks = []
    
    num_models = risk_matrix.size(0)
    aggregator = aggregation_function(name='cvar')
    
    print("\n--- Calculating CVaR Trade-off Curve ---")
    
    for alpha in tqdm(alphas, desc="Calculating CVaR for $\\alpha$"):
        model_risks_at_alpha = []
        for i in range(num_models):
            # Extract out-of-domain risks for model i
            base_risks = risk_matrix[i, :]
            ood_mask = torch.ones_like(base_risks, dtype=torch.bool)
            ood_mask[i] = False
            risks_for_aggregation = base_risks[ood_mask]

            aggregated_risk = aggregator.aggregate(risks_for_aggregation, alpha=alpha)
            model_risks_at_alpha.append(aggregated_risk.item())
            
        # Average the aggregated risks across all 10 models for this alpha
        avg_cvar_risks.append(np.mean(model_risks_at_alpha))

    # Plotting the result
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, avg_cvar_risks, marker='o', linestyle='-', color='indigo')
    plt.xlabel('$\\alpha$ (CVaR Risk Level: Lower $\\alpha$ = More Risk-Averse)')
    plt.ylabel('Average Aggregated Risk (MSE)')
    plt.title('Average CVaR Risk vs. Risk Tolerance $\\alpha$')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(avg_cvar_risks[-1], color='gray', linestyle=':', label=f'Mean Risk $\\approx$ {avg_cvar_risks[-1]:.5f}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- 3. VISUALIZATION OF AGGREGATION ---

def visualize_aggregations(risk_matrix, aggregation_functions):
    """
    Visualizes the distribution of a single model's risks across its *out-of-domain* target domains.
    """
    sns.set_style("whitegrid")
    
    # We choose the risks for the first model trained (M0) across all 10 Target Domains
    # To simulate the DomainBed LOOCV environment, we exclude the self-risk (R[0, 0])
    base_risks = risk_matrix[0, :]
    
    # Select risks excluding the self-domain risk (R[0, 0])
    # This simulates Model 0's performance on the 9 *other* environments
    risks_for_aggregation = torch.cat([base_risks[:0], base_risks[1:]])
    
    # If using all risks (including self-risk) use:
    # risks_for_aggregation = base_risks

    print(f"\n--- Aggregating risks for Model 0 on {risks_for_aggregation.numel()} out-of-domain targets ---")
    
    for func_name, params in aggregation_functions.items():
        param_name = params['name']
        param_values = params['values']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Use a colormap to get colors for each parameter value
        cmap = plt.cm.get_cmap("RdBu_r")
        norm = plt.Normalize(vmin=min(param_values), vmax=max(param_values))

        # Plot the base KDE once for a clear, filled background
        sns.kdeplot(risks_for_aggregation.numpy(), ax=ax, color='skyblue', fill=True, alpha=0.3, linewidth=0)

        # Loop to plot each vertical line for aggregated risk
        for param_value in tqdm(param_values, desc=f"Plotting {func_name}"):
            color = cmap(norm(param_value))
            
            # Use the out-of-domain risk vector for aggregation
            aggregator = aggregation_function(name=func_name) 
            agg_kwargs = {param_name: param_value}
            
            if func_name == "soft_cvar":
                agg_kwargs = {'alpha': param_value, 'eta': 1.0}
            
            try:
                aggregated_risk = aggregator.aggregate(risks_for_aggregation, **agg_kwargs)
                
                # Plot the vertical line corresponding to the aggregated risk
                ax.axvline(aggregated_risk.item(), linestyle='--', color=color, 
                           label=f'{param_value:.2f}', zorder=2, alpha=0.8, linewidth=2)
            except NotImplementedError as e:
                print(f"Skipping {func_name}: {e}")
                continue
        
        # Add a colorbar and remove the seaborn-generated legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label=f'${param_name}$', size=12, labelpad=5)
        
        # Customize the plot
        ax.set_title(f'Aggregated Risk for {func_name.upper()} on Model 0 Out-of-Domain Risks (9 Targets)')
        ax.set_xlabel('Risk Value (MSE on Target Domain)')
        ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("Starting Credit Card Default Experiment (DomainBed LOOCV structure simulation)...")
    
    # 1. Load data
    df_data = load_credit_card_default_data()
    
    # 2. Setup and Fit Preprocessor GLOBALLY on the entire feature space
    fitted_preprocessor = setup_and_fit_preprocessor(df_data)
    
    # 3. Create domain splits
    train_domain_data, target_domain_data = create_domain_splits(df_data)
    
    # 4. Train models (M0-M9, one per domain) - data is transformed inside this function
    domain_models = train_domain_models(train_domain_data, fitted_preprocessor)
    
    # 5. Calculate risk matrix (R[i, j] = risk of Model_i on Target_Domain_j) - data is transformed here as well
    risk_matrix_tensor = calculate_risk_matrix(target_domain_data, domain_models, fitted_preprocessor)
    
    # 6. Create performance metrics table
    create_metrics_table(risk_matrix_tensor)
    
    # 7. Plot full CVaR trade-off curve
    plot_cvar_tradeoff(risk_matrix_tensor)

    # 8. Define the aggregation functions and their parameter ranges
    aggregation_functions_to_plot = {
        'cvar': {'name': 'alpha', 'values': np.linspace(0.5, 0.99, 10)},
        'var': {'name': 'alpha', 'values': np.linspace(0.5, 0.99, 10)},
        'ph': {'name': 'xi', 'values': np.linspace(0.1, 5.0, 10)},
        'entropic': {'name': 'eta', 'values': np.linspace(0.1, 10.0, 10)},
        'soft_cvar': {'name': 'alpha', 'values': np.linspace(0.5, 0.99, 10)}
    }
    
    # 9. Visualize the aggregations 
    visualize_aggregations(risk_matrix_tensor, aggregation_functions_to_plot)
