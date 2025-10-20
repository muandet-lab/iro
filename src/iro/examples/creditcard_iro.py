import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
# Using the specific URL for the .xls file 
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
NUM_DOMAINS = 10  # 10 domains defined by LIMIT_BAL quantiles
MAX_CLASSES = 2 # Binary classification (0: No Default, 1: Default)
# Optimization Parameters
LR = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64 # Size of minibatch for each domain

# --- 1. PYTORCH MODEL AND DATA UTILITIES ---

class LogisticRegressionModel(nn.Module):
    """
    A simple PyTorch-based Logistic Regression equivalent for gradient-based training.
    """
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1) # Output logits

def loss_fn(y_pred_logits, y_true):
    """
    Standard Binary Cross-Entropy Loss with Logits (unreduced).
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    return bce_loss(y_pred_logits, y_true.float())

class DomainDataset(Dataset):
    """Wraps domain data for PyTorch DataLoader."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. DATA LOADING AND DOMAIN SPLITTING (Matching previous script's logic) ---

def load_credit_card_default_data():
    """
    Loads and preprocesses the UCI Credit Card Default dataset using pd.read_excel.
    """
    print(f"Loading data from: {DATA_URL}")
    df = pd.read_excel(DATA_URL, header=1)
    df = df.drop(columns=['ID'])
    df.rename(columns={'default payment next month': 'TARGET'}, inplace=True) 
    
    # Clean column names (matching previous script)
    df.columns = [col.replace(' ', '_').upper() for col in df.columns]
    
    # Basic data cleaning (matching previous script)
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    
    return df

def setup_and_fit_preprocessor(df_full):
    """
    Creates and fits the preprocessor globally on the entire feature space.
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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    X_full = df_full.drop(columns=['TARGET'])
    preprocessor.fit(X_full)
    
    return preprocessor

def create_domain_splits(df):
    """
    Splits the full dataframe into 10 domains based on LIMIT_BAL quantiles,
    then splits each domain into 80/20 train/test sets (DomainBed style).
    
    Returns: 
        - train_domain_data: dict of {'D0': (X_train, y_train), ...}
        - target_domain_data: dict of {'D0': (X_test, y_test), ...}
    """
    # Create 10 domains based on quantiles of LIMIT_BAL (the domain variable)
    df['DOMAIN_ID'] = pd.qcut(df['LIMIT_BAL'], q=NUM_DOMAINS, labels=False, duplicates='drop')
    
    train_domain_data = {}
    target_domain_data = {}
    
    # Iterate through unique domains
    unique_domains = df['DOMAIN_ID'].unique()
    
    for i, domain_id in enumerate(unique_domains):
        # Data for this specific domain
        df_domain = df[df['DOMAIN_ID'] == domain_id].copy()
        
        # Split domain into train (80%) and test (20%)
        df_train, df_test = train_test_split(
            df_domain, 
            test_size=0.2, 
            random_state=42, 
            stratify=df_domain['TARGET']
        )
        
        # Extract features and targets for train set
        X_train = df_train.drop(columns=['TARGET', 'DOMAIN_ID']) 
        y_train = df_train['TARGET'].values
        
        # Extract features and targets for test set
        X_test = df_test.drop(columns=['TARGET', 'DOMAIN_ID'])
        y_test = df_test['TARGET'].values
        
        # Store raw dataframes/arrays
        train_domain_data[f'D{i}'] = {'X': X_train, 'y': y_train}
        target_domain_data[f'D{i}'] = {'X': X_test, 'y': y_test}
        
    return train_domain_data, target_domain_data

def create_domain_dataloaders(train_domain_data, target_domain_data, preprocessor):
    """
    Converts raw domain splits into PyTorch DataLoaders (for training) 
    and transformed NumPy arrays (for final evaluation).
    """
    domain_dataloaders = {}
    
    # 1. Create Train DataLoaders (Minibatches from each domain)
    for name, data in train_domain_data.items():
        # Transform using the globally fitted preprocessor
        X_transformed = preprocessor.transform(data['X'])
        y_train = data['y']
        
        dataset = DomainDataset(X_transformed, y_train)
        # Use drop_last=True for consistent risk profile size during IRO training
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        domain_dataloaders[name] = loader

    # 2. Prepare Global Test Data for final evaluation (transformed)
    target_data_transformed = {}
    
    # Get input dimension from the first transformed domain
    first_X_transformed = preprocessor.transform(target_domain_data['D0']['X'])
    input_dim = first_X_transformed.shape[1]
    
    for name, data in target_domain_data.items():
        X_transformed = preprocessor.transform(data['X'])
        target_data_transformed[name] = {'X': X_transformed, 'y': data['y']}


    return domain_dataloaders, target_data_transformed, input_dim

# --- 3. TRAINING ALGORITHMS ---

def train_irm_optimized_model(domain_dataloaders, input_dim, algorithm_name, alpha_value=0.5):
    """
    Trains a single model using a risk optimization objective (Mean, CVaR, or IRO).
    """
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    aggregator = aggregation_function(name='cvar') 
    
    # Get iterators for all domain data loaders
    domain_names = list(domain_dataloaders.keys())
    domain_iterators = {name: iter(loader) for name, loader in domain_dataloaders.items()}
    num_domains = len(domain_dataloaders)
    
    print(f"\n--- Training {algorithm_name} Optimized Model ---")

    # Determine number of steps per epoch by the smallest loader
    try:
        min_steps = min(len(loader) for loader in domain_dataloaders.values())
    except ValueError:
        print("ERROR: No domains found or DataLoaders are empty.")
        return model
        
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"{algorithm_name} Training"):
        model.train()
        
        for step in range(min_steps):
            optimizer.zero_grad()
            
            # Collect the unreduced loss (risk) from each domain's minibatch
            env_risks = []
            
            for i in range(num_domains):
                domain_name = domain_names[i]
                try:
                    X_domain, y_domain = next(domain_iterators[domain_name])
                except StopIteration:
                    # Reset iterator if one domain runs out
                    domain_iterators[domain_name] = iter(domain_dataloaders[domain_name])
                    X_domain, y_domain = next(domain_iterators[domain_name])

                # Get unreduced loss (one loss value per sample)
                y_pred_logits = model(X_domain)
                loss_unreduced = loss_fn(y_pred_logits, y_domain.float())
                
                # The average loss (empirical risk) for this domain's minibatch
                # NOTE: The paper's IRO minimizes the expectation of aggregated risk over alpha.
                # Here, we use the average minibatch loss as the domain risk R_i.
                env_risks.append(loss_unreduced.mean().reshape(1))

            # Stack the domain risks (minibatch risk profile R_in)
            env_risks_tensor = torch.cat(env_risks)
            
            # --- IRO / Risk Optimization Objective ---
            if algorithm_name == 'IRO':
                # 1. Sample alpha (preference parameter lambda) from the preference space Q=[0,1]
                # Using Uniform(0.01, 0.99) as a simple parameter space Q.
                alpha = np.random.uniform(0.01, 0.99)
                
                # 2. Compute the aggregated risk rho_lambda(R_in)
                aggregated_loss = aggregator.aggregate(env_risks_tensor, alpha=alpha)
            
            elif algorithm_name == 'ERM':
                # ERM is equivalent to Mean Risk
                aggregated_loss = env_risks_tensor.mean()
            
            elif algorithm_name == 'CVaR':
                # CVaR (Fixed alpha) 
                aggregated_loss = aggregator.aggregate(env_risks_tensor, alpha=alpha_value)

            # --- Optimization Step ---
            aggregated_loss.backward()
            optimizer.step()

    return model

# --- 4. EVALUATION ---

def calculate_risk_matrix(target_domain_data, model):
    """
    Calculates the risk vector R where R[j] is the risk (MSE)
    of the model on data from target_domain_j.
    """
    model.eval()
    target_names = list(target_domain_data.keys())
    num_targets = len(target_names)
    risk_vector = np.zeros(num_targets)
    
    print("\n--- Calculating Risk Vector (OOD MSE) for Model ---")

    with torch.no_grad():
        for j in tqdm(range(num_targets), desc="Calculating Target Domain Risk"):
            data_j = target_domain_data[target_names[j]]
            
            # Transform to PyTorch tensors
            X_tensor = torch.from_numpy(data_j['X']).float()
            y_numpy = data_j['y']
            
            # Predict probabilities
            y_pred_logits = model(X_tensor)
            y_pred_proba = torch.sigmoid(y_pred_logits).numpy() 
            
            # MSE of predicted probabilities vs true labels (standard metric for DGIL)
            risk_vector[j] = mean_squared_error(y_numpy, y_pred_proba)
            
    return torch.from_numpy(risk_vector).float()

def create_iro_metrics_table(risk_vector, model_name):
    """
    Calculates and prints a table summarizing the risk profile of the model.
    """
    # Define the specific metrics to calculate
    metrics = {
        "Mean Risk (ERM)": ('mean', {}), 
        "CVaR (alpha=0.9)": ('cvar', {'alpha': 0.9}), 
        "CVaR (alpha=0.5)": ('cvar', {'alpha': 0.5}), 
        "CVaR (alpha=0.2)": ('cvar', {'alpha': 0.2}),
        "Worst-Case Risk": ('worst_case', {}), 
    }

    results = {}
    
    print(f"\n--- {model_name} Aggregated Risk Performance ---")

    for metric_name, (func_name, kwargs) in metrics.items():
        # Use the fallback or imported AggregationFunction
        aggregator = aggregation_function(name=func_name)
        
        try:
            aggregated_risk = aggregator.aggregate(risk_vector, **kwargs)
            results[metric_name] = aggregated_risk.item()
        except NotImplementedError:
            results[metric_name] = np.nan

    # Format and print the table
    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Risk Value (MSE)'])
    df_results['Risk Value (MSE)'] = df_results['Risk Value (MSE)'].map('{:.5f}'.format)
    
    print(f"\n--- {model_name} Performance Table (Risk Aggregation) ---")
    print("Metric: Risk of the Single Model on all 10 Target Domains")
    print(df_results.to_markdown())
    print("-" * 70)

def plot_cvar_comparison(iro_risks, erm_risks):
    """Plots the CVaR trade-off curve for comparison."""
    alphas = np.linspace(0.01, 0.99, 50)
    iro_cvar_risks = []
    erm_cvar_risks = []
    
    # Use the fallback or imported AggregationFunction
    aggregator = aggregation_function(name='cvar')
    
    for alpha in alphas:
        iro_cvar = aggregator.aggregate(iro_risks, alpha=alpha)
        erm_cvar = aggregator.aggregate(erm_risks, alpha=alpha)
        iro_cvar_risks.append(iro_cvar.item())
        erm_cvar_risks.append(erm_cvar.item())

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, iro_cvar_risks, linestyle='-', color='indigo', linewidth=2, label='IRO Optimized Model')
    plt.plot(alphas, erm_cvar_risks, linestyle='--', color='red', linewidth=2, label='ERM Optimized Model (Baseline)')
    
    plt.xlabel('$\\alpha$ (CVaR Risk Level: Lower $\\alpha$ = More Risk-Averse)')
    plt.ylabel('Aggregated Risk (MSE) on All Target Domains')
    plt.title('CVaR Trade-off: IRO vs. ERM Optimized Models (Credit Card Default)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sns.set_style("whitegrid")
    
    print("Starting Credit Card IRO Optimization Experiment (v2)...")
    
    # 1. Load data and setup Preprocessor
    df_data = load_credit_card_default_data()
    fitted_preprocessor = setup_and_fit_preprocessor(df_data)
    
    # 2. Create Domain Splits (raw dataframes)
    raw_train_domain_data, raw_target_domain_data = create_domain_splits(df_data)

    # 3. Transform and create PyTorch DataLoaders for IRO Training
    train_dataloaders, target_domain_data, input_dim = create_domain_dataloaders(
        raw_train_domain_data, raw_target_domain_data, fitted_preprocessor
    )
    
    print(f"Data Prep Complete. Input Dimension for Model: {input_dim}")
    print(f"Number of training domains (for R_in): {len(train_dataloaders)}")
    print(f"Number of target domains (for R_test): {len(target_domain_data)}")
    
    # --- 4. Train Models ---
    
    # I. Train the IRO Model (Imprecise Risk Optimization)
    iro_model = train_irm_optimized_model(
        train_dataloaders, input_dim, algorithm_name='IRO'
    )
    
    # II. Train a standard ERM (Empirical Risk Minimization) Model as a baseline
    erm_model = train_irm_optimized_model(
        train_dataloaders, input_dim, algorithm_name='ERM'
    )
    
    # --- 5. Evaluate Models on OOD Target Domains ---
    
    # I. Evaluate IRO Model
    iro_risk_vector = calculate_risk_matrix(target_domain_data, iro_model)
    create_iro_metrics_table(iro_risk_vector, 'IRO Model')
    
    # II. Evaluate ERM Model
    erm_risk_vector = calculate_risk_matrix(target_domain_data, erm_model)
    create_iro_metrics_table(erm_risk_vector, 'ERM Model (Baseline)')

    # --- 6. Comparative Visualization ---
    plot_cvar_comparison(iro_risk_vector, erm_risk_vector)
