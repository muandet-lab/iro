import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Ensure local repo src/ is on path (avoid picking up old installs)
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iro.aggregation.aggregators import AggregationFunction

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
NUM_DOMAINS = 10
LR = 1e-3
NUM_EPOCHS = 30
BATCH_SIZE = 128
ALPHA_PRIOR = (2.0, 5.0)   # Beta(a,b) for CVaR/EVaR tail level
GAMMA_LOGN = (-0.5, 0.8)   # mean, sigma for ESRM gamma

class LogisticRegressionModel(nn.Module):
    """Simple single-layer logistic regression for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

def loss_fn(y_pred_logits, y_true):
    """Unreduced BCE-with-logits."""
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
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

# 2. data loading and domain splitting

def load_credit_card_default_data():
    """Load and clean the UCI Credit Card Default dataset."""
    print(f"Loading data from: {DATA_URL}")
    df = pd.read_excel(DATA_URL, header=1)
    df = df.drop(columns=['ID'])
    df.rename(columns={'default payment next month': 'TARGET'}, inplace=True) 
    
    df.columns = [col.replace(' ', '_').upper() for col in df.columns]
    
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    
    return df

def setup_and_fit_preprocessor(df_full):
    """Create and fit the preprocessor globally on the entire feature space."""
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
    """Split by LIMIT_BAL quantiles -> per-domain train/test."""
    # create 10 domains based on quantiles of LIMIT_BAL (the domain variable)
    df['DOMAIN_ID'] = pd.qcut(df['LIMIT_BAL'], q=NUM_DOMAINS, labels=False, duplicates='drop')
    
    train_domain_data = {}
    target_domain_data = {}
    
    unique_domains = df['DOMAIN_ID'].unique()
    
    for i, domain_id in enumerate(unique_domains):
        df_domain = df[df['DOMAIN_ID'] == domain_id].copy()
        
        df_train, df_test = train_test_split(
            df_domain, 
            test_size=0.2, 
            random_state=42, 
            stratify=df_domain['TARGET']
        )
        
        X_train = df_train.drop(columns=['TARGET', 'DOMAIN_ID']) 
        y_train = df_train['TARGET'].values
        
        X_test = df_test.drop(columns=['TARGET', 'DOMAIN_ID'])
        y_test = df_test['TARGET'].values
        
        train_domain_data[f'D{i}'] = {'X': X_train, 'y': y_train}
        target_domain_data[f'D{i}'] = {'X': X_test, 'y': y_test}
        
    return train_domain_data, target_domain_data

def create_domain_dataloaders(train_domain_data, target_domain_data, preprocessor):
    """Build PyTorch loaders for train domains; keep processed targets for eval."""
    domain_dataloaders = {}
    
    for name, data in train_domain_data.items():
        X_transformed = preprocessor.transform(data['X'])
        y_train = data['y']
        
        dataset = DomainDataset(X_transformed, y_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        domain_dataloaders[name] = loader

    target_data_transformed = {}
    
    first_X_transformed = preprocessor.transform(target_domain_data['D0']['X'])
    input_dim = first_X_transformed.shape[1]
    
    for name, data in target_domain_data.items():
        X_transformed = preprocessor.transform(data['X'])
        target_data_transformed[name] = {'X': X_transformed, 'y': data['y']}


    return domain_dataloaders, target_data_transformed, input_dim


def draw_alpha():
    return np.random.beta(*ALPHA_PRIOR)


def draw_gamma():
    mu, sigma = GAMMA_LOGN
    return float(np.random.lognormal(mean=mu, sigma=sigma))


def train_risk_model(domain_dataloaders, input_dim, algorithm_name, alpha_value=0.5):
    """Train with ERM, CVaR, or IRO-style stochastic CVaR/ESRM/EVaR."""
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    agg_cvar = AggregationFunction("cvar")
    agg_esrm = AggregationFunction("exponential")
    agg_evar = AggregationFunction("evar")

    domain_names = list(domain_dataloaders.keys())
    domain_iterators = {name: iter(loader) for name, loader in domain_dataloaders.items()}
    num_domains = len(domain_dataloaders)

    try:
        min_steps = min(len(loader) for loader in domain_dataloaders.values())
    except ValueError:
        print("ERROR: No domains found or DataLoaders are empty.")
        return model

    for _ in tqdm(range(NUM_EPOCHS), desc=f"{algorithm_name} training"):
        model.train()
        for _ in range(min_steps):
            optimizer.zero_grad()
            env_risks = []
            for domain_name in domain_names:
                try:
                    X_domain, y_domain = next(domain_iterators[domain_name])
                except StopIteration:
                    domain_iterators[domain_name] = iter(domain_dataloaders[domain_name])
                    X_domain, y_domain = next(domain_iterators[domain_name])

                y_pred_logits = model(X_domain)
                env_risks.append(loss_fn(y_pred_logits, y_domain.float()).mean().reshape(1))

            env_risks_tensor = torch.cat(env_risks)

            if algorithm_name == "ERM":
                aggregated_loss = env_risks_tensor.mean()
            elif algorithm_name == "CVaR":
                aggregated_loss = agg_cvar.aggregate(env_risks_tensor, alpha=alpha_value)
            elif algorithm_name == "IRO-CVaR":
                alpha = draw_alpha()
                aggregated_loss = agg_cvar.aggregate(env_risks_tensor, alpha=alpha)
            elif algorithm_name == "IRO-ESRM":
                gamma = draw_gamma()
                aggregated_loss = agg_esrm.aggregate(env_risks_tensor, gamma=gamma)
            elif algorithm_name == "IRO-EVAR":
                alpha = draw_alpha()
                aggregated_loss = agg_evar.aggregate(env_risks_tensor, alpha=alpha)
            else:
                raise ValueError(f"Unknown algorithm {algorithm_name}")

            aggregated_loss.backward()
            optimizer.step()

    return model


def calculate_risk_vector(target_domain_data, model):
    """Risk vector R[j] = MSE(model on target_domain_j)."""
    model.eval()
    target_names = list(target_domain_data.keys())
    risk_vector = np.zeros(len(target_names))

    with torch.no_grad():
        for j, name in enumerate(target_names):
            data_j = target_domain_data[name]
            X_tensor = torch.from_numpy(data_j["X"]).float()
            y_numpy = data_j["y"]
            y_pred_logits = model(X_tensor)
            y_pred_proba = torch.sigmoid(y_pred_logits).numpy()
            risk_vector[j] = mean_squared_error(y_numpy, y_pred_proba)

    return torch.from_numpy(risk_vector).float()

def create_iro_metrics_table(risk_vector, model_name):
    """Print a small aggregation table."""
    metrics = {
        "Mean (ERM)": ("mean", {}),
        "CVaR (alpha=0.9)": ("cvar", {"alpha": 0.9}),
        "CVaR (alpha=0.5)": ("cvar", {"alpha": 0.5}),
        "EVaR (alpha=0.9)": ("evar", {"alpha": 0.9}),
        "ESRM (gamma=1.0)": ("exponential", {"gamma": 1.0}),
        "Worst-case": ("worst_case", {}),
    }

    rows = []
    for metric_name, (func_name, kwargs) in metrics.items():
        agg = AggregationFunction(func_name).aggregate(risk_vector, **kwargs)
        rows.append((metric_name, agg.item()))

    df_results = pd.DataFrame(rows, columns=["Metric", "Risk"])
    print(f"\n{model_name} aggregated-risk summary:")
    print(df_results.to_markdown(index=False))

def plot_cvar_comparison(iro_risks, erm_risks):
    """Plots CVaR trade-off curves for IRO vs ERM."""
    alphas = np.linspace(0.05, 0.95, 40)
    agg = AggregationFunction("cvar")
    iro_curve = [agg.aggregate(iro_risks, alpha=a).item() for a in alphas]
    erm_curve = [agg.aggregate(erm_risks, alpha=a).item() for a in alphas]

    plt.figure(figsize=(9, 5))
    plt.plot(alphas, iro_curve, label="IRO-CVaR", linewidth=2)
    plt.plot(alphas, erm_curve, label="ERM baseline", linewidth=2, linestyle="--")
    plt.xlabel(r"$\alpha$ (tail level)")
    plt.ylabel("Aggregated risk (MSE)")
    plt.title("CVaR trade-off")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    sns.set_style("whitegrid")
    df_data = load_credit_card_default_data()
    preprocessor = setup_and_fit_preprocessor(df_data)
    raw_train, raw_target = create_domain_splits(df_data)
    train_loaders, target_data, input_dim = create_domain_dataloaders(raw_train, raw_target, preprocessor)

    iro_cvar = train_risk_model(train_loaders, input_dim, algorithm_name="IRO-CVaR")
    iro_esrm = train_risk_model(train_loaders, input_dim, algorithm_name="IRO-ESRM")
    erm_model = train_risk_model(train_loaders, input_dim, algorithm_name="ERM")

    iro_risks = calculate_risk_vector(target_data, iro_cvar)
    iro_esrm_risks = calculate_risk_vector(target_data, iro_esrm)
    erm_risks = calculate_risk_vector(target_data, erm_model)

    create_iro_metrics_table(iro_risks, "IRO-CVaR")
    create_iro_metrics_table(iro_esrm_risks, "IRO-ESRM")
    create_iro_metrics_table(erm_risks, "ERM baseline")
    plot_cvar_comparison(iro_risks, erm_risks)


if __name__ == "__main__":
    main()
