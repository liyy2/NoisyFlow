import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.autograd as autograd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import wasserstein_distance
import ot
from sklearn.utils import resample
from scipy.stats import gaussian_kde
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}

class NonNegativeLinear(nn.Linear):
    def __init__(self, *args, beta=1.0, **kwargs):
        super(NonNegativeLinear, self).__init__(*args, **kwargs)
        self.beta = beta
        return

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        return nn.functional.softplus(self.weight, beta=self.beta)

class ICNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units,
        activation="LeakyReLU",
        softplus_W_kernels=False,
        softplus_beta=1,
        std=0.1,
        fnorm_penalty=0,
        kernel_init_fxn=None,
    ):
        super(ICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_W_kernels = softplus_W_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        units = hidden_units + [1]

        if self.softplus_W_kernels:
            def WLinear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)
        else:
            WLinear = nn.Linear

        self.W = nn.ModuleList(
            [
                WLinear(idim, odim, bias=False)
                for idim, odim in zip(units[:-1], units[1:])
            ]
        )

        self.A = nn.ModuleList(
            [nn.Linear(input_dim, odim, bias=True) for odim in units]
        )

        if kernel_init_fxn is not None:
            for layer in self.A:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.W:
                kernel_init_fxn(layer.weight)

    def forward(self, x):
        z = self.sigma(0.2)(self.A[0](x))
        z = z * z

        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))

        y = self.W[-1](z) + self.A[-1](x)

        return y

    def transport(self, x):
        assert x.requires_grad

        output = autograd.grad(
            self.forward(x).sum(),  # Sum over batch dimension
            x,
            create_graph=True,
            only_inputs=True,
            retain_graph=True
        )[0]
        return output

    def clamp_w(self):
        if self.softplus_W_kernels:
            return

        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.W)
        )

class NeuralOT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        D: float,
        eps1: float,
        eps2: float,
        delta1: float,
        delta2: float,
        delta_prime: float,
        batch_size: int,
        num_iterations: int,
        learning_rate: float,
        clip_norm: float,
        ell: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d = input_dim
        self.k = projection_dim
        self.D = D
        self.eps1 = eps1
        self.eps2 = eps2
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta_prime = delta_prime
        self.eps = eps1 + eps2
        self.delta = delta1 + delta2 + delta_prime
        self.B = batch_size
        self.T = num_iterations
        self.alpha = learning_rate
        self.C = clip_norm
        self.ell = ell

        # Initialize ICNNs with proper architecture
        hidden_units = [32]  # Single hidden layer of 32 units
        self.phi = ICNN(
            input_dim=projection_dim,
            hidden_units=hidden_units,
            activation="LeakyReLU",
            softplus_W_kernels=True,  # Using softplus, so no need for clamping
            softplus_beta=1.0,
            std=0.1,
            fnorm_penalty=0.01
        )
        
        self.psi = ICNN(
            input_dim=projection_dim,
            hidden_units=hidden_units,
            activation="LeakyReLU",
            softplus_W_kernels=True,  # Using softplus, so no need for clamping
            softplus_beta=1.0,
            std=0.1,
            fnorm_penalty=0.01
        )

    def generate_projection_matrix(self) -> torch.Tensor:
        # M ~ N(0, 1/2 * I)
        return torch.randn(self.d, self.k) / np.sqrt(2)

    def compute_SM(self) -> float:
        # S_M := sqrt(k) + sqrt(d) + sqrt(2 ln(1/delta'))
        return np.sqrt(self.k) + np.sqrt(self.d) + np.sqrt(2 * np.log(1.0/self.delta_prime))

    def compute_sigma(self, SM: float) -> float:
        # sigma := (2 D S_M / eps1) * sqrt(2 ln(1.25/delta1))
        return (2 * self.D * SM / self.eps1) * np.sqrt(2 * np.log(1.25/self.delta1))

    def compute_sigma_g(self, n_s: int) -> float:
        """
        Compute the noise scale for gradients.
        Args:
            n_s: Number of samples in source domain
        Returns:
            float: Noise scale sigma_g
        """
        # sigma_g := (q * C * sqrt(2 T ln(1/delta2)))/eps2
        # q = B / n_s (sampling ratio)
        q = self.B / n_s
        noise_scale = (q * self.C * np.sqrt(2 * self.T * np.log(1.0/self.delta2))) / self.eps2
        return noise_scale

    def forward(self, X_s: torch.Tensor, X_t: torch.Tensor, M: torch.Tensor, sigma: float) -> torch.Tensor:
        # Ensure all tensors are on the same device
        device = X_s.device
        M = M.to(device)
        
        # Project data
        X_s_tilde = X_s @ M
        X_t_tilde = X_t @ M

        # L2 normalize with epsilon for numerical stability
        X_s_tilde = X_s_tilde / (torch.norm(X_s_tilde, dim=1, keepdim=True) + 1e-8)
        X_t_tilde = X_t_tilde / (torch.norm(X_t_tilde, dim=1, keepdim=True) + 1e-8)

        # Add scaled noise
        Delta = torch.randn_like(X_s_tilde, device=device) * sigma
        X_s_noisy = X_s_tilde + Delta

        # Compute potentials with scaling
        phi_outputs = self.phi(X_s_noisy) * 1e-4  # Scale down phi outputs
        psi_outputs = self.psi(X_t_tilde) * 1e-4  # Scale down psi outputs

        # Compute normalized cost matrix
        C_mat = torch.cdist(X_s_noisy, X_t_tilde)
        C_mat = C_mat / (C_mat.max() + 1e-8) * 1e-4  # Scale down cost matrix
        C_tilde = C_mat - self.ell * (sigma**2)

        # Compute violations with scaled inputs
        violations = phi_outputs + psi_outputs.T - C_tilde

        # Compute loss with adjusted penalty term
        potential_term = torch.mean(phi_outputs) + torch.mean(psi_outputs)
        penalty_term = 0.1 * torch.mean(torch.relu(violations)**2)  # Reduced penalty coefficient
        
        loss = -potential_term + penalty_term

        return loss

    def compute_wasserstein_distance(self, X_s: torch.Tensor, X_t: torch.Tensor, M: torch.Tensor, sigma: float) -> torch.Tensor:
        """Compute the Wasserstein distance between source and target distributions."""
        X_s_tilde = X_s @ M
        X_t_tilde = X_t @ M

        # Use same L2 normalization
        
        X_s_tilde = X_s_tilde / (torch.norm(X_s_tilde, dim=1, keepdim=True) + 1e-8)
        X_t_tilde = X_t_tilde / (torch.norm(X_t_tilde, dim=1, keepdim=True) + 1e-8)
        

        Delta = torch.randn_like(X_s_tilde) * sigma * 0.001
        X_s_noisy = X_s_tilde + Delta 

        # Use same scaling as forward pass
        phi_outputs = self.phi(X_s_noisy) 
        psi_outputs = self.psi(X_t_tilde) 

        # Use same cost matrix normalization
        C_mat = torch.cdist(X_s_noisy, X_t_tilde)
        #C_mat = C_mat / (C_mat.max() + 1e-8) 

        wasserstein_dist = (torch.mean(C_mat) - torch.mean(phi_outputs) - torch.mean(psi_outputs)) * 1000
        
        return wasserstein_dist, C_mat, None

def train_dp_neural_ot(X_t, X_s, model_params):
    """Train the Neural OT model with SGD optimizer."""
    # Initialize model and parameters
    n_s = X_s.shape[0]
    n_t = X_t.shape[0]
    
    print(f"\nData dimensions:")
    print(f"Source samples: {n_s}")
    print(f"Target samples: {n_t}")
    
    model = NeuralOT(**model_params)
    
    # Change to SGD optimizer with momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_params['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4
    )

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize tracking lists
    losses = []
    wasserstein_distances = []
    iterations = []
    current_loss = float('inf')
    current_wasserstein = float('inf')
    grad_norms = torch.tensor([0.0])

    # Generate projection matrix and compute parameters
    M = model.generate_projection_matrix()
    SM = model.compute_SM()
    sigma = model.compute_sigma(SM)
    sigma_g = model.compute_sigma_g(n_s)

    # Training parameters
    T = model_params['num_iterations']
    B = model_params['batch_size']
    C = model_params['clip_norm']

    # Normalize input data
    X_s = (X_s - X_s.mean()) / (X_s.std() + 1e-8)
    X_t = (X_t - X_t.mean()) / (X_t.std() + 1e-8)

    for t in range(T):
        # Sample batch
        idx_s = np.random.choice(n_s, B, replace=True)
        idx_t = np.random.choice(n_t, B, replace=True)
        batch_X_s = X_s[idx_s]
        batch_X_t = X_t[idx_t]
        
        # Forward pass
        optimizer.zero_grad()
        loss = model(batch_X_s, batch_X_t, M, sigma)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), C)
        
        # Optimizer step
        optimizer.step()
        
        # Update current loss and compute Wasserstein distance
        current_loss = loss.item()
        with torch.no_grad():
            current_wasserstein, _, _ = model.compute_wasserstein_distance(
                X_s.clone().detach(),
                X_t.clone().detach(),
                M,
                sigma
            )
            grad_norms = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
        
        # Save checkpoint every 50 epochs
        if t % 50 == 0:
            checkpoint = {
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'wasserstein_distance': current_wasserstein.item(),
                'projection_matrix': M,
                'model_params': model_params,
                'losses': losses,
                'wasserstein_distances': wasserstein_distances,
                'iterations': iterations,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            torch.save(checkpoint, f'checkpoints/model_checkpoint_epoch_{t}.pt')
            print(f"\nSaved checkpoint at epoch {t}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if t % 25 == 0:
            losses.append(current_loss)
            wasserstein_distances.append(current_wasserstein.item())
            iterations.append(t)
            print(f"Iteration {t}")
            print(f"Loss: {current_loss:.4f}")
            print(f"Wasserstein Distance: {current_wasserstein.item():.4f}")
            print(f"Average gradient norm: {grad_norms.mean().item():.4f}")
            print(f"Noise scale: {sigma_g * C:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Save final checkpoint
    checkpoint = {
        'epoch': T,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
        'wasserstein_distance': current_wasserstein.item(),
        'projection_matrix': M,
        'model_params': model_params,
        'losses': losses,
        'wasserstein_distances': wasserstein_distances,
        'iterations': iterations,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, f'checkpoints/model_checkpoint_final.pt')
    print("\nSaved final checkpoint")

    # Transform target data before returning
    X_target_transformed = transport_target_to_source(model, X_t, M)

    # Visualize advanced transport
    visualize_advanced_transport(
        model,
       X_s,
        X_t,
        M,
        model_params['eps1'] + model_params['eps2'],
        save_dir='advanced_ot_results'
   )

    # After training is complete
    epsilon = model_params['eps1'] + model_params['eps2']  # Total epsilon
    analyze_membership_inference(
        model, 
        X_s, 
        X_t, 
        epsilon
    )
    
    return model, X_target_transformed

def train_and_evaluate_xgboost(x_train, y_train, x_test, y_test, feature_indices=None):
    """
    Train and evaluate XGBoost model on specified features.
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        feature_indices: Optional list of feature indices to use
    
    Returns:
        metrics: Dictionary of evaluation metrics
        model: Trained XGBoost model
    """
    # Convert to numpy if needed
    if torch.is_tensor(x_train):
        x_train = x_train.detach().cpu().numpy()
    if torch.is_tensor(x_test):
        x_test = x_test.detach().cpu().numpy()
    if torch.is_tensor(y_train):
        y_train = y_train.detach().cpu().numpy()
    if torch.is_tensor(y_test):
        y_test = y_test.detach().cpu().numpy()
    
    # Select features if specified
    if feature_indices is not None:
        x_train = x_train[:, feature_indices]
        x_test = x_test[:, feature_indices]
    
    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(x_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(x_test)
    
    # Calculate metrics with bootstrap
    metrics = calculate_metrics_with_bootstrap(y_test, y_pred)
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}")
    print(f"Precision: {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}")
    print(f"Recall: {metrics['recall'][0]:.4f} ± {metrics['recall'][1]:.4f}")
    print(f"F1 Score: {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}")
    
    return metrics, model

def transport_target_to_source(model: NeuralOT, X_target: np.ndarray, M: torch.Tensor) -> np.ndarray:
    """Transform target data to source domain using gradient of psi network."""
    X_target = torch.as_tensor(X_target, dtype=torch.float32)
    X_proj = X_target @ M
    X_proj.requires_grad_(True)
    
    model.psi.eval()
    transported = X_proj + model.psi.transport(X_proj)
    X_transported = transported @ M.t()
    
    return X_transported.detach().numpy()


def plot_training_curves(losses, wasserstein_distances):
    """Plot training curves showing loss and Wasserstein distance convergence."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(wasserstein_distances, label='Wasserstein Distance', color='orange')
    plt.title('Wasserstein Distance')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis_results/training_curves.png')
    plt.close()

def plot_feature_distributions(x_source, x_target, x_target_transformed):
    """Plot the distribution of features before and after transport."""
    plt.figure(figsize=(15, 5))
    
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(x_source):
        x_source = x_source.detach().numpy()
    if torch.is_tensor(x_target):
        x_target = x_target.detach().numpy()
    if torch.is_tensor(x_target_transformed):
        x_target_transformed = x_target_transformed.detach().numpy()
    
    # Before Transport
    plt.subplot(1, 2, 1)
    source_mean = np.mean(x_source, axis=1)
    target_mean = np.mean(x_target, axis=1)
    
    sns.kdeplot(data=source_mean, label='Source', color='blue')
    sns.kdeplot(data=target_mean, label='Target', color='red')
    
    plt.title('Distribution Before Transport')
    plt.xlabel('Mean Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # After Transport
    plt.subplot(1, 2, 2)
    transformed_mean = np.mean(x_target_transformed, axis=1)
    
    sns.kdeplot(data=transformed_mean, label='Transported Source', color='blue')
    sns.kdeplot(data=target_mean, label='Target', color='red')
    
    plt.title('Distribution After Transport')
    plt.xlabel('Mean Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    plt.close()

def compute_wasserstein_stats(source_data, target_data, n_bootstrap=1000):
    """Compute Wasserstein distance statistics using bootstrapping"""
    # Convert PyTorch tensors to numpy arrays if needed
    if torch.is_tensor(source_data):
        source_data = source_data.detach().cpu().numpy()
    if torch.is_tensor(target_data):
        target_data = target_data.detach().cpu().numpy()
    
    distances = []
    n_source = len(source_data)
    n_target = len(target_data)
    
    # Original distance
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target
    M = ot.dist(source_data, target_data)
    original_dist = ot.emd2(a, b, M)
    
    # Bootstrap to get distribution of distances
    for _ in range(n_bootstrap):
        source_idx = resample(range(n_source))
        target_idx = resample(range(n_target))
        
        source_boot = source_data[source_idx]
        target_boot = target_data[target_idx]
        
        M_boot = ot.dist(source_boot, target_boot)
        dist = ot.emd2(a, b, M_boot)
        distances.append(dist)
    
    distances = np.array(distances)
    return {
        'distance': original_dist,
        'std': np.std(distances),
        'normalized_distance': original_dist / np.mean(M),
        'standardized_distance': (original_dist - np.mean(distances)) / np.std(distances)
    }

def calculate_metrics_with_bootstrap(y_true, y_pred, n_bootstrap=1000):
    """
    Calculate classification metrics with bootstrap confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary containing metrics with their confidence intervals
    """
    n_samples = len(y_true)
    bootstrap_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Original metrics
    original_accuracy = accuracy_score(y_true, y_pred)
    original_precision = precision_score(y_true, y_pred, zero_division=0)
    original_recall = recall_score(y_true, y_pred, zero_division=0)
    original_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Bootstrap
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = resample(range(n_samples), n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metrics
        bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        bootstrap_metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
    
    # Calculate confidence intervals
    results = {}
    for metric in bootstrap_metrics:
        values = np.array(bootstrap_metrics[metric])
        std = np.std(values)
        results[metric] = (
            [original_accuracy, original_precision, original_recall, original_f1][
                list(bootstrap_metrics.keys()).index(metric)
            ],
            std
        )
    
    return results

def select_improved_features(X_source, X_target, X_target_transformed):
    """
    Select features where Wasserstein distance improved after transport.
    Scale distances by 1000 for better readability.
    """
    n_features = X_source.shape[1]
    improvements = []
    
    for i in range(n_features):
        # Calculate pre-transport Wasserstein distance (scaled by 1000)
        pre_transport = wasserstein_distance(X_source[:, i], X_target[:, i]) * 1000
        
        # Calculate post-transport Wasserstein distance (scaled by 1000)
        post_transport = wasserstein_distance(X_source[:, i], X_target_transformed[:, i]) * 1000
        
        # Calculate improvement (negative means better)
        improvement = post_transport - pre_transport
        improvements.append(improvement)
    
    # Select features where post-transport distance is smaller
    selected_features = np.where(np.array(improvements) < 0)[0]
    
    print(f"\nSelected {len(selected_features)} features with improved Wasserstein distance")
    for idx in selected_features:
        print(f"Feature {idx}: Improvement = {-improvements[idx]:.4f}")
    
    return selected_features, improvements

def train_multiple_epsilon(X_t, X_s, y_t, y_s, model_params, epsilons, n_runs=1):
    """
    Train models with different epsilon values (single run per epsilon).
    
    Args:
        X_t, X_s: Target and source data
        y_t, y_s: Target and source labels
        model_params: Base parameters
        epsilons: List of epsilon values to test
    """
    results = {}
    
    for eps in epsilons:
        print(f"\nTraining with ε={eps}")
        
        # Create parameters for this run
        current_params = {
            'input_dim': model_params['input_dim'],
            'projection_dim': model_params['projection_dim'],
            'D': model_params['D'],
            'eps1': eps / 2,
            'eps2': eps / 2,
            'delta1': model_params['delta1'],
            'delta2': model_params['delta2'],
            'delta_prime': model_params['delta_prime'],
            'batch_size': 32,
            'num_iterations': 500,
            'learning_rate': 0.0001,
            'clip_norm': model_params['clip_norm'],
            'ell': model_params['ell']
        }
        
        # Train model
        torch.manual_seed(42)  # Set fixed seed for reproducibility
        model, X_target_transformed = train_dp_neural_ot(X_t, X_s, current_params)
        
        # Get the latest checkpoint
        checkpoint = torch.load('checkpoints/model_checkpoint_final.pt')
        
        # Train XGBoost and get accuracy
        metrics, _ = train_and_evaluate_xgboost(X_s, y_s, X_target_transformed, y_t)
        
        # Store results
        results[eps] = {
            'losses': np.array(checkpoint['losses']),
            'losses_std': np.zeros_like(checkpoint['losses']),  # Zero std for single run
            'wasserstein_distances': np.array(checkpoint['wasserstein_distances']),
            'wasserstein_std': np.zeros_like(checkpoint['wasserstein_distances']),  # Zero std for single run
            'iterations': checkpoint['iterations'],
            'accuracy': metrics['accuracy'][0],
            'accuracy_std': metrics['accuracy'][1]
        }
        
        print(f"Accuracy for ε={eps}: {results[eps]['accuracy']:.4f}")
    
    return results

def plot_epsilon_comparison(epsilons, results):
    """
    Plot training metrics and accuracy for different epsilon values.
    """
    plt.figure(figsize=(15, 15))
    
    # Plot losses
    plt.subplot(3, 1, 1)
    for eps in epsilons:
        plt.plot(results[eps]['iterations'], 
                results[eps]['losses'], 
                label=f'ε={eps}')
    plt.title('Training Loss vs Iterations for Different ε Values for DP Neural OT')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Plot Wasserstein distances
    plt.subplot(3, 1, 2)
    for eps in epsilons:
        plt.plot(results[eps]['iterations'], 
                results[eps]['wasserstein_distances'], 
                label=f'ε={eps}')
    plt.title('Wasserstein Distance vs Iterations for Different ε Values for DP Neural OT')
    plt.xlabel('Iterations')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(3, 1, 3)
    accuracies = [results[eps]['accuracy'] for eps in epsilons]
    plt.plot(epsilons, accuracies, marker='o')
    plt.title('Classification Accuracy vs ε Values for DP Neural OT')
    plt.xlabel('ε')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis_results/epsilon_comparison.png')
    plt.close()

def plot_comprehensive_analysis(epsilons, results):
    """Create comprehensive visualizations with error bars."""
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 15))
    
    # Set color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))
    
    # 1. Training Loss Plot
    ax1 = plt.subplot(3, 1, 1)
    for i, eps in enumerate(epsilons):
        mean_loss = results[eps]['losses']
        std_loss = results[eps]['losses_std']
        iterations = results[eps]['iterations']
        
        ax1.plot(iterations, mean_loss, label=f'ε={eps}', color=colors[i])
        ax1.fill_between(iterations, 
                        mean_loss - std_loss, 
                        mean_loss + std_loss, 
                        alpha=0.2, 
                        color=colors[i])
    
    ax1.set_title('Training Loss Dynamics', fontsize=14, pad=20)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Total Wasserstein Distance Plot
    ax2 = plt.subplot(3, 1, 2)
    for i, eps in enumerate(epsilons):
        # Sum up Wasserstein distances across all features
        total_wd = np.sum(results[eps]['wasserstein_distances'], axis=1) if len(results[eps]['wasserstein_distances'].shape) > 1 else results[eps]['wasserstein_distances']
        total_wd_std = np.sum(results[eps]['wasserstein_std'], axis=1) if len(results[eps]['wasserstein_std'].shape) > 1 else results[eps]['wasserstein_std']
        iterations = results[eps]['iterations']
        
        ax2.plot(iterations, total_wd, label=f'ε={eps}', color=colors[i])
        ax2.fill_between(iterations, 
                        total_wd - total_wd_std, 
                        total_wd + total_wd_std, 
                        alpha=0.2, 
                        color=colors[i])
    
    ax2.set_title('Total Wasserstein Distance Evolution for DP Neural OT', fontsize=14, pad=20)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Total Wasserstein Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs Epsilon Plot
    ax3 = plt.subplot(3, 1, 3)
    accuracies = [results[eps]['accuracy'] for eps in epsilons]
    accuracy_stds = [results[eps]['accuracy_std'] for eps in epsilons]
    
    # Plot accuracy with error bars
    ax3.errorbar(epsilons, accuracies, yerr=accuracy_stds, 
                 fmt='o-', capsize=5, capthick=2, elinewidth=2,
                 color='blue', label='Accuracy')
    
    ax3.set_title('Classification Accuracy vs Privacy Budget for DP Neural OT', fontsize=14, pad=20)
    ax3.set_xlabel('Privacy Budget (ε)')
    ax3.set_ylabel('Accuracy')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add text box with statistics
    stats_text = "Privacy-Utility Trade-off:\n"
    for eps, acc, std in zip(epsilons, accuracies, accuracy_stds):
        stats_text += f"ε={eps}: {acc:.3f} ± {std:.3f}\n"
        # Add final total Wasserstein distance
        final_wd = np.sum(results[eps]['wasserstein_distances'][-1])
        stats_text += f"  Final Total W-dist: {final_wd:.3f}\n"
    
    ax3.text(1.05, 0.5, stats_text, transform=ax3.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('analysis_results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate figure for privacy budget distribution
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(epsilons))
    
    plt.bar(x - width/2, [eps/2 for eps in epsilons], width, 
            label='eps1', color='#2ecc71', alpha=0.7)
    plt.bar(x + width/2, [eps/2 for eps in epsilons], width, 
            label='eps2', color='#e74c3c', alpha=0.7)
    
    plt.title('Privacy Budget Distribution', fontsize=14, pad=20)
    plt.xlabel('Total Privacy Budget (ε)')
    plt.ylabel('Component Values')
    plt.xticks(x, [str(eps) for eps in epsilons])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_results/privacy_budget_distribution.png', dpi=300)
    plt.close()

def visualize_advanced_transport(model: NeuralOT, X_source: torch.Tensor, X_target: torch.Tensor, M: torch.Tensor, epsilon: float, save_dir='advanced_ot_results'):
    """
    Create advanced visualizations of the neural optimal transport using UMAP and other techniques.
    
    Args:
        model: Trained NeuralOT model
        X_source: Source domain data
        X_target: Target domain data
        M: Projection matrix
        epsilon: Privacy parameter epsilon
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get transported data
    X_transported = transport_target_to_source(model, X_target, M)
    
    # Convert all data to numpy
    if torch.is_tensor(X_source):
        X_source = X_source.detach().cpu().numpy()
    if torch.is_tensor(X_target):
        X_target = X_target.detach().cpu().numpy()
    
    # Combine data for dimensionality reduction
    X_combined = np.vstack([X_source, X_target, X_transported])
    
    # 1. UMAP Visualization
    plt.figure(figsize=(15, 10))
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_combined)
    
    # Split embeddings
    n_source = len(X_source)
    n_target = len(X_target)
    source_umap = embedding[:n_source]
    target_umap = embedding[n_source:n_source+n_target]
    transported_umap = embedding[n_source+n_target:]
    
    # Plot points
    plt.scatter(source_umap[:, 0], source_umap[:, 1], c='blue', label='Source', alpha=0.6)
    plt.scatter(target_umap[:, 0], target_umap[:, 1], c='red', label='Target', alpha=0.6)
    plt.scatter(transported_umap[:, 0], transported_umap[:, 1], c='green', label='Transported', alpha=0.6)
    
    # Plot transport paths
    for i in range(len(target_umap)):
        plt.arrow(target_umap[i, 0], target_umap[i, 1],
                 transported_umap[i, 0] - target_umap[i, 0],
                 transported_umap[i, 1] - target_umap[i, 1],
                 alpha=0.2, head_width=0.1, color='gray')
    
    plt.title(f'UMAP Transport Visualization (ε={epsilon})')
    plt.legend()
    plt.savefig(f'{save_dir}/umap_transport_eps_{epsilon}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    combined_tsne = tsne.fit_transform(X_combined)
    
    source_tsne = combined_tsne[:n_source]
    target_tsne = combined_tsne[n_source:n_source+n_target]
    transported_tsne = combined_tsne[n_source+n_target:]
    
    plt.figure(figsize=(15, 10))
    plt.scatter(source_tsne[:, 0], source_tsne[:, 1], c='blue', label='Source', alpha=0.6)
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], c='red', label='Target', alpha=0.6)
    plt.scatter(transported_tsne[:, 0], transported_tsne[:, 1], c='green', label='Transported', alpha=0.6)
    
    for i in range(len(target_tsne)):
        plt.arrow(target_tsne[i, 0], target_tsne[i, 1],
                 transported_tsne[i, 0] - target_tsne[i, 0],
                 transported_tsne[i, 1] - target_tsne[i, 1],
                 alpha=0.2, head_width=0.1, color='gray')
    
    plt.title(f't-SNE Transport Visualization (ε={epsilon})')
    plt.legend()
    plt.savefig(f'{save_dir}/tsne_transport_eps_{epsilon}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature-wise Density Comparison
    plt.figure(figsize=(15, 5))
    
    for i in range(min(3, X_source.shape[1])):  # Show first 3 features
        plt.subplot(1, 3, i+1)
        
        kde_source = gaussian_kde(X_source[:, i])
        kde_target = gaussian_kde(X_target[:, i])
        kde_transported = gaussian_kde(X_transported[:, i])
        
        x_range = np.linspace(min(X_source[:, i].min(), X_target[:, i].min()),
                             max(X_source[:, i].max(), X_target[:, i].max()), 100)
        
        plt.plot(x_range, kde_source(x_range), 'b-', label='Source')
        plt.plot(x_range, kde_target(x_range), 'r-', label='Target')
        plt.plot(x_range, kde_transported(x_range), 'g-', label='Transported')
        
        plt.title(f'Feature {i+1} Density')
        plt.legend()
    
    plt.suptitle(f'Density Comparison (ε={epsilon})')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/density_comparison_eps_{epsilon}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_membership_inference(model, X_source, X_target, epsilon):
    """
    Analyze membership inference risk for X_target using transport function.
    """
    device = X_target.device
    M = model.generate_projection_matrix().to(device)
    
    # Project target data
    X_target_proj = X_target @ M
    X_target_proj.requires_grad_(True)
    
    # Get transport outputs
    model.psi.eval()  # Set to evaluation mode
    transported = model.psi.transport(X_target_proj)
    
    # Convert to scores
    with torch.no_grad():
        scores = torch.norm(transported, dim=1).cpu().numpy()
        
        # Normalize scores to [0,1] range
        min_score, max_score = scores.min(), scores.max()
        scores = (scores - min_score) / (max_score - min_score + 1e-8)
    
    return scores.mean(), scores, None

def plot_membership_inference_comparison(all_outputs, epsilons, save_dir='member_ship_neural_ot'):
    """
    Plot membership inference analysis using entropy-based scores in separate subplots.
    """
    print(f"\nCreating directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the subplots
    fig, axes = plt.subplots(len(epsilons), 1, figsize=(10, 3*len(epsilons)))
    
    # Ensure axes is always an array
    if len(epsilons) == 1:
        axes = [axes]
    
    # Define colors for different epsilon values
    colors = ['purple', 'blue', 'green', 'lightgreen', 'yellow', 'orange', 'red', 'pink']
    
    # Add scores text for the file
    score_text = "Transport Scores:\n"
    
    for idx, (eps, outputs) in enumerate(all_outputs.items()):
        # Get scores from outputs
        if isinstance(outputs, tuple):
            scores = outputs[0]  # Use first element if tuple
        else:
            scores = outputs
            
        # Calculate entropy-based scores
        eps_small = 1e-10  # Small constant to avoid log(0)
        if len(scores.shape) > 1:  # If scores is a matrix
            entropies = -np.sum(scores * np.log(scores + eps_small), axis=1)
        else:  # If scores is already processed
            entropies = scores
            
        # Normalize entropies to [0,1] range
        scores_norm = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
        avg_score = scores_norm.mean()
        
        # Print debug information
        print(f"\nEpsilon {eps}:")
        print(f"Raw entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")
        print(f"Normalized scores range: [{scores_norm.min():.4f}, {scores_norm.max():.4f}]")
        print(f"Average score: {avg_score:.4f}")
        
        # Plot histogram
        axes[idx].hist(scores_norm, 
                      bins=30, 
                      density=True,
                      alpha=0.7,
                      color=colors[idx % len(colors)],
                      range=(0, 1))
        
        # Add title and labels
        axes[idx].set_title(f'ε={eps} (Avg Score: {avg_score:.4f})')
        axes[idx].set_xlabel('Transport Score (Normalized)')
        axes[idx].set_ylabel('Density')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 1)
        
        score_text += f"ε={eps}: {avg_score:.4f}\n"
    
    # Save plot
    plot_path = os.path.join(save_dir, 'neural_ot_membership_inference.png')
    scores_path = os.path.join(save_dir, 'neural_ot_membership_scores.txt')
    
    print(f"\nSaving plot to: {os.path.abspath(plot_path)}")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saving scores to: {os.path.abspath(scores_path)}")
    with open(scores_path, 'w') as f:
        f.write(score_text)
    
    print("\nSaved files:")
    print(f"1. Plot: {plot_path}")
    print(f"2. Scores: {scores_path}")
    
    # Also save to results directory for consistency
    alt_save_dir = 'results'
    if os.path.exists(alt_save_dir):
        alt_plot_path = os.path.join(alt_save_dir, 'neural_ot_membership_inference.png')
        alt_scores_path = os.path.join(alt_save_dir, 'neural_ot_membership_scores.txt')
        
        print(f"\nAlso saving copies to results directory:")
        plt.savefig(alt_plot_path, bbox_inches='tight', dpi=300)
        with open(alt_scores_path, 'w') as f:
            f.write(score_text)
        print(f"1. Plot: {alt_plot_path}")
        print(f"2. Scores: {alt_scores_path}")

    return score_text

# Update main execution block
if __name__ == "__main__":
    # Load and prepare data
    x_train = pd.read_excel('predictions/processed_x_train_data.xlsx')
    x_test = pd.read_excel('predictions/processed_x_test_data.xlsx')
    y_train = pd.read_excel('predictions/processed_y_train_data.xlsx')
    y_test = pd.read_excel('predictions/processed_y_test_data.xlsx')

    # Split into source and target domains
    source_indices_train = x_train['domain'] == 'source' 
    source_indices_test = x_test['domain'] == 'source'
    target_indices = x_test['domain'] == 'target'

    y_source_train = y_train.loc[source_indices_train, 'disease'].to_numpy()
    y_source_test = y_test.loc[source_indices_test, 'disease'].to_numpy()
    y_target = y_test.loc[target_indices, 'disease'].to_numpy()

    # Select feature columns and convert to float
    feature_cols = [col for col in x_train.columns if col not in ['domain']]
    
    # Explicitly convert to float32
    x_source_train = x_train.loc[source_indices_train, feature_cols].astype(np.float32).to_numpy()
    x_source_test = x_test.loc[source_indices_test, feature_cols].astype(np.float32).to_numpy()
    x_target = x_test.loc[target_indices, feature_cols].astype(np.float32).to_numpy()

    # Print shapes and check for NaN values
    print("Data shapes:")
    print(f"x_source_train: {x_source_train.shape}")
    print(f"x_target: {x_target.shape}")
    
    # Check for NaN values
    print("\nChecking for NaN values:")
    print(f"NaN in source train: {np.isnan(x_source_train).any()}")
    print(f"NaN in target: {np.isnan(x_target).any()}")

    # Normalize the data
    scaler = StandardScaler()
    x_source_train = scaler.fit_transform(x_source_train)
    x_target = scaler.transform(x_target)

    # Convert to PyTorch tensors
    x_source_train = torch.FloatTensor(x_source_train)
    x_target = torch.FloatTensor(x_target)

    # Define epsilon values to test
    epsilon_values = [0.5, 1.0, 5.0, 10.0, 20.0]
    
    # Dictionary to store outputs for each epsilon
    all_outputs = {}
    
    for eps in epsilon_values:
        print(f"\nTesting epsilon = {eps}")
        
        # Update model parameters for this epsilon
        current_params = {
            'input_dim': x_source_train.shape[1],
            'projection_dim': 256,
            'D': 0.05,
            'eps1': eps/2,
            'eps2': eps/2,
            'delta1': 1e-5,
            'delta2': 1e-5,
            'delta_prime': 1e-5,
            'batch_size': 32,
            'num_iterations': 100,
            'learning_rate': 0.00075,
            'clip_norm': 0.05,
            'ell': 0.0001
        }

        # Train model
        print(f"Training Neural OT model with ε={eps}...")
        model, X_target_transformed = train_dp_neural_ot(x_target, x_source_train, current_params)
        
        # Get membership inference outputs
        print("Performing membership inference analysis...")
        outputs = analyze_membership_inference(
            model, 
            x_source_train,
            x_target,
            eps
        )
        all_outputs[eps] = outputs
        
        print(f"Completed analysis for ε={eps}")
    
    # Plot comparison of all epsilon values
    print("\nGenerating membership inference plots...")
    plot_membership_inference_comparison(all_outputs, epsilon_values)
    print("Plots saved in neural_membership_analysis/membership_inference_all_eps.png")
    
  