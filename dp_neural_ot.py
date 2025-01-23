import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define the neural network for potentials φ_θ and ψ_η
class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(ICNN, self).__init__()
        # Simplified architecture with just one hidden layer
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units[0], 1)
        )

    def forward(self, x):
        return self.net(x)

    def transport(self, x):
        """
        Compute the transport map gradient.
        
        Parameters:
        - x: Input tensor with requires_grad=True
        
        Returns:
        - Gradient of the potential function
        """
        assert x.requires_grad
        grad_outputs = torch.ones_like(self.forward(x))
        grad = torch.autograd.grad(
            outputs=self.forward(x),
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True
        )[0]
        return grad


def dp_neural_optimal_transport(source_data, target_data, noise_std, scaling_l, 
                                projection_dim, penalty_lambda, hidden_dim, 
                                learning_rate, num_iterations, batch_size, 
                                target_epsilon, delta, noise_multiplier=1.0):
    """
    Differentially Private Neural Optimal Transport
    """
    # Step 1: Generate random projection matrix
    d = source_data.shape[1]
    M = torch.randn(d, projection_dim) * (1 / projection_dim)**0.5

    # Step 2: Transform source and target data
    Xs = source_data @ M
    Xt = target_data @ M

    # Step 3: Add noise to source data
    noise = torch.randn_like(Xs) * noise_std
    Xs_noisy = Xs + noise

    # Step 4: Initialize cost matrix
    C = torch.cdist(Xs_noisy, Xt, p=2)**2
    C_tilde = C - scaling_l * (noise_std**2)

    # Step 5: Initialize neural networks
    phi_theta = ICNN(
        input_dim=projection_dim,
        hidden_units=[hidden_dim]
    )
    psi_eta = ICNN(
        input_dim=projection_dim,
        hidden_units=[hidden_dim]
    )

    # Make models compatible with Opacus
    phi_theta = ModuleValidator.fix(phi_theta)
    psi_eta = ModuleValidator.fix(psi_eta)

    # Create optimizers
    optimizer_phi = optim.Adam(phi_theta.parameters(), lr=learning_rate)
    optimizer_psi = optim.Adam(psi_eta.parameters(), lr=learning_rate)

    # Create privacy engines
    privacy_engine_phi = PrivacyEngine()
    privacy_engine_psi = PrivacyEngine()

    # Attach privacy engines
    phi_theta, optimizer_phi, train_loader_phi = privacy_engine_phi.make_private(
        module=phi_theta,
        optimizer=optimizer_phi,
        data_loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xs_noisy),
            batch_size=batch_size,
            shuffle=True
        ),
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0
    )

    psi_eta, optimizer_psi, train_loader_psi = privacy_engine_psi.make_private(
        module=psi_eta,
        optimizer=optimizer_psi,
        data_loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt),
            batch_size=batch_size,
            shuffle=True
        ),
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0
    )

    # Initialize lists to store metrics
    loss_history = []
    wasserstein_history = []

    # Training loop
    for iteration in range(num_iterations):
        total_loss = 0
        total_wasserstein = 0
        n_batches = 0
        
        for (xs_batch,), (xt_batch,) in zip(train_loader_phi, train_loader_psi):
            optimizer_phi.zero_grad()
            optimizer_psi.zero_grad()

            # Compute potentials
            phi_s = phi_theta(xs_batch).squeeze()
            psi_t = psi_eta(xt_batch).squeeze()

            # Compute pairwise constraint violations
            C_batch = torch.cdist(xs_batch, xt_batch, p=2)**2
            C_tilde_batch = C_batch - scaling_l * (noise_std**2)
            pairwise_constraints = phi_s[:, None] + psi_t[None, :] - C_tilde_batch
            positive_part = torch.relu(pairwise_constraints)

            # Modified loss computation
            loss_phi = -phi_s.mean()
            loss_psi = -psi_t.mean()
            constraint_violation = penalty_lambda * positive_part.mean()
            loss = loss_phi + loss_psi + constraint_violation

            # Compute Wasserstein distance
            wasserstein_dist = C_tilde_batch.mean() - (phi_s.mean() + psi_t.mean())

            # Accumulate batch statistics
            total_loss += loss.item()
            total_wasserstein += wasserstein_dist.item()
            n_batches += 1

            # Backward pass and optimization
            loss.backward()
            optimizer_phi.step()
            optimizer_psi.step()

            # Apply weight constraints through gradient clipping instead of clamping
            torch.nn.utils.clip_grad_norm_(phi_theta.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(psi_eta.parameters(), max_norm=1.0)

        # Compute averages
        avg_loss = total_loss / n_batches
        avg_wasserstein = total_wasserstein / n_batches

        # Store metrics
        loss_history.append(avg_loss)
        wasserstein_history.append(avg_wasserstein)

        if iteration % 100 == 0:
            epsilon_phi = privacy_engine_phi.get_epsilon(delta)
            epsilon_psi = privacy_engine_psi.get_epsilon(delta)
            print(f"Iteration {iteration}:")
            print(f"  Loss = {avg_loss:.4f}")
            print(f"  Wasserstein Distance = {avg_wasserstein:.4f}")
            print(f"  ε_phi = {epsilon_phi:.2f}, ε_psi = {epsilon_psi:.2f}")
            print("-" * 50)

    return phi_theta, psi_eta, C_tilde, loss_history, wasserstein_history, M


def plot_and_save_metrics(loss_history, wasserstein_history, loss_filename='loss_history.png', wasserstein_filename='wasserstein_history.png'):
    """
    Plots and saves the loss and Wasserstein distance histories.

    Parameters:
    - loss_history: List of loss values over iterations.
    - wasserstein_history: List of Wasserstein distance values over iterations.
    - loss_filename: Filename for saving the loss plot.
    - wasserstein_filename: Filename for saving the Wasserstein distance plot.
    """
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_filename)
    plt.close()

    # Plot Wasserstein distance history
    plt.figure(figsize=(10, 6))
    plt.plot(wasserstein_history, label='Wasserstein Distance', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(wasserstein_filename)
    plt.close()


def plot_loss_components(phi_losses, psi_losses, constraint_violations, save_path='loss_components.png'):
    """Visualize different components of the loss function"""
    plt.figure(figsize=(12, 6))
    plt.plot(phi_losses, label='φ Network Loss', alpha=0.7)
    plt.plot(psi_losses, label='ψ Network Loss', alpha=0.7)
    plt.plot(constraint_violations, label='Constraint Violations', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.title('Components of the Loss Function')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_optimal_transport(source_data, target_data, phi_theta, M, 
                              C_tilde, save_dir='ot_visualizations/'):
    """
    Create various visualizations for optimal transport results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Cost Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(C_tilde.detach().numpy(), cmap='viridis')
    plt.title('Optimal Transport Cost Matrix')
    plt.xlabel('Target Points')
    plt.ylabel('Source Points')
    plt.savefig(os.path.join(save_dir, 'cost_matrix_heatmap.png'))
    plt.close()
    
    # 2. Projected Data Visualization
    # Project source and target data
    source_proj = source_data @ M
    target_proj = target_data @ M
    
    # Get transformed representations
    source_transformed = phi_theta(source_proj).detach().numpy()
    target_transformed = phi_theta(target_proj).detach().numpy()
    
    # Create scatter plot using first two dimensions of projected space
    plt.figure(figsize=(12, 8))
    plt.scatter(source_proj[:, 0].detach().numpy(), 
               source_proj[:, 1].detach().numpy(), 
               c='blue', alpha=0.6, label='Source')
    plt.scatter(target_proj[:, 0].detach().numpy(), 
               target_proj[:, 1].detach().numpy(), 
               c='red', alpha=0.6, label='Target')
    
    # Draw transport paths (for subset of points to avoid cluttering)
    n_paths = min(50, len(source_proj))
    for i in range(n_paths):
        plt.arrow(source_proj[i, 0].item(), source_proj[i, 1].item(),
                 target_proj[i % len(target_proj), 0].item() - source_proj[i, 0].item(),
                 target_proj[i % len(target_proj), 1].item() - source_proj[i, 1].item(),
                 alpha=0.2, head_width=0.05)
    
    plt.title('Projected Space Transport Paths')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'transport_paths.png'))
    plt.close()
    
    # 3. Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before transport (using first projected dimension)
    sns.kdeplot(data=source_proj[:, 0].detach().numpy(), ax=axes[0], label='Source', color='blue')
    sns.kdeplot(data=target_proj[:, 0].detach().numpy(), ax=axes[0], label='Target', color='red')
    axes[0].set_title('Distribution Before Transport')
    axes[0].legend()
    
    # After transport
    sns.kdeplot(data=source_transformed.flatten(), ax=axes[1], label='Transported Source', color='blue')
    sns.kdeplot(data=target_transformed.flatten(), ax=axes[1], label='Target', color='red')
    axes[1].set_title('Distribution After Transport')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'))
    plt.close()
    
    # 4. Feature-wise Transport Effect
    n_features = min(5, M.shape[1])  # Show first 5 features of projected space
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3*n_features))
    
    if n_features == 1:
        axes = [axes]  # Make axes iterable when there's only one feature
        
    for i in range(n_features):
        sns.kdeplot(data=source_proj[:, i].detach().numpy(), ax=axes[i], label='Source', color='blue')
        sns.kdeplot(data=target_proj[:, i].detach().numpy(), ax=axes[i], label='Target', color='red')
        axes[i].set_title(f'Projected Feature {i+1} Distribution')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_wise_transport.png'))
    plt.close()


def analyze_feature_transport(source_data, target_data, phi_theta, M, feature_cols):
    """
    Analyzes the effectiveness of transport for each feature by comparing
    Wasserstein distances before and after transport.
    
    Parameters:
    - source_data: Original source domain data (torch.Tensor)
    - target_data: Original target domain data (torch.Tensor)
    - phi_theta: Trained potential network
    - M: Projection matrix
    - feature_cols: List of feature names
    
    Returns:
    - DataFrame with Wasserstein distances and improvement metrics for each feature
    """
    # Project data
    source_proj = source_data @ M
    target_proj = target_data @ M
    
    # Get transformed representations
    source_transformed = phi_theta(source_proj).detach()
    target_transformed = phi_theta(target_proj).detach()
    
    results = []
    
    # Calculate Wasserstein distance for each feature before and after transport
    for i, feature_name in enumerate(feature_cols):
        # Original distributions
        source_feat = source_data[:, i].detach().numpy()
        target_feat = target_data[:, i].detach().numpy()
        
        # Calculate empirical Wasserstein distance before transport
        # Using sorted values difference as a proxy for 1D Wasserstein distance
        orig_wasserstein = np.mean(np.abs(np.sort(source_feat) - np.sort(target_feat)))
        
        # Calculate Wasserstein distance after transport
        source_trans = source_transformed[:, 0].numpy()  # Using first dimension of transformation
        target_trans = target_transformed[:, 0].numpy()
        trans_wasserstein = np.mean(np.abs(np.sort(source_trans) - np.sort(target_trans)))
        
        # Calculate improvement
        improvement = (orig_wasserstein - trans_wasserstein) / orig_wasserstein * 100
        
        results.append({
            'Feature': feature_name,
            'Original_Wasserstein': orig_wasserstein,
            'Transformed_Wasserstein': trans_wasserstein,
            'Improvement_Percentage': improvement
        })
    
    # Create DataFrame and sort by improvement
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Improvement_Percentage', ascending=False)
    
    return results_df


def transport_data(X, model_phi, M):
    """
    Compute the transport map for new data points X using the trained model_phi.
    Compatible with Opacus privacy wrapper.
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    X_proj = X @ M
    
    # Create a clean copy of the model without Opacus
    if hasattr(model_phi, '_module'):
        # Create new ICNN instance
        clean_model = ICNN(
            input_dim=model_phi._module.net[0].in_features,
            hidden_units=[model_phi._module.net[0].out_features]
        )
        # Copy state dict from original model
        clean_model.load_state_dict(model_phi._module.state_dict())
    else:
        clean_model = model_phi
    
    # Set to eval mode
    clean_model.eval()
    
    try:
        with torch.enable_grad():
            # Enable gradients for input
            X_proj.requires_grad_(True)
            
            # Forward pass
            outputs = clean_model(X_proj)
            
            # Compute gradients
            grad_outputs = torch.ones_like(outputs)
            gradients = torch.autograd.grad(
                outputs=outputs,
                inputs=X_proj,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Transported data in projected space
            X_transported_proj = X_proj - gradients
            # Map back to original space
            X_transported = X_transported_proj @ M.t()
            
    finally:
        # Clean up
        clean_model.train()
        
    return X_transported.detach().numpy()




# Example setup for testing
if __name__ == "__main__":
    # Load and prepare data
    x_train = pd.read_excel('predictions/processed_x_train_data.xlsx')
    x_test = pd.read_excel('predictions/processed_x_test_data.xlsx')
    y_train = pd.read_excel('predictions/processed_y_train_data.xlsx')
    y_test = pd.read_excel('predictions/processed_y_test_data.xlsx')

    x_source_indices_train = x_train['domain'] == 'source'
    x_target_indices_train = x_train['domain'] == 'target'
    x_target_indices_test = x_test['domain'] == 'target'

    y_source = y_train.loc[x_source_indices_train, 'disease'].to_numpy()
    y_target = y_test.loc[x_target_indices_test, 'disease'].to_numpy()

    feature_cols = [col for col in x_train.columns if col not in ['domain']]
    x_source_train = x_train.loc[x_source_indices_train, feature_cols].to_numpy()
    x_target_train = x_train.loc[x_target_indices_train, feature_cols].to_numpy()
    x_target_test = x_test.loc[x_target_indices_test, feature_cols].to_numpy()

    # Convert data to float type before creating tensors
    x_source_train = x_source_train.astype(np.float32)
    x_target_train = x_target_train.astype(np.float32)
    x_target_test = x_target_test.astype(np.float32)

    # Convert to PyTorch tensors
    x_source = torch.FloatTensor(x_source_train)
    x_target = torch.FloatTensor(x_target_train)
    x_target_test = torch.FloatTensor(x_target_test)
    
    print(f"Source data shape: {x_source.shape}")
    print(f"Target data shape: {x_target.shape}")

    # Updated parameters for Opacus
    noise_std = 0
    scaling_l = 0.5
    projection_dim = 100
    penalty_lambda = 1.0
    hidden_dim = 64
    learning_rate = 1e-3
    num_iterations = 1000
    batch_size = 32
    target_epsilon = 0 # Privacy budget
    target_epsilon = 0 # Privacy budget
    delta = 1e-5         # Privacy delta

    # Run the algorithm with Opacus parameters and correctly unpack all returned values
    phi_theta, psi_eta, C_tilde, loss_history, wasserstein_history, M = dp_neural_optimal_transport(
        x_source, x_target, noise_std, scaling_l,
        projection_dim, penalty_lambda, hidden_dim,
        learning_rate, num_iterations, batch_size,
        target_epsilon, delta
    )
    
    # Plot and save the metrics
    plot_and_save_metrics(loss_history, wasserstein_history)

    print("\nPlots have been saved as 'loss_history.png' and 'wasserstein_history.png'")

    print("\nApplying transport to target data...")
    try:
        # Access the original model if wrapped by Opacus
        if hasattr(phi_theta, '_module'):
            print("Using clean model copy for transport...")
            
        transported_target_data = transport_data(x_target_test, phi_theta, M)
        print(f"Successfully transported data with shape: {transported_target_data.shape}")
        
        # Train XGBoost on the transported data
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        
        print("\nTraining XGBoost model...")
        xgb_model.fit(x_source, y_source)
        y_pred = xgb_model.predict(transported_target_data)
        
        # Evaluate
        accuracy = accuracy_score(y_target, y_pred)
        precision = precision_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)
        
        print("\nModel Performance on Transported Target Data:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
    except Exception as e:
        print(f"Error during transport: {str(e)}")
        import traceback
        traceback.print_exc()

 