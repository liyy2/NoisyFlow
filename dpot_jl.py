import numpy as np
import ot
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class DPOT_JL:
    def __init__(self, epsilon, delta, projection_dim, regularization=0.1):
        self.epsilon = epsilon
        self.delta = delta
        self.projection_dim = projection_dim
        self.regularization = regularization

    def _generate_jl_matrix(self, input_dim):
        return np.random.normal(0, 1 / self.projection_dim, (input_dim, self.projection_dim))

    def _generate_noise_matrix(self, M, shape):
        w = np.sqrt(np.max(np.sum(M**2, axis=1)))
        sigma = w * np.sqrt(2 * (np.log(1/self.delta) + self.epsilon) / self.epsilon)
        return np.random.normal(0, sigma, shape)

    def _get_sigma(self, M):
        # Increase the noise scale to decrease accuracy
        w = np.sqrt(np.max(np.sum(M**2, axis=1)))
        return w * np.sqrt(2 * np.log(1/self.delta) + self.epsilon) / self.epsilon

    def fit_transform(self, X_source, X_target, max_iterations=2000):
        X_source = np.asarray(X_source, dtype=np.float64)
        X_target = np.asarray(X_target, dtype=np.float64)

        n_source, input_dim = X_source.shape
        n_target = X_target.shape[0]

        self.projection_matrix = self._generate_jl_matrix(input_dim)
        
        # Increase noise magnitude
        noise_matrix = self._generate_noise_matrix(self.projection_matrix, (n_target, self.projection_dim))
        
        X_source_projected = X_source @ self.projection_matrix
        X_target_projected = X_target @ self.projection_matrix
        X_target_noisy = X_target_projected + noise_matrix

        # Calculate true Euclidean distances for Wasserstein distance
        diff = X_target_noisy[:, np.newaxis, :] - X_source_projected[np.newaxis, :, :]
        cost_matrix = np.sum(diff**2, axis=2)  # Shape: (n_target, n_source)
        
        sigma = self._get_sigma(self.projection_matrix)

        # Don't subtract noise variance to keep true distances
        cost_matrix -= self.projection_dim * (sigma**2)  # Remove this line

        # Add numerical stability while preserving relative distances
        cost_matrix = np.clip(cost_matrix, 0, None)  # Only clip negative values
        
        # Scale the cost matrix to prevent numerical overflow
        scale_factor = np.max(np.abs(cost_matrix))
        if scale_factor > 0:
            cost_matrix = cost_matrix / scale_factor
        
        u = np.ones(n_target) / n_target
        v = np.ones(n_source) / n_source

        # Track iterations
        wasserstein_distances = []
        losses = []
        
        def callback(current_P):
            # Calculate current Wasserstein distance (total, not average)
            current_wd = np.sum(current_P * cost_matrix) * scale_factor
            wasserstein_distances.append(current_wd)
            
            # Calculate current loss (entropy regularized OT objective)
            current_loss = np.sum(current_P * cost_matrix) * scale_factor + \
                          self.regularization * np.sum(current_P * np.log(current_P + 1e-10))
            losses.append(current_loss)
            return False

        transport_plan = ot.sinkhorn(
            u, v,
            cost_matrix,
            reg=self.regularization,
            method='sinkhorn_stabilized',
            numItermax=max_iterations,
            stopThr=1e-6,
            verbose=True,
            callback=callback
        )

        # Handle numerical issues in transport plan
        transport_plan = np.nan_to_num(transport_plan)
        transport_plan = np.clip(transport_plan, 0, 1)  # Ensure valid probabilities
        
        # Normalize transport plan
        row_sums = transport_plan.sum(axis=1, keepdims=True)
        transport_plan = transport_plan / (row_sums + 1e-10)
        
        # Calculate true Wasserstein distance (total instead of mean)
        wasserstein_distance = np.sum(transport_plan * cost_matrix) * scale_factor
        wasserstein_std = np.std(transport_plan * cost_matrix) * scale_factor
        
        return transport_plan, wasserstein_distance, wasserstein_std, cost_matrix, wasserstein_distances, losses

    def transport_data(self, X_target, transport_plan):
        """
        Transport the target data using the computed transport plan to align with target domain.
        
        Args:
            X_target: Target dataset of shape (n_target, n_features)
            transport_plan: Transport plan of shape (n_target, n_source)
            
        Returns:
            X_transformed: Transported target data of shape (n_target, n_features)
        """
        # transport_plan shape: (n_target, n_source)
        # X_target shape: (n_target, n_features)
        # For correct multiplication, we need to transpose the transport plan
        X_transformed = transport_plan.T @ X_target
        
        return X_transformed


def save_metrics_to_json(results_df, wasserstein_histories, loss_histories):
    """Save metrics to JSON files."""
    # Create results directory if it doesn't exist
    os.makedirs('jl_results', exist_ok=True)
    
    # Prepare wasserstein data
    wasserstein_data = {
        str(eps): {
            'final_distance': float(row['wasserstein_distance']),
            'std': float(row['wasserstein_std']),
            'all_distances': [float(d) for d in wasserstein_histories[i]]
        } for i, (eps, row) in enumerate(results_df.iterrows())
    }
    
    # Prepare accuracy data
    accuracy_data = {
        str(eps): {
            'accuracy': float(row['accuracy']),
            'precision': float(row['precision']),
            'recall': float(row['recall']),
            'f1': float(row['f1'])
        } for eps, row in results_df.iterrows()
    }
    
    # Prepare loss data
    loss_data = {
        str(eps): {
            'all_losses': [float(l) for l in loss_histories[i]]
        } for i, eps in enumerate(results_df['epsilon'])
    }
    
    # Save to JSON files
    with open('jl_results/wasserstein_distances.json', 'w') as f:
        json.dump(wasserstein_data, f, indent=4)
    
    with open('jl_results/classification_metrics.json', 'w') as f:
        json.dump(accuracy_data, f, indent=4)
    
    with open('jl_results/loss_histories.json', 'w') as f:
        json.dump(loss_data, f, indent=4)
    
    print("\nMetrics saved to JSON files in jl_results/")


def visualize_transport_plans(results_df, transport_plans, epsilons):
    """
    Create a grid visualization of transport plans for different epsilon values.
    
    Args:
        results_df: DataFrame containing results for each epsilon
        transport_plans: List of transport plan matrices
        epsilons: List of epsilon values used
    """
    # Calculate grid dimensions
    n_plots = len(epsilons)
    n_cols = min(3, n_plots)  # Maximum 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    
    for idx, (epsilon, transport_plan) in enumerate(zip(epsilons, transport_plans)):
        # Get corresponding metrics
        metrics = results_df[results_df['epsilon'] == epsilon].iloc[0]
        
        # Create subplot
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Plot heatmap
        sns.heatmap(transport_plan, cmap='viridis', cbar_kws={'label': 'Transport Weight'})
        
        # Add title with metrics
        plt.title(f'Transport Plan (ε={epsilon})\n' + 
                 f'W-dist: {metrics["wasserstein_distance"]:.2f}\n' +
                 f'Accuracy: {metrics["accuracy"]:.2f}')
        plt.xlabel('Source Points')
        plt.ylabel('Target Points')
    
    plt.tight_layout()
    plt.savefig('jl_results/transport_plans_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_transport_analysis(transport_plan, X_source, X_target, epsilon, save_dir='jl_results'):
    """
    Create multiple visualizations of the optimal transport plan.
    
    Args:
        transport_plan: The computed transport plan matrix
        X_source: Source domain data
        X_target: Target domain data
        epsilon: Privacy parameter epsilon
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Transport Plan Heatmap with Marginals
    plt.figure(figsize=(12, 12))
    
    # Main heatmap
    gs = plt.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:, :-1])
    ax_right = plt.subplot(gs[1:, -1])
    ax_top = plt.subplot(gs[0, :-1])
    
    # Plot main heatmap with corrected cbar parameters
    sns.heatmap(transport_plan, cmap='viridis', ax=ax_main, 
                cbar_kws={'label': 'Transport Weight'})  # Changed this line
    
    # Plot marginals
    row_sums = transport_plan.sum(axis=1)
    col_sums = transport_plan.sum(axis=0)
    
    ax_top.plot(col_sums, color='blue')
    ax_right.plot(row_sums, range(len(row_sums)), color='red')
    
    ax_top.set_title(f'Transport Plan with Marginals (ε={epsilon})')
    plt.savefig(f'{save_dir}/transport_plan_marginals_eps_{epsilon}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Rest of the visualizations remain the same...


def analyze_membership_inference_dpot(transport_plan, x_target_test, epsilon):
    """
    Analyze membership inference risk for target test data using entropy of transport plan rows.
    
    Args:
        transport_plan: The computed transport plan
        x_target_test: Test data from target domain
        epsilon: Privacy parameter epsilon
    Returns:
        scores: Normalized entropy scores for target test data
    """
    # Calculate entropy for each row in transport plan
    # Higher entropy means more uniform/random transport (more private)
    # Lower entropy means more concentrated transport (less private)
    eps = 1e-10  # Small constant to avoid log(0)
    entropies = -np.sum(transport_plan * np.log(transport_plan + eps), axis=1)
    
    # Normalize entropies to [0,1] range
    scores = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
    
    # Print additional debug info
    print(f"Raw entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")
    print(f"Transport plan stats:")
    print(f"- Min value: {transport_plan.min():.4f}")
    print(f"- Max value: {transport_plan.max():.4f}")
    print(f"- Mean value: {transport_plan.mean():.4f}")
    
    return scores

def plot_dpot_membership_inference(all_outputs, epsilons, x_target_test, save_dir='member_ship_dp_ot'):
    """
    Plot membership inference analysis for DP-OT in separate subplots.
    Save results in the same directory as other results.
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
    
    for idx, (eps, transport_plan) in enumerate(zip(epsilons, transport_plans)):
        # Calculate membership inference scores for target test data
        scores = analyze_membership_inference_dpot(transport_plan, x_target_test, eps)
        avg_score = scores.mean()
        
        # Print debug information
        print(f"\nEpsilon {eps}:")
        print(f"Transport plan shape: {transport_plan.shape}")
        print(f"Target test data shape: {x_target_test.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Average score: {avg_score:.4f}")
        
        # Plot histogram
        axes[idx].hist(scores, 
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
    
    plt.tight_layout()
    
    # Save with clear filenames
    plot_path = os.path.join(save_dir, 'dpot_membership_inference.png')
    scores_path = os.path.join(save_dir, 'dpot_membership_scores.txt')
    
    print(f"Saving plot to: {os.path.abspath(plot_path)}")
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
        alt_plot_path = os.path.join(alt_save_dir, 'dpot_membership_inference.png')
        alt_scores_path = os.path.join(alt_save_dir, 'dpot_membership_scores.txt')
        
        print(f"\nAlso saving copies to results directory:")
        plt.savefig(alt_plot_path, bbox_inches='tight', dpi=300)
        with open(alt_scores_path, 'w') as f:
            f.write(score_text)
        print(f"1. Plot: {alt_plot_path}")
        print(f"2. Scores: {alt_scores_path}")

    return score_text


if __name__ == "__main__":
    np.random.seed(42)

    # Load and prepare data
    x_train = pd.read_excel('predictions/processed_x_train_data.xlsx')
    x_test = pd.read_excel('predictions/processed_x_test_data.xlsx')
    y_train = pd.read_excel('predictions/processed_y_train_data.xlsx')
    y_test = pd.read_excel('predictions/processed_y_test_data.xlsx')

    source_indices_train = x_train['domain'] == 'source' 
    source_indices_test = x_test['domain'] == 'source'
    target_indices = x_test['domain'] == 'target'

    y_source_train = y_train.loc[source_indices_train, 'disease'].to_numpy()
    y_source_test = y_test.loc[source_indices_test, 'disease'].to_numpy()

    y_target = y_test.loc[target_indices, 'disease'].to_numpy()

    feature_cols = [col for col in x_train.columns if col not in ['domain']]
    
    
    x_source_train = x_train.loc[source_indices_train, feature_cols].to_numpy()
    x_source_test = x_test.loc[source_indices_test, feature_cols].to_numpy()
    x_target = x_test.loc[target_indices, feature_cols].to_numpy()

    print(f"Source data shape: {x_source_train.shape}")
    print(f"Target data shape: {x_target.shape}")

    # Define epsilon values to test
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []

    # Create figure for iteration plots
    plt.figure(figsize=(15, 6))
    
    # Initialize lists to store histories
    wasserstein_histories = []
    loss_histories = []
    transport_plans = []  # Add this at the start of the main block
    
    # Initialize dictionary to store outputs for each epsilon
    membership_outputs = {}
    is_member = np.zeros(len(x_target))  # Create once, same for all epsilons
    
    for epsilon in epsilon_values:
        print(f"\nTesting epsilon = {epsilon}")
        dpot_jl = DPOT_JL(
            epsilon=epsilon,
            delta=1e-5,
            projection_dim=64,
            regularization=0.01
        )
        
        # Get transport plan and iteration history
        transport_plan, wasserstein_distance, wasserstein_std, cost_matrix, \
        wasserstein_history, loss_history = dpot_jl.fit_transform(x_source_train, x_target)
        
        # Add visualization of transport plan
        plt.figure(figsize=(10, 8))
        sns.heatmap(transport_plan, cmap='viridis', cbar_kws={'label': 'Transport Weight'})
        plt.title(f'Optimal Transport Plan (ε={epsilon})')
        plt.xlabel('Source Points')
        plt.ylabel('Target Points')
        plt.savefig(f'jl_results/transport_plan_eps_{epsilon}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot iteration histories
        plt.subplot(1, 2, 1)
        plt.plot(wasserstein_history, label=f'ε={epsilon}')
        plt.xlabel('Iteration')
        plt.ylabel('Total Wasserstein Distance')
        plt.title('Wasserstein Distance vs Iterations for DPOT-JL')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss_history, label=f'ε={epsilon}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Iterations')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        # Transform the target data
        x_target_transported = dpot_jl.transport_data(x_target, transport_plan)

        # Train and evaluate
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )

        xgb_model.fit(x_source_train, y_source_train)
        y_pred = xgb_model.predict(x_target)

        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_target, y_pred)
        accuracy = accuracy_score(y_target, y_pred)

        # Print results for this epsilon
        print(f"Wasserstein Distance: {wasserstein_distance:.4f} ± {wasserstein_std:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {np.mean(precision):.4f}")
        print(f"Recall: {np.mean(recall):.4f}")
        print(f"F1 Score: {np.mean(f1):.4f}")

        # Store results
        results.append({
            'epsilon': epsilon,
            'wasserstein_distance': wasserstein_distance,
            'wasserstein_std': wasserstein_std,
            'accuracy': accuracy,
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1)
        })

        # Store histories
        wasserstein_histories.append(wasserstein_history)
        loss_histories.append(loss_history)
        transport_plans.append(transport_plan)
        
        # Store outputs for this epsilon
        target_outputs = analyze_membership_inference_dpot(
            transport_plan,
            x_target,
            epsilon
        )
        membership_outputs[epsilon] = target_outputs
    
    plt.tight_layout()
    plt.savefig('jl_results/iteration_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    save_metrics_to_json(results_df, wasserstein_histories, loss_histories)

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.plot(results_df['epsilon'], results_df['wasserstein_distance'], marker='o')
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Wasserstein Distance')
    ax1.set_xscale('log')
    ax1.set_title('Epsilon vs Wasserstein Distance for DPOT-JL')

    ax2.plot(results_df['epsilon'], results_df['accuracy'], marker='o')
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Accuracy')
    ax2.set_xscale('log')
    ax2.set_title('Epsilon vs Accuracy for DPOT-JL')

    ax3.plot(results_df['epsilon'], results_df['precision'], marker='o', label='Precision')
    ax3.plot(results_df['epsilon'], results_df['recall'], marker='o', label='Recall')
    ax3.set_xlabel('Epsilon')
    ax3.set_ylabel('Score')
    ax3.set_xscale('log')
    ax3.set_title('Epsilon vs Precision/Recall for DPOT-JL')
    ax3.legend()

    ax4.plot(results_df['epsilon'], results_df['f1'], marker='o')
    ax4.set_xlabel('Epsilon')
    ax4.set_ylabel('F1 Score')
    ax4.set_xscale('log')
    ax4.set_title('Epsilon vs F1 Score for DPOT-JL')

    plt.tight_layout()
    plt.savefig('jl_results/epsilon_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize transport plans
    visualize_transport_plans(results_df, transport_plans, epsilon_values)

    # Visualize transport analysis
    for eps in epsilon_values:
        visualize_transport_analysis(transport_plan, x_source_train, x_target, eps)

    # Plot combined membership inference analysis after the loop
    plot_dpot_membership_inference(membership_outputs, epsilon_values, x_target)


