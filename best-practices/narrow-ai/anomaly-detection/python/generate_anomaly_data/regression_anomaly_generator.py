import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
n_samples = 1000
n_features = 5
noise_level = 0.5
n_anomalies = 30  # Number of anomalies to inject
anomaly_magnitude_factor = 3  # How many times the noise std dev

# Feature names
feature_names = [f'feature_{i+1}' for i in range(n_features)]

# Random seed for reproducibility
random_seed = 42

print(f"Generating regression dataset with {n_samples} samples and {n_features} features.")
print(f"Noise Level (Std Dev): {noise_level}")
print(f"Number of Anomalies: {n_anomalies}, Magnitude Factor: {anomaly_magnitude_factor}")

np.random.seed(random_seed)

# --- Generate Features ---
# Create correlated features with some structure
# Base features
X = np.random.randn(n_samples, n_features)

# Add some correlation between features
X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)
X[:, 2] = 0.5 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples)

# Scale features to have different ranges
scales = np.array([10, 20, 5, 15, 8])
offsets = np.array([50, 100, 25, 75, 40])
for i in range(n_features):
    X[:, i] = X[:, i] * scales[i] + offsets[i]

# --- Generate Target Variable ---
# True relationship: combination of linear and nonlinear terms
true_coefficients = np.array([0.5, -0.3, 0.8, 0.2, -0.4])
target = np.dot(X, true_coefficients)

# Add some nonlinear components
target += 0.01 * (X[:, 0] ** 2)  # Quadratic term
target += 0.5 * X[:, 1] * X[:, 2] / 100  # Interaction term
target += 2 * np.sin(X[:, 3] / 15)  # Periodic component

# Add Gaussian noise
noise = np.random.normal(loc=0, scale=noise_level * np.std(target), size=n_samples)
target_with_noise = target + noise

print(f"\nTarget variable generated with linear and nonlinear components.")
print(f"Target standard deviation: {np.std(target_with_noise):.2f}")

# --- Inject Anomalies ---
# We'll inject different types of anomalies:
# 1. Outliers in target values (vertical outliers)
# 2. Outliers in feature space (leverage points)
# 3. Outliers in both (influential points)

anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
anomaly_types = np.random.choice(['target', 'leverage', 'influential'], 
                                size=n_anomalies, 
                                p=[0.4, 0.3, 0.3])

anomalies_mask = np.zeros(n_samples, dtype=int)
anomalies_mask[anomaly_indices] = 1

# Copy arrays for anomalous data
X_anomalous = X.copy()
target_anomalous = target_with_noise.copy()

for idx, anomaly_type in zip(anomaly_indices, anomaly_types):
    if anomaly_type == 'target':
        # Vertical outlier: only change target
        direction = np.random.choice([-1, 1])
        target_anomalous[idx] += direction * anomaly_magnitude_factor * np.std(target_with_noise)
    
    elif anomaly_type == 'leverage':
        # Leverage point: outlier in feature space
        # Move point far from center in feature space
        feature_to_modify = np.random.choice(n_features, size=2, replace=False)
        for feat in feature_to_modify:
            direction = np.random.choice([-1, 1])
            X_anomalous[idx, feat] += direction * anomaly_magnitude_factor * np.std(X[:, feat])
    
    else:  # influential
        # Influential point: outlier in both feature and target space
        # Modify features
        feature_to_modify = np.random.choice(n_features, size=2, replace=False)
        for feat in feature_to_modify:
            direction = np.random.choice([-1, 1])
            X_anomalous[idx, feat] += direction * anomaly_magnitude_factor * np.std(X[:, feat])
        # Modify target
        direction = np.random.choice([-1, 1])
        target_anomalous[idx] += direction * anomaly_magnitude_factor * np.std(target_with_noise)

print(f"\nInjected {n_anomalies} anomalies:")
print(f"- Target outliers: {sum(anomaly_types == 'target')}")
print(f"- Leverage points: {sum(anomaly_types == 'leverage')}")
print(f"- Influential points: {sum(anomaly_types == 'influential')}")

# --- Create DataFrame and Save ---
# Combine features and target
data_dict = {feat: X_anomalous[:, i] for i, feat in enumerate(feature_names)}
data_dict['target'] = target_anomalous
data_dict['is_anomaly'] = anomalies_mask
data_dict['anomaly_type'] = ['normal'] * n_samples

# Set anomaly types
for idx, atype in zip(anomaly_indices, anomaly_types):
    data_dict['anomaly_type'][idx] = atype

df = pd.DataFrame(data_dict)

# Define filename
csv_filename = 'synthetic_regression_anomaly_data.csv'

# Save to CSV
df.to_csv(csv_filename, index=False)

print(f"\nDataset saved to '{csv_filename}'")
print(f"\nFirst 5 rows of the generated data:\n{df.head().to_string()}")
print(f"\nAnomaly distribution:")
print(df['anomaly_type'].value_counts())

# --- Visualization ---
try:
    # 1. Scatter plot of first two features with anomalies highlighted
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    normal_mask = df['is_anomaly'] == 0
    anomaly_mask = df['is_anomaly'] == 1
    
    plt.scatter(df.loc[normal_mask, 'feature_1'], 
                df.loc[normal_mask, 'feature_2'], 
                c='blue', alpha=0.5, label='Normal', s=30)
    
    # Plot different anomaly types with different colors
    colors = {'target': 'red', 'leverage': 'orange', 'influential': 'purple'}
    for atype, color in colors.items():
        mask = df['anomaly_type'] == atype
        if mask.any():
            plt.scatter(df.loc[mask, 'feature_1'], 
                       df.loc[mask, 'feature_2'], 
                       c=color, label=f'{atype.capitalize()} anomaly', 
                       s=100, marker='x', linewidth=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Feature Space View (Features 1 & 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Target vs Feature 1 with anomalies
    plt.subplot(1, 3, 2)
    plt.scatter(df.loc[normal_mask, 'feature_1'], 
                df.loc[normal_mask, 'target'], 
                c='blue', alpha=0.5, label='Normal', s=30)
    
    for atype, color in colors.items():
        mask = df['anomaly_type'] == atype
        if mask.any():
            plt.scatter(df.loc[mask, 'feature_1'], 
                       df.loc[mask, 'target'], 
                       c=color, label=f'{atype.capitalize()} anomaly', 
                       s=100, marker='x', linewidth=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Target')
    plt.title('Target vs Feature 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Residual plot (simple linear regression residuals)
    plt.subplot(1, 3, 3)
    # Fit simple linear regression for visualization
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_anomalous, target_anomalous)
    predictions = lr.predict(X_anomalous)
    residuals = target_anomalous - predictions
    
    plt.scatter(predictions[normal_mask], 
                residuals[normal_mask], 
                c='blue', alpha=0.5, label='Normal', s=30)
    
    for atype, color in colors.items():
        mask = df['anomaly_type'] == atype
        if mask.any():
            idx_mask = np.where(mask)[0]
            plt.scatter(predictions[idx_mask], 
                       residuals[idx_mask], 
                       c=color, label=f'{atype.capitalize()} anomaly', 
                       s=100, marker='x', linewidth=2)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = 'synthetic_regression_anomaly_plots.png'
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlots saved to '{plot_filename}'")
    # plt.show()  # Commented out for script execution

except Exception as e:
    print(f"\nCould not generate plots. Error: {e}")
    print("Please ensure matplotlib and sklearn are installed.")

# --- Generate summary statistics ---
print("\n--- Dataset Summary ---")
print(f"Total samples: {n_samples}")
print(f"Features: {n_features}")
print(f"Anomalies: {n_anomalies} ({100*n_anomalies/n_samples:.1f}%)")
print("\nFeature statistics:")
for feat in feature_names:
    print(f"  {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")
print(f"\nTarget statistics:")
print(f"  mean={df['target'].mean():.2f}, std={df['target'].std():.2f}")