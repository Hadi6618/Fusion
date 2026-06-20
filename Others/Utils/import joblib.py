import joblib

# Load the GMM model
gmm = joblib.load("Others/Results/gmm_components_1.joblib")

print(f"Model Type: {type(gmm)}")
print(f"Converged: {gmm.converged_} (Took {gmm.n_iter_} iterations)")
print("-" * 40)

# Extract core GMM parameters
print(f"Number of Mixture Components (Gaussians): {gmm.n_components}")
print(f"Covariance Type: {gmm.covariance_type}")
print("-" * 40)
print(f"Weights Shape:     {gmm.weights_.shape}")
print(f"Means Shape:       {gmm.means_.shape}")
print(f"Covariances Shape: {gmm.covariances_.shape}")

print("-" * 70)

# Load the GMM model
gmm = joblib.load("Others/Results/gmm_components_5.joblib")

print(f"Model Type: {type(gmm)}")
print(f"Converged: {gmm.converged_} (Took {gmm.n_iter_} iterations)")
print("-" * 40)

# Extract core GMM parameters
print(f"Number of Mixture Components (Gaussians): {gmm.n_components}")
print(f"Covariance Type: {gmm.covariance_type}")
print("-" * 40)
print(f"Weights Shape:     {gmm.weights_.shape}")
print(f"Means Shape:       {gmm.means_.shape}")
print(f"Covariances Shape: {gmm.covariances_.shape}")