import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import  transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
import numpy as np
from collections import Counter
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim

def plot_2d_manifold(vae, n=20, digit_size=28, device='cuda'):
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate a grid of values between 0.05 and 0.95 percentiles of a normal distribution
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    vae.eval()  # Set VAE to evaluation mode
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                digit=vae.decode(z_sample)
                # Pass z to VAE Decoder 
                # Write your code here
                digit=digit.reshape(28, 28)
              
                new_digit=digit
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] =new_digit


    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.axis('off')
    plt.savefig("generated_images.png")



class FilteredNpzDataset(Dataset):
    def __init__(self, file_path, target_labels, transform):
        # Load data from .npz file
        data = np.load(file_path)
        images = data['data']
        labels = data['labels']
        
        # Filter examples based on target labels
        mask = np.isin(labels, target_labels)
        self.images = images[mask]
        self.labels = labels[mask]
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image and label at the specified index
        image = self.images[idx]
        label = self.labels[idx]
        image = self.transform(image)
        # Convert label to a torch tensor
        label = torch.tensor(label, dtype=torch.long)
        return image, label


#0.8
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder network with Batch Normalization and Dropout
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        # Latent variables (mu and logvar)
        self.fc_mu = nn.Linear(32, 4)    # Latent mu (4-dimensional for better expressiveness)
        self.fc_logvar = nn.Linear(32, 4) # Latent logvar (4-dimensional)
        
        # Decoder network with Batch Normalization
        self.fc8 = nn.Linear(4, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.fc9 = nn.Linear(32, 128)
        self.bn9 = nn.BatchNorm1d(128)
        self.fc10 = nn.Linear(128, 512)
        self.bn10 = nn.BatchNorm1d(512)
        self.fc11 = nn.Linear(512, 512)
        self.bn11 = nn.BatchNorm1d(512)
        self.fc12 = nn.Linear(512, 784)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2)))
        h4 = F.relu(self.bn4(self.fc4(h3)))
        mu = self.fc_mu(h4)
        logvar = self.fc_logvar(h4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h8 = F.relu(self.bn8(self.fc8(z)))    # Latent: 4 -> Hidden: 32
        h9 = F.relu(self.bn9(self.fc9(h8)))   # Hidden: 32 -> 128
        h10 = F.relu(self.bn10(self.fc10(h9))) # Hidden: 128 -> 512
        h11 = F.relu(self.bn11(self.fc11(h10))) # Hidden: 512 -> 512
        h12 = self.fc12(h11) # Hidden: 512 -> Output: 784
        return torch.sigmoid(h12)   # Sigmoid to ensure output is in the range [0, 1]

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input images to vectors
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar



def loss_function(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"BCE: {BCE}")
    # print(f"KLD: {KLD}")
    return BCE + KLD

# Loading MNIST data and creating DataLoader
def load_mnist_data(path, keep_labels=[1, 4, 8]):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FilteredNpzDataset(path,keep_labels,transform)
    return dataset

class GaussianMixtureModel:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.weights = torch.ones(n_clusters) / n_clusters  # Equal initial weights
        self.cluster_labels={}
    def gaussian_density(self, x, mean, covariance):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean=mean.to(device)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x=x.to(device)
        covariance=covariance.to(device)
        dim = mean.size(0)
        cov_inv = torch.inverse(covariance)
        diff = x - mean
        exponent = -0.5 * torch.dot(diff, torch.mv(cov_inv, diff))
        return torch.exp(exponent) / torch.sqrt(((2 * torch.pi) ** dim) * torch.det(covariance))
    def e_step(self, data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)  # Move data to device
        responsibilities = torch.zeros((data.shape[0], self.n_clusters)).to(device)  # Ensure responsibilities are on the correct device
        for i in range(data.shape[0]):
            for j in range(self.n_clusters):
                responsibilities[i, j] = self.weights[j] * self.gaussian_density(data[i], self.means[j], self.covariances[j])
            responsibilities[i] /= responsibilities[i].sum()  # Normalize responsibilities
        return responsibilities
    def m_step(self, data, responsibilities):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)  # Move data to device
        responsibilities = responsibilities.to(device)  # Move responsibilities to device
        N_k = responsibilities.sum(0)  # Sum of responsibilities for each cluster
        for k in range(self.n_clusters):
            # Update the mean of the k-th cluster
            self.means[k] = (responsibilities[:, k].unsqueeze(1) * data).sum(0) / N_k[k]

            # Compute the difference from the mean
            diff = data - self.means[k]

            # Update the covariance matrix for the k-th cluster
            self.covariances[k] = (responsibilities[:, k].unsqueeze(1) * diff).t() @ diff / N_k[k]

            # Update the weight of the k-th cluster
            self.weights[k] = N_k[k] / data.size(0)
    def train(self, initial_means, data, max_iters=50, tol=1e-4):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.means = initial_means.to(device)  # Move initial means to device
        self.covariances = [torch.eye(initial_means.size(1)).to(device) for _ in range(self.n_clusters)]  # Move covariances to device
        data = data.to(device)  # Move data to device

        for iteration in range(max_iters):
            old_means = self.means.clone()
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)
            
            print(f"Iteration {iteration + 1}/{max_iters}")
            
            # Check for convergence
            if torch.all(torch.abs(self.means - old_means) < tol):
                break
        return responsibilities
    def set_cluster_labels(self, val_latent_vectors, val_labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_latent_vectors = val_latent_vectors.to(device)  # Move validation latent vectors to device

        responsibilities = self.e_step(val_latent_vectors.cpu().numpy())  # Use CPU for numpy ops if needed
        cluster_indices = responsibilities.argmax(dim=-1)

        # For each cluster, find the majority label
        for cluster_idx in range(self.n_clusters):
            cluster_samples = val_labels[cluster_indices == cluster_idx]

            if len(cluster_samples) > 0:
                majority_label = Counter(cluster_samples.numpy()).most_common(1)[0][0]
                self.cluster_labels[cluster_idx] = majority_label

        return self.cluster_labels

    def load_gmm_parameters(self, filename="gmm_params.pkl"):
        """
        Loads GMM parameters from a file and initializes the model attributes.
        
        Args:
            filename: Name of the file to load the parameters from.
        """
        with open(filename, "rb") as f:
            gmm_params = pickle.load(f)
        
        # Set the GMM model parameters
        self.means = gmm_params["means"]
        self.covariances = gmm_params["covariances"]
        self.weights = gmm_params["weights"]
        self.cluster_labels=gmm_params['cluster_labels']
        print(f"GMM parameters loaded from {filename}")
    def predict(self, latent_vectors):
        """
        Predicts the most likely cluster label for each latent vector.

        Args:
            gmm_model: Trained GMM model for clustering the latent vectors.
            latent_vectors: Tensor or numpy array of latent vectors from the VAE model.
            cluster_labels: Dictionary mapping cluster index to cluster label (majority label).
        
        Returns:
            predicted_labels: List of predicted cluster labels for each latent vector.
        """
        responsibilities = self.e_step(latent_vectors)
        cluster_indices=responsibilities.argmax(dim=-1)
        # Map the cluster indices to the actual labels using the cluster_labels dictionary
        predicted_labels = [self.cluster_labels[int(index)] for index in cluster_indices]
        
        # Return the predicted labels
        return predicted_labels
    def save_gmm_parameters(self, filename="gmm.params.pkl"):
        """
        Saves the parameters of the GMM model to a file.
        
        Args:
            gmm: Trained Gaussian Mixture Model object.
            filename: Name of the file to save the parameters to.
        """
        # Extract the GMM parameters (assuming your GMM class has attributes like means, covariances, and weights)
        gmm_params = {
            "means": self.means,  # Replace with your GMM's means attribute
            "covariances": self.covariances,  # Replace with your GMM's covariances attribute
            "weights": self.weights,  # Replace with your GMM's weights attribute
            "cluster_labels": self.cluster_labels
        }
        
        # Save the parameters to a pickle file
        with open(filename, "wb") as f:
            pickle.dump(gmm_params, f)
        print(f"GMM parameters saved to {filename}")
    def plot_gmm_ellipses(self):
        """
        Plots each Gaussian distribution in the GMM as an ellipse.
        
        Args:
            means (list or np.ndarray): List or array of cluster centers (means).
            covariances (list or np.ndarray): List or array of covariance matrices.
        """
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Plot each Gaussian component as an ellipse
        for i, (mean, cov) in enumerate(zip(self.means, self.covariances)):
            # Calculate eigenvalues and eigenvectors for the covariance matrix
            cov = cov.cpu().numpy()
            mean = mean.cpu().numpy()
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Scale the eigenvalues to represent the ellipse's axes lengths
            axis_length = 2 * np.sqrt(eigenvalues)
            
            # Determine the angle for the ellipse rotation (from the first eigenvector)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            
            # Create the ellipse based on the mean, axis length, and angle
            ellipse = Ellipse(
                xy=mean,
                width=axis_length[0],
                height=axis_length[1],
                angle=angle,
                edgecolor='black',
                facecolor='none',
                linewidth=1.5
            )
            
            # Plot the ellipse and mark the center
            ax.add_patch(ellipse)
            plt.plot(mean[0], mean[1], 'o', markersize=8, label=f'Cluster {i} Mean')

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("GMM Clusters with Gaussian Ellipses")
        plt.legend()
        plt.savefig('gmm_vis.png')

def calculate_initial_means(val_loader, model, keep_labels=[1, 4, 8]):
    """
    Calculate initial means for each digit class to initialize GMM clusters.
    
    Args:
        val_loader: DataLoader for the validation dataset.
        model: Trained VAE model to extract latent vectors.
        keep_labels: List of digit classes to calculate initial means for.
    
    Returns:
        initial_means: Tensor containing the initial mean for each digit class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    latent_vectors = {label: [] for label in keep_labels}
    
    # Collect latent vectors for each digit class
    with torch.no_grad():
        for data, target in val_loader:
            data = data.view(-1, 784)  # Flatten MNIST images to 784-dimensional vectors
            data=data.to(device)
            mu, _ = model.encode(data)  # Get the mean (mu) from the VAE encoder
            
            for label in keep_labels:
                mask = (target == label)  # Find images of the current label
                latent_vectors[label].append(mu[mask])  # Append the corresponding latent vectors

    # Calculate the mean for each class
    initial_means = []
    for label in keep_labels:
        if len(latent_vectors[label]) > 0:
            class_latent_vectors = torch.cat(latent_vectors[label], dim=0)
            class_mean = class_latent_vectors.mean(dim=0)
            initial_means.append(class_mean)
    
    return torch.stack(initial_means)

def show_reconstruction(model, val_loader, n=15):
    model.eval()
    data, labels = next(iter(val_loader))
    
    data = data.to(device)
    recon_data, _, _ = model(data)
    
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        # Original images
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(recon_data[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
    plt.savefig("val_images.png")
    # plt.show()

def save_reconstruction(model,val_loader):
    model.eval()
    data, labels = next(iter(val_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    data = data.to(device)
    recon_data, _, _ = model(data)
    recon_data_np = recon_data.cpu().detach().numpy()

# Save the NumPy array to a .npz file
    np.savez("vae_reconstructed.npz", recon_data=recon_data_np)


def extract_latent_vectors(data_loader, model):
    latent_vectors = []
    labels = []
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    with torch.no_grad():
        for data, target in data_loader:
            # Ensure the batch is not empty
            if data.size(0) == 0:
                continue
            
            # Move data to the appropriate device and reshape if needed
            data = data.to(device).view(-1, 784)
            
            # Pass data through the encoder to get the mean (mu) of the latent distribution
            mu, _ = model.encode(data)
            if mu.dim() == 0:
                mu = mu.unsqueeze(0)  # Add a dimension to make it 1-dimensional
            
            # Ensure target has the right shape as well
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            # Ensure mu and target are on the CPU and append to lists
            latent_vectors.append(mu.cpu())
            labels.append(target.cpu())
    
    # Concatenate all latent vectors and labels if they have valid entries

    if latent_vectors and labels:
        return torch.cat(latent_vectors), torch.cat(labels)
    else:
        raise ValueError("No valid latent vectors or labels were extracted.")

def classify_image(image, vae_model, gmm_model):
    vae_model.eval()
    with torch.no_grad():
        mu, _ = vae_model.encode(image.unsqueeze(0))  # Get latent vector for the image
    max_likelihood, label = None, None
    for i in range(gmm_model.n_clusters):
        likelihood = gmm_model.weights[i] * gmm_model.gaussian_density(mu.squeeze(), gmm_model.means[i], gmm_model.covariances[i])
        if max_likelihood is None or likelihood > max_likelihood:
            max_likelihood = likelihood
            label = i
    return label
def train_model(train_loader,num_epochs,model,device,optimizer,val_loader,vaePath):
    # Training loop (example)
    model.to(device)
    best_val_accuracy=0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} started")
        model.train()
        train_loss = 0
        for _, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)  
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        model.eval()
        correct = 0
        total = 0
        validation_loss = 0
        threshold = 0.5  # Set threshold based on SSIM (adjust based on needs)

        with torch.no_grad():
            for _, (data, target) in enumerate(val_loader):
                data = data.to(device)
                
                # Get model outputs
                recon_batch, mu, logvar = model(data)
                
                # Calculate loss
                loss = loss_function(recon_batch, data, mu, logvar)
                validation_loss += loss.item()
                
                # Convert images to numpy for SSIM
                original_images = data.view(-1, 28, 28).cpu().numpy()
                reconstructed_images = recon_batch.view(-1, 28, 28).cpu().numpy()
                
                # Calculate SSIM for each image and compare with threshold
                ssim_scores = []
                for original, reconstructed in zip(original_images, reconstructed_images):
                    ssim_score = ssim(original, reconstructed, data_range=original.max() - original.min())
                    
                    ssim_scores.append(ssim_score)
                    if ssim_score > threshold:
                        correct += 1  # Count as correct if SSIM is above threshold
                    total += 1  # Total number of images
                mse = F.mse_loss(recon_batch, data.view(-1,784), reduction='mean')

                mean_ssim = sum(ssim_scores) / len(ssim_scores)
                print(f"Mean SSIM Score: {mean_ssim:.4f}")
                print("1- Mean Squared Error:", 1-mse.item())

            # Calculate validation accuracy and loss
            validation_accuracy = correct / total
            print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
            if validation_accuracy>best_val_accuracy:  
                best_val_accuracy=validation_accuracy
                print("model saved")
                torch.save(model.state_dict(), vaePath)

    # train_latent_vectors, _ = extract_latent_vectors(train_loader, model)
    
    # initial_means = calculate_initial_means(val_loader, model)

    # # Initialize and train GMM
    # gmm = GaussianMixtureModel(n_clusters=3)
    # gmm.train(initial_means,train_latent_vectors)
    # val_latent_vectors, val_labels = extract_latent_vectors(val_loader,model)
    # gmm.set_cluster_labels(val_latent_vectors,val_labels)
    # gmm.save_gmm_parameters(filename=gmmPath)


def test_model(test_loader, vae_model, gmm_model):
    """
    Predicts the most likely cluster label for each example in the test set.

    Args:
        test_loader: DataLoader for the test dataset.
        vae_model: Trained VAE model with an encoder for generating latent vectors.
        gmm_model: Trained GMM model for clustering the latent vectors.
        cluster_labels: Dictionary mapping cluster index to cluster label (majority label).
    
    Returns:
        predicted_labels: List of predicted cluster labels for each example in the test set.
    """
    
    # Concatenate all latent vectors
    latent_vectors,_ = extract_latent_vectors(test_loader,vae_model)
    
    # Map cluster indices to cluster labels
    predicted_labels = gmm_model.predict(latent_vectors.cpu().numpy())
    
    # Save the predicted labels to a CSV file
    df = pd.DataFrame(predicted_labels, columns=["Predicted_Label"])
    df.to_csv("vae.csv", index=False)


def visualise_latent_space(latent_vectors_2d,labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Class Labels")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("2D Scatter Plot of VAE Latent Space")
    plt.grid(True)
    plt.savefig("latent_space_visualise.png")

if __name__ == "__main__":
    # Get command-line arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 50
    learning_rate = 1e-3
    batch_size = 64

    print("Creating model...")
    # Create the model and optimizer
    model = VAE().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

    # Loading dataset
    if len(sys.argv) == 4:  # VAE reconstruction
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3
        test_dataset_recon = load_mnist_data(path_to_test_dataset_recon)
        test_loader_recon = DataLoader(test_dataset_recon, batch_size=batch_size, shuffle=False,num_workers=4)
        model.load_state_dict(torch.load(vaePath, map_location=device, weights_only=True))  
        save_reconstruction(model, test_loader_recon)
        # show_reconstruction(model,test_loader_recon)

    elif len(sys.argv) == 5:  # Class prediction
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4
        test_dataset = load_mnist_data(path_to_test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
        model.load_state_dict(torch.load(vaePath, map_location=device, weights_only=True))  
        gmm = GaussianMixtureModel(n_clusters=3)
        gmm.load_gmm_parameters(gmmPath)
        gmm.plot_gmm_ellipses()
        test_model(test_loader,model,gmm)



    else:  # Training phase
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5

        train_dataset = load_mnist_data(path_to_train_dataset)
        val_dataset = load_mnist_data(path_to_val_dataset)
        # Split the training data into training and validation datasets (e.g., 80% train, 20% val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
        print("Training...")
        train_model(train_loader,num_epochs,model,device,optimizer,val_loader,vaePath)
        # show_reconstruction(model, val_loader)
        # plot_2d_manifold(model, n=20, digit_size=28, device=device)
        # latent_vectors,labels=extract_latent_vectors(train_dataset,model)
        # visualise_latent_space(latent_vectors,labels)
