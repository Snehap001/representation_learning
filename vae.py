import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim

def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device='cuda'):
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


def save_gmm_parameters(gmm, filename="gmm.params.pkl"):
    """
    Saves the parameters of the GMM model to a file.
    
    Args:
        gmm: Trained Gaussian Mixture Model object.
        filename: Name of the file to save the parameters to.
    """
    # Extract the GMM parameters (assuming your GMM class has attributes like means, covariances, and weights)
    gmm_params = {
        "means": gmm.means,  # Replace with your GMM's means attribute
        "covariances": gmm.covariances,  # Replace with your GMM's covariances attribute
        "weights": gmm.weights  # Replace with your GMM's weights attribute
    }
    
    # Save the parameters to a pickle file
    with open(filename, "wb") as f:
        pickle.dump(gmm_params, f)
    print(f"GMM parameters saved to {filename}")


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
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # First layer, input 784, output 256
        self.fc2 = nn.Linear(256, 128)  # Second layer, input 256, output 128
        self.fc3 = nn.Linear(128, 64)   # Third layer, input 128, output 64
        self.fc4 = nn.Linear(64, 16)     # Output layer, input 64, output 2
        
        self.fc_mu = nn.Linear(16, 2)
        self.fc_logvar = nn.Linear(16, 2)

        # Decoder network
        self.fc5 = nn.Linear(2, 16)   # First layer now takes input of size 2
        self.fc6 = nn.Linear(16, 64)
        self.fc7 = nn.Linear(64, 128)
        self.fc8 = nn.Linear(128, 256)
        self.fc9 = nn.Linear(256, 784)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        mu = self.fc_mu(h4)
        logvar = self.fc_logvar(h4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h5 = F.relu(self.fc5(z))  # Now expecting latent space size of 2
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        h9 = self.fc9(h8)
        return torch.sigmoid(h9)  # Sigmoid activation to reconstruct input between 0 and 1
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input images to vectors
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar



# VAE Loss Function
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
    def __init__(self, n_clusters, initial_means):
        self.n_clusters = n_clusters
        self.means = initial_means  # Initialize with validation means
        self.covariances = [torch.eye(initial_means.size(1)) for _ in range(n_clusters)]
        self.weights = torch.ones(n_clusters) / n_clusters  # Equal initial weights

    def gaussian_density(self, x, mean, covariance):
        dim = mean.size(0)
        cov_inv = torch.inverse(covariance)
        diff = x - mean
        exponent = -0.5 * torch.dot(diff, torch.mv(cov_inv, diff))
        return torch.exp(exponent) / torch.sqrt(((2 * torch.pi) ** dim) * torch.det(covariance))

    def e_step(self, data):
        responsibilities = torch.zeros((data.size(0), self.n_clusters))
        for i in range(data.size(0)):
            for j in range(self.n_clusters):
                responsibilities[i, j] = self.weights[j] * self.gaussian_density(data[i], self.means[j], self.covariances[j])
            responsibilities[i] /= responsibilities[i].sum()  # Normalize responsibilities
        return responsibilities

    def m_step(self, data, responsibilities):
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

    def train(self, data, max_iters=100, tol=1e-4):
        for iteration in range(max_iters):
            old_means = self.means.clone()
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)
            print(f"iter :{iteration}")
            if torch.all(torch.abs(self.means - old_means) < tol):
                break
        return responsibilities
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
    model.eval()
    latent_vectors = {label: [] for label in keep_labels}
    
    # Collect latent vectors for each digit class
    with torch.no_grad():
        for data, target in val_loader:
            data = data.view(-1, 784)  # Flatten MNIST images to 784-dimensional vectors
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
    
    data = data.to(device)
    recon_data, _, _ = model(data)
    recon_data_np = recon_data.cpu().detach().numpy()

# Save the NumPy array to a .npz file
    np.savez("vae_reconstructed.npz", recon_data=recon_data_np)


def extract_latent_vectors(data_loader, model):
    latent_vectors = []
    labels = []
    model.eval()  # Set model to evaluation mode
    
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
            

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")
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
                mean_ssim = sum(ssim_scores) / len(ssim_scores)
                print(f"Mean SSIM Score: {mean_ssim:.4f}")

            # Calculate validation accuracy and loss
            validation_accuracy = correct / total
            average_validation_loss = validation_loss / len(val_loader)
            print(f"Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy * 100:.2f}%")
            if validation_accuracy>best_val_accuracy:  
                best_val_accuracy=validation_accuracy
                print("model saved")
                torch.save(model.state_dict(), vaePath)

    # train_latent_vectors, _ = extract_latent_vectors(train_loader, model)
    
    # initial_means = calculate_initial_means(train_loader, model)

    # Initialize and train GMM
    # gmm = GaussianMixtureModel(n_clusters=3, initial_means=initial_means)
    # gmm.train(train_latent_vectors)
    # save_gmm_parameters(gmm, filename=gmmPath)

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
    input_dim = 784  # 28x28 images flattened to 784 dimensions
    hidden_dim = 512
    latent_dim = 2
    num_epochs = 25
    learning_rate = 1e-3
    batch_size = 128

    print("Creating model...")
    # Create the model and optimizer
    model = VAE().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loading dataset
    if len(sys.argv) == 4:  # VAE reconstruction
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3
        test_dataset_recon = load_mnist_data(path_to_test_dataset_recon)
        test_loader_recon = DataLoader(test_dataset_recon, batch_size=batch_size, shuffle=False)
        model.load_state_dict(torch.load("vae.pth", map_location=device, weights_only=True))  
        save_reconstruction(model, test_loader_recon)
        # show_reconstruction(model,test_loader_recon)

    elif len(sys.argv) == 5:  # Class prediction
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4
        test_dataset = load_mnist_data(path_to_test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:  # Training phase
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5

        train_dataset = load_mnist_data(path_to_train_dataset)
        val_dataset = load_mnist_data(path_to_val_dataset)
        # Split the training data into training and validation datasets (e.g., 80% train, 20% val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print("Training...")
        train_model(train_loader,num_epochs,model,device,optimizer,val_loader,vaePath)
        show_reconstruction(model, val_loader)
        plot_2d_manifold(model, latent_dim=2, n=20, digit_size=28, device=device)
        latent_vectors,labels=extract_latent_vectors(train_dataset,model)
        visualise_latent_space(latent_vectors,labels)

        
