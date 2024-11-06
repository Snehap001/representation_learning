import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

import pickle

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


# Define a custom dataset to filter MNIST digits (1, 4, 8)
class SubsetMNIST(Dataset):
    def __init__(self, dataset, keep_labels=[1, 4, 8]):
        self.keep_labels = keep_labels
        self.data = []
        self.targets = []
        
        for i in range(len(dataset)):
            if dataset.targets[i] in keep_labels:
                self.data.append(dataset.data[i])
                self.targets.append(dataset.targets[i])

        self.data = torch.stack(self.data).float() / 255.0  # Normalize pixel values to [0, 1]
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch norm after first layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch norm after second layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # Batch norm after third layer
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder network
        self.fc4 = nn.Linear(latent_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)  # Batch norm after fourth layer
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)  # Batch norm after fifth layer
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)  # Batch norm after sixth layer
        self.fc7 = nn.Linear(hidden_dim, input_dim)


    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2)))
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h4 = F.relu(self.bn4(self.fc4(z)))
        h5 = F.relu(self.bn5(self.fc5(h4)))
        h6 = F.relu(self.bn6(self.fc6(h5)))
        return torch.sigmoid(self.fc7(h6))
    
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
def load_mnist_data(path, keep_labels=[1, 4, 8], train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=path, train=train, transform=transform, download=True)
    dataset = SubsetMNIST(dataset, keep_labels=keep_labels)
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
def extract_latent_vectors(data_loader, model):
    
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            
            data = data.to(device)
            data = data.view(-1, 784)
         
            mu, _ = model.encode(data)
            latent_vectors.append(mu)
            labels.append(target)
    return torch.cat(latent_vectors), torch.cat(labels)
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
    num_epochs = 10
    learning_rate = 1e-3
    batch_size = 64

    # Create the model and optimizer
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loading dataset
    if len(sys.argv) == 4:  # VAE reconstruction
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3
        test_dataset_recon = load_mnist_data(path_to_test_dataset_recon, train=False)
        test_loader_recon = DataLoader(test_dataset_recon, batch_size=batch_size, shuffle=False)

    elif len(sys.argv) == 5:  # Class prediction
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4
        test_dataset = load_mnist_data(path_to_test_dataset, train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:  # Training phase
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5

        train_dataset = load_mnist_data(path_to_train_dataset, train=True)
        val_dataset = load_mnist_data(path_to_val_dataset, train=True)
        # Split the training data into training and validation datasets (e.g., 80% train, 20% val)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"arg1:{arg1}, arg2:{arg2}, arg3:{arg3}, arg4:{arg4}, arg5:{arg5}")
    print(f"Device: {device}")
    
    # Training loop (example)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model.forward(data)
     
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}")
    torch.save(model.state_dict(), vaePath)

    train_latent_vectors, train_labels = extract_latent_vectors(train_loader, model)
    val_latent_vectors, val_labels = extract_latent_vectors(val_loader, model)  
    initial_means = calculate_initial_means(val_loader, model)

    # Initialize and train GMM
    gmm = GaussianMixtureModel(n_clusters=3, initial_means=initial_means)
    gmm.train(train_latent_vectors)
    save_gmm_parameters(gmm, filename=gmmPath)
