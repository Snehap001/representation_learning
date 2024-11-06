import sys

#Any additional sklearn import will flagged as error in autograder
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
import skimage.metrics as image_metrics  #SSIM

# import torch


if __name__ == "__main__": 
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

if len(sys.argv)==4:### Running code for vae reconstruction.
    path_to_test_dataset_recon = arg1
    test_reconstruction = arg2
    vaePath = arg3
    
elif len(sys.argv)==5:###Running code for class prediction during testing
    path_to_test_dataset = arg1
    test_classifier = arg2
    vaePath = arg3
    gmmPath = arg4

else:### Running code for training. save the model in the same directory with name "vae.pth"
    path_to_train_dataset = arg1
    path_to_val_dataset = arg2
    trainStatus = arg3
    vaePath = arg4
    gmmPath = arg5


print(f"arg1:{arg1}, arg2:{arg2}, arg3:{arg3}, arg4:{arg4}, arg5:{arg5}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Custom dataset filter to keep only digits 1, 4, and 8

# class SubsetMNIST(torch.utils.data.Dataset):
#     def __init__(self, dataset, keep_labels=[1, 4, 8]):
#         ## YOUR CODE HERE
#         pass

#     def __len__(self):
#         ## YOUR CODE HERE
#         pass
    
#     def __getitem__(self, idx):
#         ## YOUR CODE HERE
#         pass
    


# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(VAE, self).__init__()
#         ## YOUR CODE HERE
#         pass
#     def encode(self, x):
#         ## YOUR CODE HERE
#         pass
    
#     def reparameterize(self, mu, logvar):
#         ## YOUR CODE HERE
#         pass
    
#     def decode(self, z):
#         ## YOUR CODE HERE
#         pass
    
#     def forward(self, x):
#         ## YOUR CODE HERE
#         pass


# def loss_function(recon_x, x, mu, logvar):
#         ## YOUR CODE HERE
#         pass