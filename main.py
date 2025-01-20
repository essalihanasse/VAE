import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
transform = transforms.Compose(
    [transforms.ToTensor()])
class VAE(nn.Module):
    def __init__(self,input_dim=784,hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu=nn.Linear(hidden_dim, latent_dim)
        self.fc_var=nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() 
        )
    def encode(self,x):
        h=self.encoder(x)
        mu, log_var=self.fc_mu(h), self.fc_var(h)
        return mu, log_var
       
    
    def reparametrize(self,mu, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std)
        z=mu+eps*std
        return z
    def decode(self,z):
        x=self.decoder(z)
        return x
    def forward(self,x):
        mu, log_var=self.encode(x)
        z=self.reparametrize(mu, log_var)
        x_g=self.decode(z)
        return x_g, mu, log_var
def loss_function(x, recon_x, mu, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # Changed from size_average to reduction='sum'
    return BCE + KLD


def train_vae(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        
        loss = loss_function(data, recon_batch, mu, log_var) 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def main():
    # Hyperparameters
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose(
    [transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    for epoch in range(1, epochs + 1):
        loss = train_vae(model, train_loader, optimizer, device, epoch)
        loss_history.append(loss)
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    return model

# Generate samples from trained model
def generate_samples(model, num_samples=10):
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, 20).to(next(model.parameters()).device)
        # Decode the latent vectors
        samples = model.decode(z)
        # Reshape samples for visualization
        samples = samples.view(-1, 28, 28).cpu()
    return samples

if __name__ == "__main__":
    model = main()
    # Generate and visualize samples
    samples = generate_samples(model)
    
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    plt.show()