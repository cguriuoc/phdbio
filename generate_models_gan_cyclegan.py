import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import os

torch.cuda.empty_cache()  #  clean the memory before anything else
print("Memory Clean!!")

class KidneyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root_dir, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create datasets
train_dataset = KidneyDataset('/home/carles/local_data/Test/Test/', transform=transform)
val_dataset = KidneyDataset('/home/carles/local_data//Validation/Validation/', transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Define the Generator and Discriminator networks

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Initial layer
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),  # Add Dropout
            # Additional layers
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),  # Add Dropout
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),  # Add Dropout
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Output layer
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),  # Guarantees output size [batch_size, 1, 1, 1]
            nn.Flatten(),  # Converts to [batch_size, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize the models, loss function, and optimizers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 200  # Original 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))  #canging value from 0.0002
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

sample_dir = '/uoc/PHD/research/KMC/images/generated_images_nogrid'
# sample_dir = '/home/carles/local_data/generated_images_G4'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Taining loop

num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        
        n_critic = 5
        for _ in range(n_critic):
            # Train Discriminator Multiple times per generator update
            optimizer_D.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # adding *0.9 (label smoothing)
        output = discriminator(real_images)
        # print("Discriminator output shape:", output.shape)
        d_loss_real = criterion(output, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(z)
        fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1  # adding +0.1
        output = discriminator(fake_images.detach())
        d_loss_fake = criterion(output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 100 == 0:
           print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
    with torch.no_grad():
        fake_images = generator(torch.randn(64, latent_dim, 1, 1).to(device))
        # save_image(fake_images, f"{sample_dir}/epoch_{epoch+1}_step_{i+1}_grid.png", normalize=True, nrow=8)
        for j, img in enumerate(fake_images):
             save_image(img, f"{sample_dir}/epoch_{epoch+1}_step_{i+1}_img_{j+1}.png", normalize=True)

# Save images at the end of each epoch
with torch.no_grad():
    fake_images = generator(torch.randn(64, latent_dim, 1, 1).to(device))
    # save_image(fake_images, f"{sample_dir}/epoch_{epoch+1}_final_grid.png", normalize=True, nrow=8)
    for j, img in enumerate(fake_images):
         save_image(img, f"{sample_dir}/epoch_{epoch+1}_final_img_{j+1}.png", normalize=True)


    # Save generated images or model checkpoints if needed





