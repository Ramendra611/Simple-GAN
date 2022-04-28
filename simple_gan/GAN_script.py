# from locale import normalize
# from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
                nn.Linear(img_dim, 128),
                nn.LeakyReLU(0.01),
                nn.Linear(128, 1),
                nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # to make sure output is [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

#################### setting hyper parameters ######################
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64 
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50


disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [ 
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,),(0.3081,)), # 0.1307 and 0.3081 are mean and sd of Mnist dataset
        transforms.Normalize((0.5,),(0.5,)), # 0.1307 and 0.3081 are mean and sd of Mnist dataset
    ])
################## loading the data #########################
dataset = datasets.MNIST(root = "dataset/", transform = transforms, download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)


##################    optimisers ############################### 
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

##################### for tensorboard #########
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

########### training loop #########
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader): 
        real = real.view(-1, 784).to(device) # flattening the  image
        batch_size = real.shape[0] #size of the batch


        # Training the discriminator : max log(D(real)) + log(1 - D(G(z)) )
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        # for real
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) #label for real images is 1
        # for fake
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))


        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True) # fake has to be used again so retaining it
        opt_disc.step()


        # Training the Generator : min log(1 - D(G(z))) <--> leads to slow gradients so we use max log(D(G(z)))
        output = disc(fake).view(-1) # fake is reused since retain_graph was True
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # Setting up tensorboard
        if batch_idx == 0:
            print(
                f"Epoch:[{epoch}/{num_epochs}] Loss D: {lossD:.4f}, Loss G:{lossG:.4f}")
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
                )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
                )
            step += 1