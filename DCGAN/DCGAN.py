import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialise_weight

######## setting hyperparameters #######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64


###### Transforms and dataloader #####
transforms = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)]),

    ]
)

dataset = datasets.MNIST(root = "dataset/", train = True, transform=transforms , download = True)
# dataset = datasets.ImageFolder(root = "celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialise_weight(gen)
initialise_weight(disc)

######## optimisers ########
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas = (0.5, 0.999) )
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device )
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()


##### training loop ############
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader): 
        real = real.to(device) 
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Training the discriminator : max log(D(real)) + log(1 - D(G(z)) )
        disc_real = disc(real).reshape(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) #label for real images is 1   
        # for fake
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) #label for fake images is 1


        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True) # fake has to be used again so retaining it
        opt_disc.step()


        # Training the Generator : min log(1 - D(G(z))) <--> leads to slow gradients so we use max log(D(G(z)))
        output = disc(fake).reshape(-1) # fake is reused since retain_graph was True
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # Setting up tensorboard
        if batch_idx == 0:
            print(
                f"Epoch:[{epoch}/{NUM_EPOCHS}] | Batch:[{batch_idx}/{len(loader)}]  Loss D: {lossD:.4f}, Loss G:{lossG:.4f}")
        with torch.no_grad():
            fake = gen(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
                )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
                )
            step += 1

