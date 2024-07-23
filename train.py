import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
from smaller_model import UNET
from tqdm import tqdm
import wandb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    # set the wandb project where this run will be logged
    project="Depth Prediction",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "UNET",
    "dataset": "Landscape-Color",
    "batch_size" : 32,
    "latent_dims" : 512,
    "epochs": 100,
    }
)

class CustomImageData(Dataset):
    def __init__(self, dataset_dir, transform_rgb=None, transforms_depth = None):
        self.root_dir = dataset_dir
        self.transform_rgb = transform_rgb
        self.transform_depth = transforms_depth

        self.image_files = []

        for dir_path, dir_names, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".jpg"):
                    rgb_path = os.path.join(dir_path, filename)
                    depth_path = os.path.join(dir_path, filename[:-4] + '.png')
                    if os.path.exists(depth_path):
                        self.image_files.append((rgb_path, depth_path))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        rgb_image, depth_image = self.image_files[idx]
        
        RGB_IMAGE = Image.open(rgb_image).convert("RGB")
        DEPTH_IMAGE = Image.open(depth_image)

        if self.transform_rgb:
            RGB_IMAGE = self.transform_rgb(RGB_IMAGE)
        
        if self.transform_depth:
            DEPTH_IMAGE = self.transform_depth(DEPTH_IMAGE)
        
        return RGB_IMAGE, DEPTH_IMAGE

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
dataset_dir = os.path.join(project_dir, "Depth-Prediction/nyu_data/data/nyu2_train")


transform_rgb  = transforms.Compose([
    transforms.Resize((96, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_depth= transforms.Compose([
    transforms.Resize((96, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomImageData(dataset_dir=dataset_dir, transform_rgb=transform_rgb, transforms_depth=transform_depth)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=12, shuffle=False)

input_channels = 3

out_channels = 1

hidden_dims = [128, 128, 256, 512, 512]
model = UNET(input_channels, out_channels, hidden_dims).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# Training loop
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for color_images, depth_images in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        color_images = color_images.to(device)
        depth_images = depth_images.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(color_images)
        
        # Compute loss
        loss = model.find_loss(reconstructed, depth_images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_avg = train_loss / len(train_dataloader)
    wandb.log({"Train Loss": train_loss_avg})

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}')

    # Validation loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for color_images, depth_images in valid_dataloader:
            color_images = color_images.to(device)
            depth_images = depth_images.to(device)
            reconstructed = model(color_images)
            loss = model.find_loss(reconstructed, depth_images)
            test_loss += loss.item()
    valid_loss_avg = test_loss / len(valid_dataloader)
    wandb.log({"Test Loss": valid_loss_avg})
    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(valid_dataloader)}')

    wandb.log({"Original_Image" : [wandb.Image(color_images[0], caption=f"Original Image")]})
    wandb.log({"Depth Image" : [wandb.Image(depth_images[0], caption=f"Original Image")]})
    wandb.log({"Depth Image Reconstructed": [wandb.Image(reconstructed[0], caption=f"Image Depth")]})


# Save the model
model_path = os.path.join(project_dir, 'Depth_Model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')