import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
dataset_dir = os.path.join(project_dir, "nyu_data/data/nyu2_train")

transform_rgb  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

transform_depth= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomImageData(gray_dir=dataset_dir, transform_rgb=transform_rgb, transforms_depth=transform_depth)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

input_channels = 3

out_channels = 1

hidden_dims = [128, 128, 256, 512, 512]
model = UNet(input_channels, out_channels, hidden_dims).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Training loop
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for gray_images, color_images in train_dataloader:
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(gray_images)
        
        # Compute loss
        loss = model.find_loss(reconstructed, color_images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}')

    # Validation loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for gray_images, color_images in valid_dataloader:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            reconstructed, _ = model(gray_images)
            loss = model.find_loss(reconstructed, color_images)
            test_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(valid_dataloader)}')

# Save the model
model_path = os.path.join(project_dir, 'ae_model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')