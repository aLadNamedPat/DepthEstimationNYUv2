import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomImageData(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.gray_dir = dataset_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        gray_img_name = os.path.join(self.gray_dir, self.image_files[idx])

        gray_image = Image.open(gray_img_name).convert('L')  # Convert to grayscale

        if self.transform:
            gray_image = self.transform(gray_image)

        return gray_image, color_image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
dataset_dir = os.path.join(project_dir, "nyu_data/data/nyu2_train")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomImageData(gray_dir=dataset_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

input_channels = 1

out_channels = 3

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
        for gray_images, color_images in test_dataloader:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            reconstructed, _ = model(gray_images)
            loss = model.find_loss(reconstructed, color_images)
            test_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_dataloader)}')

# Save the model
model_path = os.path.join(project_dir, 'ae_model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')