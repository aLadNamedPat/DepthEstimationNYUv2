import torch
import torch.nn.functional as F
import torch.nn as nn
from Residual import ResidualBlock

# Simple UNET Architecture with a RESNET backbone

# Borrowed from the Diffusion code with a changed 
class MultiInputSequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, ResidualBlock):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input

class UNet(nn.Module):
    def __init__(
        self, 
        input_channels : int,
        output_channels : int,
        hidden_dims : list[int]
    ) -> None:
        
        super(UNet, self).__init__()

        self.encoder_store = []

        self.encoder_store.append(
            self.encoder_layer(input_channels, hidden_dims[0])
        )

        #Build a densely connected encoder with many skip connections
        for i in range(len(hidden_dims) - 1):
           self.encoder_store.append(
                self.encoder_layer(hidden_dims[i],
                                    hidden_dims[i + 1])
            )

        self.encoder = MultiInputSequential(
            *self.encoder_store
        )

        self.decoder_store = []

        hidden_dims.reverse() #Reverse the hidden state list

        for i in range(len(hidden_dims) - 1):
            self.decoder_store.append(
                self.decoder_layer(
                    hidden_dims[i] * 2,
                    hidden_dims[i + 1]
                )
            )

        self.decoder = MultiInputSequential(*self.decoder_store)

        self.fl = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1] * 2,
                hidden_dims[-1],
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding= 1
            ),
            nn.BatchNorm2d(
                hidden_dims[-1]
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                output_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.Tanh()
        )
    
    def downsample(
        self,
        input_channels : int,
        output_channels: int,
    ):
        downsample = nn.Sequential(
            nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride = 2,
            ),
            nn.LeakyReLU(),
        )
        return downsample
    
    def upsample(
        self,
        input_channels : int,
        output_channels : int,
    ):
        upsample = MultiInputSequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size = 2,
                stride = 2
            ),
            nn.LeakyReLU()
        )

        return upsample   
    
    def encoder_layer(
        self,
        input_channels : int,
        output_channels : int
        ):
            block = ResidualBlock(input_channels, output_channels)
            block_two = ResidualBlock(output_channels, output_channels)
            block_three = self.downsample(output_channels, output_channels)

            return MultiInputSequential(
                block,
                block_two,
                block_three
            )

    def decoder_layer(
        self,
        input_channels : int,
        output_channels : int
    ):
        block = ResidualBlock(input_channels, output_channels)
        block_two = ResidualBlock(output_channels, output_channels)
        block_three = self.upsample(output_channels, output_channels)

        #Use ConvTranspose2d to upsample back to the original image size
        return MultiInputSequential(
            block,
            block_two,
            block_three
        )

    def encode(
        self, 
        input : torch.Tensor
    ):  
        skip_connections = []
        for layer in self.encoder_store:
            input = layer(input)
            skip_connections.append(input)  #Append all the inputs from the previous layers [128, 128, 256, 512, 512]

        return input, skip_connections
        
    def decode(
        self,
        input : torch.Tensor,
        skip_connections : list,
    ):
        
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.decoder_store):
            input = torch.cat((input, skip_connections[i]), dim = 1)
            input = layer(input)
        a = torch.cat((input, skip_connections[-1]), dim = 1)
        a =  self.fl(a)

        return a

    def forward(
        self,
        input : torch.Tensor
    ) -> torch.Tensor:
        
        z, skip_connections = self.encode(input)
        a = self.decode(z, skip_connections, a)

        return a
    
    def gradient_x(
        self,
        img
    ):
        # So we need to find the difference between the changes in x and take the absolute value of that
        # Image is seen as batch_size, image_channels, x, y

        gx = img[:, :, 1:, :] - img[:, :, :-1, :]
        return gx

    def gradient_y(
        self,
        img
    ):
        gy = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gy
    # Need to calculate the difference between the edge changes in the x and y direction
    def gradient_edge_loss(
        self,
        predicted_depth,
        actual_depth,
    ):
        gx_loss = torch.abs(self.gradient_x(predicted_depth) - self.gradient_x(actual_depth))
        gy_loss = torch.abs(self.gradient_y(predicted_depth) - self.gradient_y(actual_depth))

        total_nums = torch.flatten(gx_loss + gy_loss).shape[0]
        total_loss = torch.sum(torch.flatten(gx_loss + gy_loss)) / total_nums
        return total_loss

    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    def SSIM_loss(
        self,
        predicted_depth,
        actual_depth,
        K1 = 0.001,
        K2 = 0.003,
        scale = 1,
    ):
        C1 = K1 ** 2
        C2 = K2 ** 2

        total_lens = torch.flatten(predicted_depth).shape[0]
        predicted_mean = torch.sum(torch.flatten(predicted_depth)) / total_lens
        actual_mean = torch.sum(torch.flatten(actual_depth)) / total_lens

        predicted_mean = torch.sum(torch.flatten(predicted_depth)) / total_lens
        actual_mean = torch.sum(torch.flatten(actual_depth)) / total_lens

        var_predicted = (torch.sum(torch.flatten((predicted_depth - predicted_mean) ** 2)) / (total_lens - 1))
        var_actual = (torch.sum(torch.flatten(actual_depth - actual_mean ** 2)) / (total_lens - 1))
        covariance = torch.mul((actual_depth - actual_mean), (predicted_depth - predicted_mean)) / (total_lens - 1)
        
        # No contrastive loss used here

        luminance_loss = (2 * predicted_mean * actual_mean + C1) / (predicted_mean ** 2 + actual_mean ** 2 + C1)
        structural_loss = (2 * covariance + C2) / (var_predicted + var_actual + C2)
        # Since loss is between [-1, 1], and we want worse results to be more positive, then subtract from 1 and multiply by scale factor
        loss = (1 - luminance_loss * structural_loss) * scale

        return loss
    
    #Compute the loss of the diffusion model
    def find_loss(
        self,   
        predicted_depth,
        actual_depth,
        mse_coeff = 0.6,
        edge_coeff = 0,
        ssim_coeff = 0,
    ) -> int:
        MSE_LOSS = F.mse_loss(predicted_depth, actual_depth) # This is the MSE loss
        GRE_LOSS = self.gradient_edge_loss(predicted_depth, actual_depth)
        SSIM_LOSS = self.SSIM_loss(predicted_depth, actual_depth)

        loss = MSE_LOSS * mse_coeff + GRE_LOSS * edge_coeff + ssim_coeff * SSIM_LOSS

        return loss