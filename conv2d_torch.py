import torch
import torch.nn as nn

# input tensor with dimensions: batch size 1, 3 input channels, height 5, width 5
input_tensor = torch.randn(1, 3, 5, 5)

conv_layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

# Perform the convolution operation
output_tensor = conv_layer(input_tensor)

# Display the shapes of input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
