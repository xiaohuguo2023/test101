import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input tensor with dimensions: batch size 1, 3 input channels, height 5, width 5
input_tensor = torch.randn(1, 3, 224, 224).to(device)

conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).to(device)

# Perform the convolution operation and proifle
with profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
    with profiler.record_function("forward_pass"):
         output_tensor = conv_layer(input_tensor)
# Print profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# Display the shapes of input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
