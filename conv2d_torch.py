import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

# input tensor with dimensions: batch size 1, 3 input channels, height 5, width 5
input_tensor = torch.randn(1, 3, 224, 224)

conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# Perform the convolution operation and proifle
with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        output_tensor = conv_layer(input_tensor)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# Display the shapes of input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
