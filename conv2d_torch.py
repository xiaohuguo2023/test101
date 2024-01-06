import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input tensor with dimensions: batch size 1, 3 input channels, height 128, width 128
input_tensor = torch.randn(1, 3, 128, 128).to(device)
target_tensor = torch.randn(1, 6, 128, 128).to(device)

class cnnModel(nn.Module):
    def __init__(self):
        super(cnnModel, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv_layer(x)

# Instantiate the model and move it to GPU
model = cnnModel().to(device)

output_tensor = model(input_tensor)

# Calculate loss (using a simple mean squared error loss)
lossFn = nn.MSELoss()
loss = lossFn(output_tensor, target_tensor)

model.zero_grad()
loss.backward()

# Perform the convolution operation and proifle
#with profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
#    with profiler.record_function("forward_pass"):
#output_tensor = model(input_tensor)
# Print profiler results
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# Display the shapes of input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
print("Target tensor shape:", target_tensor.shape)
