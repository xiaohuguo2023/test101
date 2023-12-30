import torch
import torch.nn as nn
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time=time.time()
        value=func(*args, *kwargs)
        end_time=time.time()
        run_time=end_time-start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

# Define a simple MLP (Multi-Layer Perceptron) model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 50) # Input layer (10 neurons) to hidden layer (50 neurons)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)  # Hidden layer to output layer (1 neuron)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@timer
def forward_pass(model, input_tensor):
# Create an instance of the model
    return model(input_tensor)

# Generate a random input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10
model=SimpleMLP()

output= forward_pass(model, input_tensor)

# Measure the time taken for the forward pass
start_time = time.time()
output = model(input_tensor)
end_time = time.time()

# Calculate the time taken
run_time = end_time - start_time

run_time, output.detach()

