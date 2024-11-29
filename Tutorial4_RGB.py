import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time

torch.manual_seed(73)

# no resize just conversion to tensor
transform = transforms.Compose([
    transforms.ToTensor() 
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# Check one sample
sample, _ = train_data[0]
print(f"Sample shape: {sample.shape}")  # Should print torch.Size([3, 32, 32])

# Define ConvNet model with 3D Convolution using im2col
class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        self.kernel_size = 7
        self.stride = 3
        self.conv1_channels_out = 4
        
        # Calculate output size after convolution
        # CIFAR-10 images are 32x32, with kernel=7, stride=3, no padding
        self.H_out = (32 - self.kernel_size) // self.stride + 1  # Calculate height after conv
        self.W_out = (32 - self.kernel_size) // self.stride + 1  # Calculate width after conv
        self.fc1_input = self.conv1_channels_out * self.H_out * self.W_out  
        self.fc1 = torch.nn.Linear(self.fc1_input, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

        # Define conv1 layer as a normal Conv2d (not used in actual conv but for weight initialization)
        self.conv1 = torch.nn.Conv2d(3, self.conv1_channels_out, kernel_size=self.kernel_size, stride=self.stride)

    def im2col_3d(self, input_tensor, kernel_size, stride):
        C, H, W = input_tensor.shape

        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        cols = []
        for i in range(0, H - kernel_size + 1, stride):
            for j in range(0, W - kernel_size + 1, stride):
                # Extract patches across all channels and flatten
                patch = input_tensor[:, i:i+kernel_size, j:j+kernel_size].reshape(-1)
                cols.append(patch)
        return np.array(cols).T, H_out, W_out

    def conv3d_im2col(self, input_tensor, kernel, stride):
        C_in, H, W = input_tensor.shape
        C_out, _, kernel_size, _ = kernel.shape
        im2col_matrix, H_out, W_out = self.im2col_3d(input_tensor, kernel_size, stride)

        # kernel is already a numpy array, so no need for .detach().numpy()
        kernel_reshaped = kernel.reshape(C_out, -1)  # Flatten the kernel for multiplication
        output = np.dot(kernel_reshaped, im2col_matrix)
        
        return output.reshape(C_out, H_out, W_out)

    def forward(self, x):
        # Assuming x is a batch of images with shape (batch_size, C, H, W)
        batch_size, C, H, W = x.shape
        x_np = x.numpy()
        
        # Convolution via im2col (3D convolution)
        output = np.zeros((batch_size, self.conv1_channels_out, self.H_out, self.W_out))
        for i in range(batch_size):
            output[i] = self.conv3d_im2col(x_np[i], self.conv1.weight.detach().numpy(), self.stride)
        
        # Convert output back to a PyTorch tensor
        output_tensor = torch.tensor(output, dtype=torch.float32)  # Ensure dtype is float32
        
        # Apply squared activation
        output_tensor = output_tensor * output_tensor
        
        # Flatten and pass through fully connected layers
        output_tensor = output_tensor.view(output_tensor.size(0), -1)
        x = self.fc1(output_tensor)
        x = x * x  # Square activation
        x = self.fc2(x)
        return x

# Loading the saved model
def load_model(model, model_save_path="mnist_convnet.pth"):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

model = ConvNet()

def train(model, train_loader, criterion, optimizer, n_epochs=10, model_save_path="convnet.pth"):
    model.train()
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Calculate average loss for the epoch
        train_loss = train_loss / len(train_loader)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')
    
    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    model.eval()
    return model

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
#model = train(model, train_loader, criterion, optimizer, n_epochs=10, model_save_path="convnet.pth")
model = load_model(model, model_save_path="convnet.pth")
def test(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = [0. for i in range(10)]  # Correct predictions per class
    class_total = [0. for i in range(10)]  # Total samples per class

    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        _, pred = torch.max(output, 1)

        correct = pred.eq(target)  # Returns a tensor of booleans

        for i in range(len(target)):
            label = target[i].item()  # Get the label for this instance
            class_correct[label] += correct[i].item()  # Add to the correct count for the class
            class_total[label] += 1  # Add to the total count for the class

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(class_correct[label])}/{int(class_total[label])})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * sum(class_correct) / sum(class_total))}% ' 
        f'({int(sum(class_correct))}/{int(sum(class_total))})'
    )

test(model, test_loader, criterion)
