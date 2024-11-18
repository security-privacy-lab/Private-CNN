import torch
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts
import numpy as np
import time

torch.manual_seed(73)

# Downsample CIFAR-10 images to 16x16
transform = transforms.Compose([
    transforms.Resize((16, 16)),  # Resize images to 16x16
    transforms.ToTensor()
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

# Define ConvNet model
class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(4 * 4 * 4, hidden)  # Adjust input size
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x  # Square activation
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = x * x  # Square activation
        x = self.fc2(x)
        return x

model = ConvNet()

# Define TenSEAL-compatible encrypted model
class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels,
            torch_nn.conv1.in_channels,
            torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            enc_result = sum(
                enc_x[i].conv2d_im2col(kernel[i], windows_nb)
                for i in range(len(kernel))
            ) + bias
            enc_channels.append(enc_result)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x.square_()  # Square activation
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias  # Fully connected 1
        enc_x.square_()  # Square activation
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias  # Fully connected 2
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# TenSEAL context with increased poly_modulus_degree
bits_scale = 26
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,  # Increased to handle larger inputs
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)
context.global_scale = pow(2, bits_scale)
context.generate_galois_keys()

# Define training function
def train(model, train_loader, criterion, optimizer, n_epochs=10, model_save_path="mnist_convnet.pth"):
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
model = train(model, train_loader, criterion, optimizer, n_epochs=10, model_save_path="convnet.pth")

def test(model, test_loader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = [0. for i in range(10)]  # Correct predictions per class
    class_total = [0. for i in range(10)]  # Total samples per class

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        
        # compare predictions to true label
        correct = pred.eq(target)  # Returns a tensor of booleans

        # calculate test accuracy for each object class
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

# Example call for testing
test(model, test_loader, criterion)


# Encrypted Test function
def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    test_loss = 0.0
    class_correct = [0.0] * 10
    class_total = [0.0] * 10

    for batch_idx, (data, target) in enumerate(test_loader):
        start_time = time.time_ns()

        # Encode and encrypt
        channels_enc = []
        windows_nb = None
        for channel in data[0]:  # Process each channel independently
            enc_channel, windows_nb = ts.im2col_encoding(
                context, channel.numpy(), kernel_shape[0], kernel_shape[1], stride
            )
            channels_enc.append(enc_channel)

        elapsed_time_ns = time.time_ns() - start_time
        print(f"Time for encoding and encryption in batch {batch_idx + 1}: {elapsed_time_ns} ns")

        # Encrypted evaluation
        enc_output = model(channels_enc, windows_nb)

        # Decrypt result
        output = torch.tensor(enc_output.decrypt()).view(1, -1)

        # Compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # Predictions
        _, pred = torch.max(output, 1)
        correct = pred.eq(target).item()
        class_correct[target.item()] += correct
        class_total[target.item()] += 1

        # Only print progress for every 100th batch
        if batch_idx % 100 == 0:
            print(f'Test Batch {batch_idx + 1}/{len(test_loader)}')

    # Test summary
    test_loss /= sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(f'Test Accuracy of {label}: {100 * class_correct[label] / class_total[label]:.2f}% '
              f'({class_correct[label]:.0f}/{class_total[label]:.0f})')

    print(f'\nOverall Test Accuracy: {100 * sum(class_correct) / sum(class_total):.2f}%')

# Run the encrypted test
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

enc_model = EncConvNet(model)
enc_test(context, enc_model, test_loader, torch.nn.CrossEntropyLoss(), kernel_shape, stride)
