import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 1) Prepare CIFAR-10 Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

# 2) Custom MyConv2dIm2Col Layer
class MyConv2dIm2Col(nn.Module):
    """
    Manually performs 2D convolution using F.unfold (im2col) + matrix multiply.
    We'll ignore dilation and groups for simplicity.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(MyConv2dIm2Col, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (kh, kw)
        self.stride = stride            # (sh, sw)
        self.padding = padding          # (ph, pw)

        # Learnable weights: shape [out_channels, in_channels, kh, kw]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * 0.01
        )
        # Bias: shape [out_channels]
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        x: shape [N, in_channels, H, W]
        Steps:
          1) F.unfold => [N, C*kh*kw, L]
          2) Reshape & matmul => [N, out_channels, L]
          3) Reshape => [N, out_channels, out_h, out_w]
          4) Add bias
        """
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # 1) Unfold => shape [N, C*kh*kw, L]
        patches = F.unfold(
            x,
            kernel_size=(kh, kw),
            dilation=1,
            padding=(ph, pw),
            stride=(sh, sw),
        )  # => [N, C*kh*kw, L]

        # 2) Flatten weight => [out_channels, C*kh*kw]
        w = self.weight.view(self.out_channels, -1)

        # Compute output height/width
        out_h = (H + 2*ph - kh)//sh + 1
        out_w = (W + 2*pw - kw)//sw + 1
        L = out_h * out_w  # same as patches.shape[2]

        # 3) We'll do a standard 2D matmul:  
        # patches => [N, C*kh*kw, L] => transpose => [N, L, C*kh*kw]
        patches_t = patches.transpose(1, 2)     # => [N, L, C*kh*kw]
        NLC = patches_t.shape[0] * patches_t.shape[1]  # N*L
        CKHW = patches_t.shape[2]                     # C*kh*kw

        # => reshape => [N*L, C*kh*kw]
        patches_2d = patches_t.reshape(NLC, CKHW)

        # w => shape [out_channels, C*kh*kw]
        # => w_t => shape [C*kh*kw, out_channels]
        w_t = w.transpose(0, 1)  # => [CKHW, out_channels]

        # multiply => shape [N*L, out_channels]
        out_2d = patches_2d @ w_t  # => [N*L, out_channels]

        # reshape => [N, L, out_channels]
        out_3d = out_2d.view(N, L, self.out_channels)

        # permute => [N, out_channels, L]
        out = out_3d.permute(0, 2, 1)

        # reshape => [N, out_channels, out_h, out_w]
        out = out.reshape(N, self.out_channels, out_h, out_w)

        # 4) Add bias => shape [out_channels], broadcast
        out = out + self.bias.view(1, -1, 1, 1)

        return out

# 3) Our CNN combining MyConv2dIm2Col and nn.Conv2d
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv => manual im2col
        self.conv1 = MyConv2dIm2Col(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Second conv => standard nn.Conv2d
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 2x2 max pool
        self.pool = nn.MaxPool2d(2, 2)

        # After conv1 + conv2 (stride=1, padding=1 => 32x32 stays)
        # Two pools => 16x16 => 8x8
        # Final channels=32 => flatten=32*8*8=2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 1) First conv => ReLU => pool => [N,16,16,16]
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # 2) Second conv => ReLU => pool => [N,32,8,8]
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten => shape [N, 2048]
        # Use .reshape instead of .view to avoid "view size is not compatible..." errors
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4) Instantiate Model, Loss, Optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5) Training Loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 6) Testing / Evaluation
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on CIFAR-10: {accuracy:.2f}%")
