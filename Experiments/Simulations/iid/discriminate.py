#This code is to generate iid DNA subsequences with correct prob distribution and discriminate them from the original DNA subsequences.
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from generate_iid import generate
import torch.utils.data as data_utils
import torch.nn.functional as F



# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
size = 50000000

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True, 
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size, 
#                                           shuffle=False)

# # Convolutional neural network (two convolutional layers)
# class ConvNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(7*7*32, num_classes)
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out

# model = ConvNet(num_classes).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# # Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def load_data(seq, window=500, stride=200):
    seq = np.squeeze(seq)
    data = strided_app(seq, window, stride)
    X = data[:, :]
    # Y = label * np.ones((X.shape[0], 1))
    X = np.expand_dims(X, 1)
    return X

# def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#     if shuffle:
#         indices = np.arange(inputs.shape[0])
#         np.random.shuffle(indices)
#     for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
#         # if(start_idx + batchsize >= inputs.shape[0]):
#         #   break;

#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batchsize]
#         else:
#             excerpt = slice(start_idx, start_idx + batchsize)
#         yield inputs[excerpt], targets[excerpt]

def split(X, Y, ratio=0.9):
    spl = np.int32(0.9*len(X))
    X_train = X[:spl]
    Y_train = Y[:spl]
    X_val = X[spl:]
    Y_val = Y[spl:]

    return X_train, Y_train, X_val, Y_val


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = (nn.Linear(63*4, 1))
        
    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = F.sigmoid(self.fc(out))
        return out

def preprocess(true_data, false_data):
    true_data = load_data(true_data)
    false_data = load_data(false_data)

    # print(true_data.shape, false_data.shape)
    true_labels = np.ones((true_data.shape[0], 1))
    false_labels = np.zeros((false_data.shape[0], 1))

    X = np.concatenate([true_data, false_data], axis=0)
    Y = np.concatenate([true_labels, false_labels], axis=0)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    print(X.shape, Y.shape)

    X = X[indices]
    Y = Y[indices]

    return X, Y

def main():
    ref = np.load('../reference.npy')
    true_data = ref[:size]
    false_data = generate(file='../reference.npy', size=size)
    true_test_data = ref[-size:]
    false_test_data = generate(file='../reference.npy', size=size)

    X, Y = preprocess(true_data, false_data)
    Xtest, Ytest = preprocess(true_test_data, false_test_data)

    X_train, Y_train, X_val, Y_val = split(X, Y)

    train = data_utils.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = data_utils.DataLoader(train, batch_size=64, shuffle=True)

    test = data_utils.TensorDataset(torch.FloatTensor(Xtest), torch.FloatTensor(Ytest))
    test_loader = data_utils.DataLoader(test, batch_size=128, shuffle=True)

    model = ConvNet().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Train the model
    total_step = len(train_loader)
    print("Training Start")
    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # print("MOhit")
            # print(images.size())
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


    # Test the model
    print("Testing Start")
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        loss_list = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_list.append(loss)

        print('Test Loss of the model on the 10000 sub sequences: {}'.format(np.mean(loss_list)))




    


if __name__ == "__main__":
    main()

