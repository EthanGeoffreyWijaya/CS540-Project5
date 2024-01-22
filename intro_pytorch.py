import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    ct = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if (training):
        data_set = datasets.FashionMNIST('./data',train=True,download=True,transform=ct)
    else:
        data_set = datasets.FashionMNIST('./data', train=False,transform=ct)
    
    return torch.utils.data.DataLoader(data_set, batch_size = 64)


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    return model



def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for e in range(T):
        total = 0
        correct = 0
        runLoss = 0.0
        for data in train_loader:
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            runLoss += loss.item() * 64
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print("Train Epoch: " + str(e) + " Accuracy: " + str(correct) 
              + "/" + str(total) + "(%.2f%%) Loss: %.3f" % (100*correct/total, runLoss/60000))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    correct = 0
    total = 0
    runLoss = 0.0
    with torch.no_grad():
        for data in test_loader:
           images, labels = data
           outputs = model(images)
           loss = criterion(outputs, labels)
           runLoss += loss.item() * 64
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted==labels).sum().item()

        if (show_loss):
            print("Average loss: %.4f" % (runLoss / 60000))

        print("Accuracy: %.2f%%" % (100*correct/total))




def predict_label(model, test_images, index):
    class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
    output = model(test_images[index])
    prob = F.softmax(output, dim=1)
    for i in range(3):
        m = torch.max(prob, 1)
        print(class_names[m.indices] + ": %.2f%%" % (100 * float(m.values)))
        prob[0][m.indices] = 0




if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    """
    trainl = get_data_loader()
    print(type(trainl))
    print(trainl.dataset)
    testl = get_data_loader(False)
    model = build_model()
    print(model)
    train_model(model, trainl, criterion, 5)
    evaluate_model(model, testl, criterion, False)
    test_images, _ = next(iter(testl))
    predict_label(model, test_images, 2)
    """
