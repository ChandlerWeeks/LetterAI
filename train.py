import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image

#!pip install torch
#!pip install pandas
#!pip install numpy


# dataset: https://www.kaggle.com/datasets/crawford/emnist/data?select=emnist-byclass-test.csv

# Convolutional neural network 
class ConvNet(nn.Module):
    def __init__(self):
        label_map = pd.read_csv("./training/emnist-balance-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None, 
                        squeeze=True
                       )
        label_dictionary = {}
        for index, label in enumerate(label_map):
            label_dictionary[index] = chr(label)
        self.mapping = label_dictionary

        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)  # Adjusted to match the output of layer2
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 47)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
    def train_model(self):
        # Load the training data
        self.get_train_data()

        # Define the loss function and the optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        avg = 0
        # Training loop
        for epoch in range(self.train_images.shape[0]):
            #if epoch > 10000:
                #break
            # Zero the gradients
            optimizer.zero_grad()

            # Separate the label and the image
            label = torch.tensor([self.train_labels[epoch]], dtype=torch.long)
            image = torch.tensor(self.train_images[epoch], dtype=torch.float32)

            # Flatten the image
            image = image.view(1, 1, 28, 28)

            # Forward pass
            output = self.forward(image)

            #print prediction vs actual
            #print(f"Prediction: {self.mapping[torch.argmax(output).item()]}, Actual: {self.mapping[label.item()]}")

            # Calculate and display the loss
            loss = criterion(output, label)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            avg += loss.item()
            if epoch % 1000 == 0 and epoch != 0:
                print(f"average loss at {epoch}: {avg / 1000}")
                avg = 0


        torch.save(self.state_dict(), 'model.pth')

    def predict(self, x):
        return self.forward(x)
    
    def load_model(self):
        self.load_state_dict(torch.load('model.pth'))
        self.eval()
    
    def get_train_data(self):
        train_data = pd.read_csv("./training/emnist-balanced-train.csv")
        x1 = np.array(train_data.iloc[:, 1:])

        self.train_images = x1.reshape(112799, 28, 28, 1)
    
        # Set the labels
        self.train_labels = train_data.iloc[:, 0].values

    def get_test_data(self):
        test_data = pd.read_csv("./training/emnist-balanced-test.csv")
        x1 = np.array(test_data.iloc[:, 1:])
        # normalize
        test_x = x1 / 255
        test_x_number = test_x.shape[0]
    
        self.test_images = x1.reshape(test_x_number, 28, 28, 1)
    
        # Set the labels
        self.test_labels = test_data.iloc[:, 0].values

    def save_image_as_png(self, image, label):
        image = image.cpu().numpy()
        image_2d = image.reshape(28, 28)
        image_2d = (image_2d * 255).astype(np.uint8)
        image = Image.fromarray(image_2d)
        image.save('example.png')
        print(label)

    def convert_image(self, image):
        image_array = np.array(image)
        image_array = 255 - image_array
        image_data = image_array.reshape(1, 1, 28, 28) / 255

        return torch.from_numpy(image_data).float()
    
    def test(self):
        self.get_test_data()
        self.load_model()
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(1000):
                i = np.random.randint(0, len(self.test_images))
                label = torch.tensor([self.test_labels[i]], dtype=torch.long)
                image = torch.tensor(self.test_images[i], dtype=torch.float32)
                #self.save_image_as_png(image, label)
                image = image.view(1, 1, 28, 28)
                outputs = self.forward(image)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        print(f"Accuracy: {100 * correct / total}")
    

if __name__ == "__main__":
    nn = ConvNet()
    # nn.train_model()
    nn.test()