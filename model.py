import torch.nn as nn
import torch
import torch.nn.functional as F
class CNNModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Add pooling and FC layers based on your image size
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152 , 256)  # adjust input size based on your image dimensions
        self.fc2 = nn.Linear(256, num_classes)  # for binary classification (2 classes)
        self.dropout = nn.Dropout2d(p=0.5)
    def forward(self, x):
          # Convolutional blocks
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
    
    # Classifier
        x = x.view(x.size(0), -1)
        # print(f"Flattened shape: {x.shape}")
        
        # Fully connected layers with dropout and Leaky ReLU activation
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        
        # Final output layer (classification)
        x = self.fc2(x)
        return x

model = CNNModel(num_classes=1)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))


def predict(tensor, model):
    model.eval()
    yhat = model(tensor)  # Get the model output
    yhat = yhat.clone().detach()  # Ensure the output is not part of the computation graph
    return yhat  # Return the tenso

from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(256), nullable=False)


class DiagnosisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)  # Link to session['username']
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    # graph_image = db.Column(db.Text, nullable=True)  # base64 image string
    symptoms = db.Column(db.String(300), nullable=True)
    pain_level = db.Column(db.String(100), nullable=True)
    family_history = db.Column(db.String(100), nullable=True)