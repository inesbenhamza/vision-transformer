from dataset import getdata
from model import VisionTransformer

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


device = torch.device("mps" if torch.mps.is_available else "cpu")
batch_size = 32
learning_rate = 3e-4
num_classes = 10  

train_loader, val_loader = getdata(batch_size=batch_size, )

model = VisionTransformer(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 5


for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {acc:.2f}%")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Val Loss: {val_loss:.4f}, Accuracy: {acc:.2f}%")



torch.save(model.state_dict(), "vit_eurosat.pth")