from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, random_split 

def getdata(batch_size=32, val_split=0.2):
    Transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = EuroSAT(root='/.data', download=True, transform=Transforms)
    _= dataset [0]
    print(f"Downloaded to: {dataset.root}")
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader