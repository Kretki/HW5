import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FaceNetModel
from tqdm import tqdm
import yaml

def train(cfg_path="params.yaml"):
    with open(cfg_path) as f:
        params = yaml.safe_load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEVICE: {device}")

        transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        train_ds = datasets.ImageFolder(os.path.join(params['preprocessed_data']['train_dir'], ".."), transform=transform)
        train_dl = DataLoader(train_ds, params['train']['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print("----LOADED----")
        model = FaceNetModel(classify=True, num_classes=params['train']['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.head.parameters(), lr=params['train']['lr'])
        print("----START----")
        for epoch in range(1, params['train']['epochs']+1):
            model.train()
            running_loss = 0.0
            images_processed = 0
            progress_bar = tqdm(
                train_dl,
                desc=f"Epoch {epoch}/{params['train']['epochs']}",
                leave=False
            )
            for batch_idx, (images, labels) in enumerate(progress_bar):

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                progress_bar.set_postfix(dict(loss=f"{avg_loss:.4f}"))

            print(f"Epoch {epoch}/{params['train']['epochs']} — Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "face_classifier.pt")
        print("----TRAINED----")

if __name__ == "__main__":
    train()