from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import ViT
import time

class MNIST_model_test():
    def __init__(self):
        pass
    def train(self, model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)

        return running_loss/total, correct/total

    def val(self, model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)

                loss = criterion(out, labels)
                running_loss += loss.item() * imgs.size(0)
                pred = out.argmax(dim=1)

                correct += (pred == labels).sum().item()
                total += imgs.size(0)
        return running_loss/ total, correct/ total

    def get_DataLoaders(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_ds = datasets.MNIST(root='.', train=True, transform=transform, download=True)
        val_ds = datasets.MNIST(root='.', train=False, transform=transform, download=True)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,num_workers=4, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,num_workers=4, pin_memory=True, prefetch_factor=2)
        
        return train_loader, val_loader

    def main(self):
        EPOCHS = 10
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        train_loader, val_loader = self.get_DataLoaders()
        model = ViT(n_channels=1, num_classses=10, img_size=28, patch_size=4, embed_dim=64, num_heads=8, depth=6, mlp_ratio=4.0, qkv_bias=True, p=0.0)
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.02)

        best_acc = 0.0

        for epoch in range(1, EPOCHS+1):
            t0 = time.time()
            train_loss, train_acc = self.train(model, train_loader, criterion, optimizer, device=DEVICE)
            val_loss, val_acc = self.val(model, val_loader, criterion, device=DEVICE)

            t1 = time.time()
            print(f"Epoch: {epoch:02d}  time: {(t1-t0):.1f}s\ttrain_loss:{train_loss:.4f}\ttrain_acc:{train_acc:.4f}\tval_loss:{val_loss:.4f}\tval_acc:{val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
        print(f"Best Model Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    tester = MNIST_model_test()
    tester.main()