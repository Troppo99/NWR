import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
import math
import argparse


class Config:
    BASE_DIR = r"D:\NWR\sources\AlAhad\images\wallpaper1"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VALID_DIR = os.path.join(BASE_DIR, "valid")

    # Model and training parameters
    MODEL_DIR = os.path.join(BASE_DIR, "Models", "Wallpaper")
    CHECKPOINT_FILENAME = "checkpoint.pth.tar"
    BEST_MODEL_FILENAME = "best_model.pth"
    FINAL_MODEL_FILENAME = "wallpaper_v1.pth"
    MODEL_PATH = os.path.join(BASE_DIR, "Models", "Wallpaper", "best.pt")  # Path to pre-trained model if any

    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    STEP_SIZE = 10
    GAMMA = 0.5

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations
    TRANSFORM = transforms.Compose([transforms.Resize((640, 640)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    # Checkpointing
    LOAD_CHECKPOINT = True  # Set to True to load from existing checkpoint
    SAVE_CHECKPOINT_EVERY_EPOCH = True  # Save checkpoint every epoch
    SAVE_BEST_MODEL = True  # Save the best model based on validation loss


class CustomImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.filenames = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"))])
        self.length = len(self.filenames)
        if self.length == 0:
            raise RuntimeError(f"No images found in directory: {folder}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                img_path = os.path.join(self.folder, self.filenames[idx])
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, img
            except FileNotFoundError:
                print(f"Warning: File {self.filenames[idx]} not found. Skipping.")
                del self.filenames[idx]
                self.length -= 1
                if self.length == 0:
                    raise RuntimeError("No images left to load.")
                idx = idx % self.length


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved at '{filename}'")


def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch, best_valid_loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float("inf")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an Autoencoder on Wallpaper Images")
    parser.add_argument("--base_dir", type=str, default=Config.BASE_DIR, help="Base directory containing 'train' and 'valid' folders")
    parser.add_argument("--model_dir", type=str, default=Config.MODEL_DIR, help="Directory to save models and checkpoints")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS, help="Number of workers for DataLoader")
    parser.add_argument("--learning_rate", type=float, default=Config.LEARNING_RATE, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--step_size", type=int, default=Config.STEP_SIZE, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=Config.GAMMA, help="Gamma for learning rate scheduler")
    parser.add_argument("--checkpoint", type=str, default=Config.CHECKPOINT_FILENAME, help="Filename for saving/loading checkpoints")
    parser.add_argument("--save_best", action="store_true", help="Save the best model based on validation loss")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    Config.BASE_DIR = args.base_dir
    Config.TRAIN_DIR = os.path.join(Config.BASE_DIR, "train")
    Config.VALID_DIR = os.path.join(Config.BASE_DIR, "valid")
    Config.MODEL_DIR = args.model_dir
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_WORKERS = args.num_workers
    Config.LEARNING_RATE = args.learning_rate
    Config.NUM_EPOCHS = args.num_epochs
    Config.STEP_SIZE = args.step_size
    Config.GAMMA = args.gamma
    Config.CHECKPOINT_FILENAME = args.checkpoint
    Config.SAVE_BEST_MODEL = args.save_best
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    try:
        train_dataset = CustomImageDataset(Config.TRAIN_DIR, transform=Config.TRANSFORM)
        print(f"Loaded {len(train_dataset)} training images.")
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)

    try:
        valid_dataset = CustomImageDataset(Config.VALID_DIR, transform=Config.TRANSFORM)
        print(f"Loaded {len(valid_dataset)} validation images.")
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)

    # Initialize model, loss function, optimizer, and scheduler
    model = Autoencoder().to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.STEP_SIZE, gamma=Config.GAMMA)

    # Initialize checkpoint
    checkpoint_path = os.path.join(Config.MODEL_DIR, Config.CHECKPOINT_FILENAME)
    start_epoch = 0
    best_valid_loss = float("inf")
    if Config.LOAD_CHECKPOINT and os.path.exists(checkpoint_path):
        start_epoch, best_valid_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    else:
        print("Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}] Training")
        for data, _ in train_loader_iter:
            data = data.to(Config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_iter.set_postfix({"Loss": loss.item()})

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_loader_iter = tqdm(valid_loader, desc=f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}] Validation")
        with torch.no_grad():
            for data, _ in valid_loader_iter:
                data = data.to(Config.DEVICE)
                output = model(data)
                loss = criterion(output, data)
                valid_loss += loss.item()
                valid_loader_iter.set_postfix({"Loss": loss.item()})

        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}], Valid Loss: {avg_valid_loss:.6f}")

        # Step the scheduler
        scheduler.step()

        # Save checkpoint
        if Config.SAVE_CHECKPOINT_EVERY_EPOCH:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_valid_loss": best_valid_loss,
            }
            save_checkpoint(checkpoint, checkpoint_path)

        # Save best model
        if Config.SAVE_BEST_MODEL and avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_path = os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_FILENAME)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss {best_valid_loss:.6f}")

    print("Training selesai!")

    # Save the final model
    final_model_path = os.path.join(Config.MODEL_DIR, Config.FINAL_MODEL_FILENAME)
    torch.save(model.state_dict(), final_model_path)
    print(f"Model telah disimpan sebagai '{final_model_path}'")


if __name__ == "__main__":
    main()
