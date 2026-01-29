import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from tqdm import tqdm

# Dataset
CSV_PATH = Path("sca/collision_results_3_limit_sampling_new.csv")

# CSV_PATH = Path("sca/collision_results_6_limit_sampling_new_8m.csv")
LABEL_COL = 21
epochs = 60
class CollisionDataset(Dataset):
    def __init__(self, csv_path: Path):
        if not csv_path.exists():
            # Fallback to default if specific file not found
            # csv_path = Path("sca/collision_results.csv")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, 0:14].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, LABEL_COL].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model -
class TransformerGamma(nn.Module):
    def __init__(self, input_dim=14, d_model=64, nhead=2, num_layers=4, dropout=0.1):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, q):
        q = q.unsqueeze(-1)
        x = self.linear_encoder(q)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)
        gamma_logits = self.classifier(x)
        γ1, γ2 = gamma_logits[:, 0], gamma_logits[:, 1]
        Γ = γ1 - γ2
        return gamma_logits, Γ


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = (all_preds == all_labels).mean()
    # Use pos_label=1 (collision) for recall to measure safety
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=1)
    return accuracy, recall

@torch.no_grad()
def evaluate_with_fixed_gamma(model, loader, device, threshold=2.0):
    model.eval()
    all_preds = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        _, gamma = model(X)
        preds = (gamma < threshold).long()
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, pos_label=1)

    print(f"\nUsing fixed Γ threshold = {threshold}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Recall:   {recall * 100:.2f}%")

# Training
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CollisionDataset(CSV_PATH)
    train_len = int(0.8 * len(ds))
    train_ds, test_ds = random_split(ds, [train_len, len(ds) - train_len])
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = TransformerGamma().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()

    loss_log, acc_log, recall_log, step_log = [], [], [], []
    best_acc = 0.0  # Track the best accuracy
    best_model_path = "../models/network/transformer_gamma_best.pt"

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, _ = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            current_loss = loss.item()
            epoch_loss += current_loss * X.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        epoch_loss /= train_len
        acc, recall = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}] Avg Loss={epoch_loss:.4f}  Acc={acc*100:.2f}%  Recall={recall*100:.2f}%")
        loss_log.append(epoch_loss)
        acc_log.append(acc * 100)
        recall_log.append(recall * 100)
        step_log.append(epoch)

        # Save model every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = f"../models/network/transformer_gamma_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save the best model based on accuracy
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated with accuracy {best_acc*100:.2f}% and saved to {best_model_path}")

    save_path = "../models/network/transformer_gamma_new_8m.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # ---------- 5. Plotting ----------
    plt.figure(figsize=(7, 4.5))
    plt.plot(step_log, loss_log, marker='o', color='crimson', linewidth=2)
    plt.title("Training Loss per Epoch", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("loss_per_epoch.png")
    plt.show()

    plt.figure(figsize=(7, 4.5))
    plt.plot(step_log, acc_log, marker='o', color='green', linewidth=2)
    plt.title("Test Accuracy per Epoch", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("accuracy_per_epoch.png")
    plt.show()

    plt.figure(figsize=(7, 4.5))
    plt.plot(step_log, recall_log, marker='o', color='blue', linewidth=2)
    plt.title("Test Recall per Epoch", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Recall (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("recall_per_epoch.png")
    plt.show()



if __name__ == "__main__":
    main()