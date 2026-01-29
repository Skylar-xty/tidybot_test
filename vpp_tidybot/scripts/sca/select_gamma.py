
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import from learning.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning import CollisionDataset, TransformerGamma, CSV_PATH

def select_gamma_threshold(model, loader, device, num_steps=1000):
    model.eval()
    all_gammas = []
    all_labels = []

    print("Computing Gamma values for validation set...")
    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(device), y.to(device)
            _, gamma = model(X)
            all_gammas.append(gamma.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_gammas = np.concatenate(all_gammas)
    all_labels = np.concatenate(all_labels)

    # Define range of thresholds
    min_g, max_g = all_gammas.min(), all_gammas.max()
    thresholds = np.linspace(min_g, max_g, num_steps)

    best_threshold = 0.0
    best_recall_viable = 0.0
    best_metrics = None

    results = []

    # Safety constraint: We want high recall for collisions (don't miss crashes)
    # The paper mentions 99.74% recall (likely for collision class).
    # So we look for thresholds that satisfy Recall(Collision) >= 0.99
    # And among those, maximize Recall(Viable).
    min_collision_recall = 0.99

    print(f"Sweeping {num_steps} thresholds from {min_g:.4f} to {max_g:.4f}...")

    for thr in thresholds:
        # Predictions: if gamma < thr, predict Collision (1), else Viable (0)
        # So:
        # Pred = 1 if gamma < thr
        # Pred = 0 if gamma >= thr
        
        preds = (all_gammas < thr).astype(int)
        
        # Confusion Matrix
        # Label 1: Collision, Label 0: Viable
        # TP: True Collision, Pred Collision
        # TN: True Viable, Pred Viable
        # FP: True Viable, Pred Collision (False Alarm)
        # FN: True Collision, Pred Viable (Missed Crash - DANGEROUS)
        
        tp = np.sum((preds == 1) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        
        total_collision = tp + fn
        total_viable = tn + fp
        
        recall_collision = tp / total_collision if total_collision > 0 else 0
        recall_viable = tn / total_viable if total_viable > 0 else 0
        accuracy = (tp + tn) / (total_collision + total_viable)
        
        results.append({
            "threshold": thr,
            "acc": accuracy,
            "recall_col": recall_collision,
            "recall_via": recall_viable
        })

    # Convert to DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Filter by safety constraint
    safe_candidates = df[df["recall_col"] >= min_collision_recall]
    
    if not safe_candidates.empty:
        # Pick the one with best Recall(Viable)
        best_row = safe_candidates.loc[safe_candidates["recall_via"].idxmax()]
        print(f"\nFound candidate satisfying Recall(Collision) >= {min_collision_recall}")
    else:
        print(f"\nWARNING: No threshold satisfies Recall(Collision) >= {min_collision_recall}")
        # Fallback: Maximize Recall(Collision)
        best_row = df.loc[df["recall_col"].idxmax()]
        
    best_threshold = best_row["threshold"]
    
    print("\nSelected Threshold Stats:")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Accuracy: {best_row['acc']*100:.2f}%")
    print(f"Recall (Collision): {best_row['recall_col']*100:.2f}%")
    print(f"Recall (Viable): {best_row['recall_via']*100:.2f}%")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df["threshold"], df["recall_col"], label="Recall (Collision)", color="red")
    plt.plot(df["threshold"], df["recall_via"], label="Recall (Viable)", color="green")
    plt.plot(df["threshold"], df["acc"], label="Accuracy", color="blue", linestyle="--")
    plt.axvline(best_threshold, color="black", linestyle=":", label=f"Selected {best_threshold:.2f}")
    plt.xlabel("Gamma Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep: Recall vs Gamma")
    plt.legend()
    plt.grid(True)
    plt.savefig("gamma_sweep.png")
    print("Saved sweep plot to gamma_sweep.png")
    
    return best_threshold

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    # Handle path differences depending on execution directory
    data_path = CSV_PATH
    if not os.path.exists(data_path):
        # Try finding it in the current directory (if running from sca/)
        local_path = os.path.basename(data_path)
        if os.path.exists(local_path):
            data_path = local_path
        else:
            print(f"Error: Dataset not found at {data_path} or {local_path}")
            return
    
    # Ensure it's a Path object if CollisionDataset expects it (though pandas reads strings too, 
    # CollisionDataset calls .exists() so we might need to wrap it if we changed it to string, 
    # but learning.py imports Path so we can use it or just rely on string if CollisionDataset handles it.
    # Looking at learning.py, it calls csv_path.exists(), so it must be a Path-like object with .exists() method 
    # OR we rely on the fact that we just checked existence. 
    # Actually CollisionDataset converts to Path or expects Path. 
    # Let's look at learning.py: 
    # class CollisionDataset(Dataset):
    #     def __init__(self, csv_path: Path):
    #         if not csv_path.exists(): ...
    # So we should pass a Path object.
    from pathlib import Path
    if isinstance(data_path, str):
        data_path = Path(data_path)

    ds = CollisionDataset(data_path)
    # Use same split as learning.py (approximate if seed not fixed, but good enough for demo)
    train_len = int(0.8 * len(ds))
    # We need to set seed to ensure we get the same split if we want to be rigorous, 
    # but learning.py didn't set a seed for the split. 
    # We'll just use the validation part.
    generator = torch.Generator().manual_seed(42) 
    train_ds, val_ds = random_split(ds, [train_len, len(ds) - train_len], generator=generator)
    
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)

    # Load Model
    model_path = "../models/network/transformer_gamma_epoch_60.pt"
    if not os.path.exists(model_path):
        # Try going up one more level if we are in sca/
        model_path_alt = "../../models/network/transformer_gamma_epoch_60.pt"
        if os.path.exists(model_path_alt):
            model_path = model_path_alt
            
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = TransformerGamma().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # Select Threshold
    best_gamma = select_gamma_threshold(model, val_loader, device)
    print(f"\nRecommended Gamma Threshold: {best_gamma}")

if __name__ == "__main__":
    main()
