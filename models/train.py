import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import yaml

from raga_classifier import RagaClassifier
from pitch_detector import PitchDetector


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class RaagaDataset(Dataset):
    def __init__(self, features, labels, pitch_labels):
        self.features     = torch.tensor(features,     dtype=torch.float32)
        self.labels       = torch.tensor(labels,       dtype=torch.long)
        self.pitch_labels = torch.tensor(pitch_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.pitch_labels[idx]


# ---------------------------------------------------------------------------
# 2. Load config + data
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    features_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", config["paths"]["features"])
    )
    features     = np.load(os.path.join(features_dir, "features.npy"))
    labels       = np.load(os.path.join(features_dir, "labels.npy"))
    pitch_labels = np.load(os.path.join(features_dir, "pitch_labels.npy"))

    print(f"Loaded features     : {features.shape}")
    print(f"Loaded labels       : {labels.shape}")
    print(f"Loaded pitch_labels : {pitch_labels.shape}")

    return features, labels, pitch_labels


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

def train(config):
    # --- Hyperparameters ---
    EPOCHS     = config.get("training", {}).get("epochs", 20)
    BATCH_SIZE = config.get("training", {}).get("batch_size", 16)
    LR         = config.get("training", {}).get("lr", 1e-3)
    VAL_SPLIT  = 0.2
    N_BINS     = config["features"]["n_bins"]
    NUM_RAGAS  = config.get("model", {}).get("num_ragas", 10)
    NUM_SWARS  = config.get("model", {}).get("num_swars", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # --- Data ---
    features, labels, pitch_labels = load_data(config)
    dataset = RaagaDataset(features, labels, pitch_labels)

    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples : {train_size}")
    print(f"Val samples   : {val_size}\n")

    # --- Models ---
    raga_model  = RagaClassifier(n_bins=N_BINS, num_ragas=NUM_RAGAS).to(device)
    pitch_model = PitchDetector(n_bins=N_BINS,  num_swars=NUM_SWARS).to(device)

    # Single optimizer trains both models together
    all_params = list(raga_model.parameters()) + list(pitch_model.parameters())
    optimizer  = torch.optim.Adam(all_params, lr=LR)

    raga_loss_fn  = nn.CrossEntropyLoss()
    # pitch labels: (batch, time) → CrossEntropyLoss expects (batch, classes, time)
    swar_loss_fn  = nn.CrossEntropyLoss()

    # --- Loop ---
    for epoch in range(1, EPOCHS + 1):
        raga_model.train()
        pitch_model.train()

        train_raga_loss  = 0.0
        train_swar_loss  = 0.0
        train_raga_correct = 0
        train_total        = 0

        for features_batch, labels_batch, pitch_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch   = labels_batch.to(device)
            pitch_batch    = pitch_batch.to(device)

            optimizer.zero_grad()

            # Forward
            raga_logits  = raga_model(features_batch)   # (batch, num_ragas)
            swar_logits  = pitch_model(features_batch)  # (batch, time, num_swars)

            # Losses
            loss_raga = raga_loss_fn(raga_logits, labels_batch)

            # CrossEntropyLoss needs (batch, classes, time) for sequence targets
            loss_swar = swar_loss_fn(
                swar_logits.permute(0, 2, 1),   # → (batch, num_swars, time)
                pitch_batch                      # → (batch, time)
            )

            # Combined loss — equal weight for now
            loss = loss_raga + loss_swar
            loss.backward()
            optimizer.step()

            train_raga_loss += loss_raga.item()
            train_swar_loss += loss_swar.item()

            preds = raga_logits.argmax(dim=1)
            train_raga_correct += (preds == labels_batch).sum().item()
            train_total        += labels_batch.size(0)

        # --- Validation ---
        raga_model.eval()
        pitch_model.eval()

        val_raga_loss    = 0.0
        val_raga_correct = 0
        val_total        = 0

        with torch.no_grad():
            for features_batch, labels_batch, pitch_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch   = labels_batch.to(device)

                raga_logits = raga_model(features_batch)
                val_raga_loss += raga_loss_fn(raga_logits, labels_batch).item()

                preds = raga_logits.argmax(dim=1)
                val_raga_correct += (preds == labels_batch).sum().item()
                val_total        += labels_batch.size(0)

        # --- Logging ---
        train_raga_acc = 100 * train_raga_correct / train_total
        val_raga_acc   = 100 * val_raga_correct   / val_total

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Raga loss: {train_raga_loss/len(train_loader):.4f} | "
            f"Swar loss: {train_swar_loss/len(train_loader):.4f} | "
            f"Train acc: {train_raga_acc:.1f}% | "
            f"Val acc: {val_raga_acc:.1f}%"
        )

    # --- Save models ---
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)

    torch.save(raga_model.state_dict(),  os.path.join(save_dir, "raga_classifier.pt"))
    torch.save(pitch_model.state_dict(), os.path.join(save_dir, "pitch_detector.pt"))

    print(f"\nModels saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()
    train(config)