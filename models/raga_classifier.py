import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Block 1: CNN — local pattern extractor
# ---------------------------------------------------------------------------
# Input:  (batch, 1, n_bins, time_frames)   — treated like a 1-channel image
# Output: (batch, 128, reduced_time_frames) — flattened across frequency bins
# ---------------------------------------------------------------------------

class CNNBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # Each Conv2d layer scans a small (3×3) window across freq × time.
        # BatchNorm stabilises training. MaxPool shrinks the frequency dimension.
        # We do NOT pool along time — LSTM needs the time axis intact.

        self.cnn = nn.Sequential(
            # Layer 1: 1 input channel → 32 feature maps
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),   # halve freq bins, keep time

            # Layer 2: 32 → 64 feature maps
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),   # halve freq bins again

            # Layer 3: 64 → 128 feature maps
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),   # halve freq bins once more
        )

    def forward(self, x):
        # x: (batch, 1, n_bins, time_frames)
        x = self.cnn(x)
        # x: (batch, 128, n_bins//8, time_frames)

        batch, channels, freq, time = x.shape

        # Merge channel and frequency dimensions so LSTM sees one vector per time step
        x = x.permute(0, 3, 1, 2)              # → (batch, time, channels, freq)
        x = x.reshape(batch, time, channels * freq)  # → (batch, time, channels*freq)

        return x


# ---------------------------------------------------------------------------
# Block 2: LSTM — temporal sequence reader
# ---------------------------------------------------------------------------
# Input:  (batch, time_frames, channels*freq_bins)
# Output: (batch, time_frames, hidden*2)  — *2 because bidirectional
# ---------------------------------------------------------------------------

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        # Bidirectional: reads the sequence left→right AND right→left.
        # Helps because in ragas, what comes AFTER a note also matters.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_size = hidden_size * 2   # bidirectional doubles the output

    def forward(self, x):
        # x: (batch, time, input_size)
        out, _ = self.lstm(x)
        # out: (batch, time, hidden*2)
        return out


# ---------------------------------------------------------------------------
# Block 3: Raga head — classifies the full clip into one raga
# ---------------------------------------------------------------------------
# Input:  (batch, time_frames, hidden*2)
# Output: (batch, num_ragas)   — raw logits, NOT probabilities
# ---------------------------------------------------------------------------

class RagaHead(nn.Module):
    def __init__(self, input_size, num_ragas=10, dropout=0.4):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_ragas),
            # No softmax here — nn.CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        # x: (batch, time, hidden*2)
        # Mean-pool across time: summarise the whole clip into one vector
        x = x.mean(dim=1)          # → (batch, hidden*2)
        x = self.classifier(x)     # → (batch, num_ragas)
        return x


# ---------------------------------------------------------------------------
# Full model: CNN → LSTM → Raga head
# ---------------------------------------------------------------------------

class RagaClassifier(nn.Module):
    def __init__(self, n_bins=128, num_ragas=10):
        super().__init__()

        self.cnn = CNNBlock()

        # After 3× MaxPool(2,1), frequency bins shrink: n_bins → n_bins // 8
        freq_after_pool = n_bins // 8
        lstm_input_size = 128 * freq_after_pool   # 128 channels × reduced freq

        self.lstm = LSTMBlock(input_size=lstm_input_size)
        self.raga_head = RagaHead(input_size=self.lstm.output_size, num_ragas=num_ragas)

    def forward(self, x):
        # x: (batch, n_bins, time_frames)  — as loaded from features.npy

        x = x.unsqueeze(1)          # → (batch, 1, n_bins, time_frames)  [add channel dim]
        x = self.cnn(x)             # → (batch, time, 128 * freq_reduced)
        x = self.lstm(x)            # → (batch, time, 512)
        logits = self.raga_head(x)  # → (batch, num_ragas)

        return logits


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH      = 4
    N_BINS     = 128
    TIME_FRAMES = 344
    NUM_RAGAS  = 10

    model = RagaClassifier(n_bins=N_BINS, num_ragas=NUM_RAGAS)
    dummy = torch.randn(BATCH, N_BINS, TIME_FRAMES)

    logits = model(dummy)

    print("Input shape  :", dummy.shape)          # (4, 128, 344)
    print("Output shape :", logits.shape)         # (4, 10)
    print("Model params :", sum(p.numel() for p in model.parameters()), "parameters")
    print("Sanity check passed ✓")