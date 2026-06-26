import torch
import torch.nn as nn
from raga_classifier import CNNBlock, LSTMBlock


# ---------------------------------------------------------------------------
# Block 3 (alternate): Swar head — predicts one swar per time frame
# ---------------------------------------------------------------------------
# Input:  (batch, time_frames, hidden*2)
# Output: (batch, time_frames, num_swars)  — raw logits per frame
# ---------------------------------------------------------------------------
# KEY DIFFERENCE from RagaHead:
#   RagaHead   → mean-pools time → one label per clip
#   SwarHead   → keeps time axis → one label per frame

class SwarHead(nn.Module):
    def __init__(self, input_size, num_swars=8, dropout=0.3):
        super().__init__()

        # Applied independently to every time frame (like a 1D convolution over time)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_swars),
            # No softmax — nn.CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        # x: (batch, time, hidden*2)
        x = self.classifier(x)   # → (batch, time, num_swars)
        return x


# ---------------------------------------------------------------------------
# Full model: CNN → LSTM → Swar head
# ---------------------------------------------------------------------------

class PitchDetector(nn.Module):
    def __init__(self, n_bins=128, num_swars=8):
        super().__init__()

        self.cnn = CNNBlock()

        freq_after_pool = n_bins // 8
        lstm_input_size = 128 * freq_after_pool

        self.lstm = LSTMBlock(input_size=lstm_input_size)
        self.swar_head = SwarHead(input_size=self.lstm.output_size, num_swars=num_swars)

    def forward(self, x):
        # x: (batch, n_bins, time_frames)

        x = x.unsqueeze(1)          # → (batch, 1, n_bins, time_frames)
        x = self.cnn(x)             # → (batch, time, 128 * freq_reduced)
        x = self.lstm(x)            # → (batch, time, 512)
        logits = self.swar_head(x)  # → (batch, time, num_swars)

        return logits


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH       = 4
    N_BINS      = 128
    TIME_FRAMES = 344
    NUM_SWARS   = 8

    model = PitchDetector(n_bins=N_BINS, num_swars=NUM_SWARS)
    dummy = torch.randn(BATCH, N_BINS, TIME_FRAMES)

    logits = model(dummy)

    print("Input shape  :", dummy.shape)     # (4, 128, 344)
    print("Output shape :", logits.shape)    # (4, 344, 8)
    print("Model params :", sum(p.numel() for p in model.parameters()), "parameters")
    print("Sanity check passed ✓")