import os
import numpy as np
import yaml

def generate_dummy_data():
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Define paths
    features_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["paths"]["features"]))
    os.makedirs(features_dir, exist_ok=True)
    
    # Extract audio/feature settings from config
    sample_rate = config["audio"]["sample_rate"]
    duration = config["audio"]["duration"]
    hop_length = config["features"]["hop_length"]
    n_bins = config["features"]["n_bins"]
    
    num_samples = 120
    num_bins = n_bins
    num_time_frames = (sample_rate * duration) // hop_length + 1
    num_ragas = 10
    num_swars = 8  # 0: Rest/Silence, 1: Sa, 2: Re, 3: Ga, 4: Ma, 5: Pa, 6: Dha, 7: Ni
    
    print(f"Generating dummy dataset in: {features_dir}")
    
    # 1. Generate features.npy (spectrograms): shape (N, n_bins, num_time_frames)
    features_path = os.path.join(features_dir, "features.npy")
    features = np.random.randn(num_samples, num_bins, num_time_frames).astype(np.float32)
    np.save(features_path, features)
    print(f" - Saved features.npy: shape {features.shape}, dtype {features.dtype}")
    
    # 2. Generate labels.npy (raga labels): shape (N,)
    labels_path = os.path.join(features_dir, "labels.npy")
    labels = np.random.randint(0, num_ragas, size=(num_samples,)).astype(np.int64)
    np.save(labels_path, labels)
    print(f" - Saved labels.npy (raga labels): shape {labels.shape}, dtype {labels.dtype}")
    
    # 3. Generate pitch_labels.npy (swar labels for frame-level pitch detection): shape (N, num_time_frames)
    pitch_labels_path = os.path.join(features_dir, "pitch_labels.npy")
    pitch_labels = np.random.randint(0, num_swars, size=(num_samples, num_time_frames)).astype(np.int64)
    np.save(pitch_labels_path, pitch_labels)
    print(f" - Saved pitch_labels.npy (swar labels): shape {pitch_labels.shape}, dtype {pitch_labels.dtype}")
    
    print("Dummy data generation completed successfully!")


if __name__ == "__main__":
    generate_dummy_data()
