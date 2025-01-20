import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

import tensorflow_hub as hub
import tensorflow as tf

import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset

# Load metadata and model
df = pd.read_csv('perch/train_metadata.csv')
AUDIO_PATH = Path('perch/train_audio')
model_path = 'perch/perch_model'
model = hub.load(model_path)
model_labels_df = pd.read_csv(hub.resolve(model_path) + "/assets/label.csv")

SAMPLE_RATE = 32000
WINDOW = 5 * SAMPLE_RATE

# Prepare label mappings
index_to_label = sorted(df.primary_label.unique())
label_to_index = {v: k for k, v in enumerate(index_to_label)}
model_labels = {v: k for k, v in enumerate(model_labels_df.ebird2021)}
model_bc_indexes = [model_labels[label] if label in model_labels else -1 for label in index_to_label]

# Identify missing birds
missing_birds = set(np.array(index_to_label)[np.array(model_bc_indexes) == -1])
print(missing_birds)

# Dataset for audio loading
class AudioDataset(Dataset):
    def __len__(self):
        return len(df)

    def __getitem__(self, i):
        filename = df.filename[i]
        audio = torchaudio.load(AUDIO_PATH / filename)[0].numpy()[0]
        # Pad or trim audio to a fixed length
        if len(audio) < WINDOW:
            audio = np.pad(audio, (0, WINDOW - len(audio)), mode='constant')
        else:
            audio = audio[:WINDOW]
        return torch.tensor(audio, dtype=torch.float32), filename


# DataLoader setup
dataloader = DataLoader(AudioDataset(), batch_size=16)

# Storage for embeddings and predictions
all_embeddings = {}
all_predictions = {}

with tf.device('/cpu:0'):
    for audio, filename in tqdm(dataloader):
        audio = audio[0]
        filename = filename[0]
        file_embeddings = []
        file_predictions = []
        for i in range(0, len(audio), WINDOW):
            clip = audio[i:i+WINDOW]
            if len(clip) < WINDOW:
                clip = np.concatenate([clip, np.zeros(WINDOW - len(clip))])
            result = model.infer_tf(clip[None, :])
            print(result.keys())
            print(result["embedding"])
            print(result["label"])
            file_embeddings.append(result["embedding"].numpy())
            prediction = np.concatenate([result["label"], -100], axis=None) # add -100 logit for unpredicted birds
            file_predictions.append(prediction[model_bc_indexes])
        all_embeddings[filename] = np.stack(file_embeddings)
        all_predictions[filename] = np.stack(file_predictions)

torch.save(all_embeddings, 'embeddings.pt')
torch.save(all_predictions, 'predictions.pt')
