import torchaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os

model = hub.load('perch/perch_model')
model_labels_df = pd.read_csv(hub.resolve('perch/perch_model') + "/assets/label.csv")
eBird_csv = pd.read_csv('eBird_taxonomy_v2024.csv')
SAMPLE_RATE = 32000
WINDOW = 5 * SAMPLE_RATE
index_to_label = sorted(model_labels_df.ebird2021)

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
    audio = waveform.numpy()[0]
    return np.pad(audio, (0, max(0, WINDOW - len(audio))), mode='constant')[:WINDOW]

def classify_bird(audio, top_n=5):
    with tf.device('/cpu:0'):
        result = model.infer_tf(audio[None, :])
        label = result.get("label")
        if label is None:
            return [("Unknown", 0.0)]
        probabilities = tf.nn.softmax(label).numpy()[0]  # Convert logits to probabilities
    
    top_indices = np.argsort(probabilities)[-top_n:][::-1]  # Get indices of top N species
    return [(get_common_name(index_to_label[int(i)]), probabilities[int(i)]) for i in top_indices]

def get_common_name(species_code):
    row = eBird_csv[eBird_csv['SPECIES_CODE'] == species_code]
    return row.iloc[0]['PRIMARY_COM_NAME'] if not row.empty else "Unknown"

def classify_bird_from_mp3(mp3_path):
    top_species = classify_bird(load_audio(mp3_path))
    print(f"file name: {os.path.basename(mp3_path)}")
    for species, confidence in top_species:
        print(f"  classification: {species}, confidence: {confidence:.4f}")

if __name__ == "__main__":
    for filename in os.listdir('soundclips'):
        if filename.endswith(".mp3"):
            classify_bird_from_mp3(os.path.join('soundclips', filename))
