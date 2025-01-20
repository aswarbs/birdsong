import pandas as pd
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load filtered UK bird songs metadata
uk_bird_songs = pd.read_csv('perch/uk_bird_songs.csv')

# Load model predictions and embeddings
all_predictions = torch.load('perch/predictions.pt')

# Prepare label mappings (assuming label_to_index and index_to_label are defined)
df = pd.read_csv('perch/train_metadata.csv')
model_labels_df = pd.read_csv("perch/perch_model/assets/label.csv")
#index_to_label = sorted(uk_bird_songs.primary_label.unique())  # Ensure labels are sorted properly
# Prepare label mappings
index_to_label = sorted(df.primary_label.unique())
label_to_index = {v: k for k, v in enumerate(index_to_label)}
model_labels = {v: k for k, v in enumerate(model_labels_df.ebird2021)}
model_bc_indexes = [model_labels[label] if label in model_labels else -1 for label in index_to_label]

label_to_index = {v: k for k, v in enumerate(index_to_label)}

# Align predictions with UK bird songs
uk_predictions = {
    filename: predictions
    for filename, predictions in all_predictions.items()
    if filename in uk_bird_songs.filename.values
}

# Check alignment of filenames
uk_bird_songs = uk_bird_songs[uk_bird_songs.filename.isin(uk_predictions.keys())].reset_index(drop=True)

# Convert aggregated logits to predicted classes
predicted_classes = torch.tensor([uk_predictions[filename].mean(axis=0).argmax() for filename in uk_bird_songs.filename])

# Convert ground truth labels to indices using the 'label_to_index' mapping
actual_classes = torch.tensor([label_to_index[label] for label in uk_bird_songs.primary_label])

# Ensure that predicted labels match the possible range of actual labels
predicted_labels_set = set(predicted_classes.numpy())
actual_labels_set = set(actual_classes.numpy())

# If any labels are outside the expected range, clip them or adjust the mapping
predicted_classes = torch.tensor([min(max(predicted_class.item(), 0), len(index_to_label) - 1) for predicted_class in predicted_classes])

# Compute accuracy
accuracy = accuracy_score(actual_classes, predicted_classes)
print(f"Accuracy on UK bird songs: {accuracy:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)