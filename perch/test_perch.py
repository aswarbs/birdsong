import torch
import pandas as pd
import numpy as np

# Load saved embeddings and predictions
all_embeddings = torch.load('perch/perch_model/embeddings.pt')
all_predictions = torch.load('perch/perch_model/predictions.pt')

# Load metadata
df = pd.read_csv('perch/perch_model/train_metadata.csv')

# Prepare label mappings
index_to_label = sorted(df.primary_label.unique())
label_to_index = {v: k for k, v in enumerate(index_to_label)}

# Aggregate predictions per file
aggregated_predictions = {
    filename: predictions.mean(axis=0)  # Average logits across windows
    for filename, predictions in all_predictions.items()
}

# Align metadata with aggregated predictions
df = df[df.filename.isin(aggregated_predictions.keys())].reset_index(drop=True)

# Convert aggregated logits to predicted classes
predicted_classes = torch.tensor([
    aggregated_predictions[filename].argmax() for filename in df.filename
])

# Convert ground truth labels to indices
actual_classes = torch.tensor([
    label_to_index[label] for label in df.primary_label
])

# Compute accuracy
correct = predicted_classes == actual_classes
accuracy = correct.float().mean()
print(f'Accuracy: {accuracy.item()}')
