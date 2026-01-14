import pandas as pd
import numpy as np

models = {
    'Model_0.90': './0.90.csv',
    'Model_0.92': './0.92.csv',
}

predictions = {}
for name, path in models.items():
    try:
        df = pd.read_csv(path)
        predictions[name] = df['label'].values
        print(f"Loaded {name}: {len(df)} samples")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

if len(predictions) < 2:
    print("Error: Need at least 2 models")
    exit()

model_names = list(predictions.keys())
n_samples = len(predictions[model_names[0]])

agree = np.sum(predictions[model_names[0]] == predictions[model_names[1]])
agree_rate = agree / n_samples * 100
print(f"\nAgreement: {agree}/{n_samples} ({agree_rate:.1f}%)")

disagree = predictions[model_names[0]] != predictions[model_names[1]]
n_disagree = np.sum(disagree)
print(f"Disagreement: {n_disagree} samples")

if n_disagree > 0:
    print(f"\nDisagreement details:")
    for idx in np.where(disagree)[0]:
        print(f"  Sample {idx}: {model_names[0]}={predictions[model_names[0]][idx]}, {model_names[1]}={predictions[model_names[1]][idx]}")
