import pandas as pd
import os


base_dataset = './dataset/formatted-reddit-dataset/'
save_path = './dataset/concat-formatted-reddit-dataset.csv'

df = pd.DataFrame()
files = [f for f in os.listdir(base_dataset) if f.endswith('.csv')]
for file in files:
    df = pd.concat([df, pd.read_csv(f"{base_dataset}{file}")])

df.to_csv(save_path, index=False)