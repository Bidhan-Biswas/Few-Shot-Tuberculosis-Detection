import pandas as pd
import os

csv_path = r"D:\Research Work\TB Classification\Shenzhen Dataset\shenzhen_metadata.csv"
img_dir = r"D:\Research Work\TB Classification\Shenzhen Dataset\images"

df = pd.read_csv(csv_path)

print("CSV study_id examples:")
print(df["study_id"].head(10).tolist())

print("\nImage filename examples:")
print(os.listdir(img_dir)[:10])