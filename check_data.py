
import pandas as pd

# Load the training dataset
data = pd.read_excel("dataset/Data_Train.xlsx")

# Display basic information
print("Dataset Shape:", data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDataset Information:")
print(data.info())

print("\nPrice Statistics:")
print(data["Price"].describe())
