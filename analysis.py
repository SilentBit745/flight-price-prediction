
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("dataset/Data_Train.xlsx")

# Remove missing values
df = df.dropna()

# 1. Average price by airline
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Airline", y="Price")
plt.xticks(rotation=45)
plt.title("Average Flight Price by Airline")
plt.tight_layout()
plt.savefig("airline_prices.png")
plt.show()

# 2. Price by number of stops
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Total_Stops", y="Price")
plt.title("Flight Price by Number of Stops")
plt.tight_layout()
plt.savefig("stops_prices.png")
plt.show()

# 3. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Flight Price Distribution")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()

print("All graphs saved successfully!")
