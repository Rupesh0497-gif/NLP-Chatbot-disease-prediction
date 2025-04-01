import pandas as pd

try:
    df = pd.read_csv("./dataset.csv")
    print("✅ File loaded successfully!")
    print(df.head())  # Show first 5 rows
except FileNotFoundError:
    print("❌ File not found! Please check the file path.")
except Exception as e:
    print(f"❌ Error: {e}")
