import pandas as pd

RAW_PATH = "data/online_retail_II.xlsx"


def load_and_clean(path=RAW_PATH):

    # Load both sheets
    df1 = pd.read_excel(path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(path, sheet_name="Year 2010-2011")

    # Combine them
    df = pd.concat([df1, df2], ignore_index=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "")

    # Mark cancelled invoices
    df["IsCancelled"] = df["Invoice"].astype(str).str.startswith("C")

    # Remove unwanted rows
    junk_codes = ["POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK"]

    df = df[
        df["CustomerID"].notna() &
        (df["Price"] > 0) &
        ~df["StockCode"].astype(str).str.upper().isin(junk_codes) &
        ((df["Quantity"] > 0) | df["IsCancelled"])
    ]

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix data types
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)  

    # Calculate transaction value
    df["LineTotal"] = df["Quantity"] * df["Price"]

    return df.reset_index(drop=True)

# Run
df = load_and_clean()

print(f"Rows: {len(df):,}")
print(f"Customers: {df['CustomerID'].nunique():,}")
print(f"Date range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")

# Save
df.to_parquet("data/cleaned_transactions.parquet", index=False)