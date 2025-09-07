import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re
from sklearn import preprocessing as sk
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

file= r"C:\Users\KOREKA\OneDrive\Masaüstü\Python Projeler\1.Online Retail RFM\online_retail_II.csv"
online_retail= pd.read_csv(file)

# Adding Region Labels
states_pt= r"C:\Users\KOREKA\OneDrive\Masaüstü\region_table.xlsx"
states = pd.read_excel(states_pt)
online_retail = pd.merge(online_retail, states, on="Country", how="left")

# Determining NA amount and filling with related subregions mean or median
numeric_columns = online_retail.select_dtypes(include=[np.number]).columns.tolist()
for x in numeric_columns:
    skewness = stats.skew(online_retail[x].dropna())
    if abs(skewness) > 1:
        online_retail[x] = online_retail[x].fillna(
            online_retail.groupby("Subregion")[x].transform("median")
        )
    else:
        online_retail[x] = online_retail[x].fillna(
            online_retail.groupby("Subregion")[x].transform("mean")
        )

# We have enough rows thus we can drop na rows.
online_retail.dropna(inplace=True)

# Remove negative values in Quantity and Price columns
online_retail = online_retail[online_retail["Price"] >= 0]
online_retail = online_retail[online_retail["Quantity"] >= 0]

# Creating RFM labels
online_retail["TotalPrice"] = online_retail["Quantity"] * online_retail["Price"]

# snapshot_date: 1 day later
online_retail["InvoiceDate"] = pd.to_datetime(online_retail["InvoiceDate"])
snapshot_date = online_retail["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = online_retail.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
})

# Rename columns
rfm.rename(columns={
    "InvoiceDate": "Recency",
    "Invoice": "Frequency",
    "TotalPrice": "Monetary"
}, inplace=True)

#Finding Q1 Q3 and IQR so that we can apply a new labels.

rfm["Recency_qcut"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
rfm["Frequency_qcut"] = pd.qcut(rfm["Recency"], 4, labels=[1, 2, 3, 4])
rfm["Monetary_qcut"] = pd.qcut(rfm["Recency"], 4, labels=[1, 2, 3, 4])

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Use RFM table
X = rfm[["Recency", "Frequency", "Monetary"]].copy()

# Invert Recency (small days = better, so we make it negative)
X["Recency"] = -X["Recency"]

# Create target y (dummy example: 1 = purchase in next 90 days, 0 = no purchase)
# In real life: use InvoiceDate to check future purchase
np.random.seed(42)
y = np.random.randint(0, 2, size=len(X))

# Standardize data (mean = 0, std = 1) so coefficients are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression model (predicts probability of purchase)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_scaled, y)

# Get coefficients (absolute values)  importance of each feature
coef = np.abs(logreg.coef_[0])

# Convert to percentage weights (sum = 100%)
weights = 100 * coef / coef.sum()  # 35.589334,  8.712934 55.697731
# Convert qcut columns to int
rfm["Recency_qcut"]   = rfm["Recency_qcut"].astype(int)
rfm["Frequency_qcut"] = rfm["Frequency_qcut"].astype(int)
rfm["Monetary_qcut"]  = rfm["Monetary_qcut"].astype(int)

# Create RFM score (multiplication)
rfm["RFM_Score"] = rfm["Recency_qcut"] * rfm["Frequency_qcut"] * rfm["Monetary_qcut"]
rfm["Customer Segment"] = pd.cut(
    rfm["RFM_Score"],
    bins=4,  # 4 groups
    labels=["At Risk", "Regular", "Loyal", "VIP"]  # from lowest to highest
)
#Dropping unrelated columns and jumping into POWER BI
rfm.drop(["Recency_qcut","Frequency_qcut","Monetary_qcut","RFM_Score"], axis=1, inplace=True)
rfm.to_excel(r"C:\Users\KOREKA\OneDrive\Masaüstü\Python Projeler\1.Online Retail RFM\rfm_table.xlsx")
online_retail.to_excel(r"C:\Users\KOREKA\OneDrive\Masaüstü\Python Projeler\1.Online Retail RFM\online_retail_cleaned.xlsx")



dataset=rfm









