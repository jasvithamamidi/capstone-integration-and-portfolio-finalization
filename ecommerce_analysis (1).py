import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Set style
sns.set(style="whitegrid")

# Load dataset (your actual file name)
df = pd.read_csv("E-commerce Customer Behavior - Sheet1.csv")

# View data
print("\nFirst 5 Rows:\n", df.head())

# Info and summary
print("\nDataset Info:\n")
df.info()

print("\nSummary Statistics:\n", df.describe())

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

print("\nColumn Names:\n", df.columns)

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Value counts
print("\nMembership Count:\n", df["Membership_Type"].value_counts())
print("\nSatisfaction Count:\n", df["Satisfaction_Level"].value_counts())

# -------------------------------
# 📊 ANALYSIS & VISUALIZATION
# -------------------------------

# 1. Membership vs Spend
membership_spend = df.groupby("Membership_Type")["Total_Spend"].mean()
print("\nAverage Spend by Membership:\n", membership_spend)

membership_spend.plot(kind='bar', color=['gold','silver','brown'])
plt.title("Average Spend by Membership Type")
plt.xlabel("Membership Type")
plt.ylabel("Average Spend")
plt.show()

# 2. Discount vs Spend
discount_spend = df.groupby("Discount_Applied")["Total_Spend"].mean()
print("\nAverage Spend by Discount:\n", discount_spend)

discount_spend.plot(kind='bar', color=['red','green'])
plt.title("Discount vs Average Spend")
plt.xlabel("Discount Applied")
plt.ylabel("Average Spend")
plt.show()

# 3. Satisfaction vs Spend
sns.boxplot(x="Satisfaction_Level", y="Total_Spend", data=df)
plt.title("Satisfaction vs Spend")
plt.show()

# 4. Age Distribution
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 5. City vs Spend
city_spend = df.groupby("City")["Total_Spend"].mean()
print("\nAverage Spend by City:\n", city_spend)

city_spend.plot(kind='bar')
plt.title("City vs Average Spend")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 🔬 HYPOTHESIS TESTING
# -------------------------------

# Split data
discount_true = df[df["Discount_Applied"] == True]["Total_Spend"]
discount_false = df[df["Discount_Applied"] == False]["Total_Spend"]

# Perform T-test
t_stat, p_value = ttest_ind(discount_true, discount_false)

print("\nT-statistic:", t_stat)
print("P-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("\nConclusion: Reject Null Hypothesis -> Discount affects spending")
else:
    print("\nConclusion: Fail to Reject Null Hypothesis -> No significant effect")

# -------------------------------
# 📉 RETENTION ANALYSIS
# -------------------------------

inactive_customers = df[df["Days_Since_Last_Purchase"] > 30]
print("\nInactive Customers Count:", len(inactive_customers))