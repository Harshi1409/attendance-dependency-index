# ============================================
# Attendance Dependency Index Project
# Course: INT375 - Data Science Toolbox
# Author: Harshita Bansal
# ============================================


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv("georgia_attendance_2024.csv")

cols = [
    'CHRONIC_ABSENT_PERC_ALL',
    'STUDENT_COUNT_ALL',
    'FIVE_OR_FEWER_PERCENT_ALL',
    'SIX_TO_FIFTEEN_PERCENT_ALL',
    'OVER_15_PERCENT_ALL'
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df.dropna(inplace=True)


df['Attendance'] = 100 - df['CHRONIC_ABSENT_PERC_ALL']


df['Performance'] = (
    df['FIVE_OR_FEWER_PERCENT_ALL'] * 3 +
    df['SIX_TO_FIFTEEN_PERCENT_ALL'] * 2 +
    df['OVER_15_PERCENT_ALL'] * 1
)


df = df[['Attendance', 'Performance']]
df = df[df['Attendance'] > 0]


adi = df['Attendance'].corr(df['Performance'])
print("ADI:", adi)


corr, p_value = pearsonr(df['Attendance'], df['Performance'])
print("Correlation:", corr)
print("P-value:", p_value)


sns.scatterplot(x='Attendance', y='Performance', data=df)
plt.title("Attendance vs Performance")
plt.show()


sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


X = df[['Attendance']]
y = df['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))


sns.regplot(x='Attendance', y='Performance', data=df)
plt.title("Regression Analysis")
plt.show()


sns.boxplot(x=df['Attendance'])
plt.title("Attendance Distribution")
plt.show()


df['Attendance_Level'] = pd.cut(df['Attendance'],
                               bins=[0, 50, 75, 100],
                               labels=['Low', 'Medium', 'High'])

sns.boxplot(x='Attendance_Level', y='Performance', data=df)
plt.title("Performance by Attendance Level")
plt.show()




