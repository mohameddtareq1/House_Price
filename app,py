# **✅IMPORT LIBRARIES:**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# **✅READ DATASET:**

df = pd.read_csv('house-prices.csv')

# **✅SHAPE OF DATASET:**

df.shape

# **✅DATA INFORMATION:**

df.head()

df.tail()

#names of columns
df.columns

# **✅CHECK NULLS:**

df.info()

df.isnull().sum()

# **✅CHECK OBJECTS:**

df["Brick"].unique()

df['Brick'].replace({'No': 0, 'Yes': 1}, inplace=True)

df["Neighborhood"].unique()

df['Neighborhood'].replace({'East': 1, 'North': 2, 'West': 3}, inplace=True)

df.info()

# **✅CHECK OUTLIERS:**

columns = ['Price', 'SqFt']

for i in columns:
    q1 = np.percentile(df[i], 25)
    q3 = np.percentile(df[i], 75)
    norm_range = (q3 - q1) * 1.5

    lower_outliers = df[df[i] < (q1 - norm_range)]

    upper_outliers = df[df[i] > (q3 + norm_range)]

    outliers = len(lower_outliers) + len(upper_outliers)

    print(f"The number of outliers in {i}: {outliers}")

    df[i] = np.where(df[i] < (q1 - norm_range), q1 - norm_range, df[i])
    df[i] = np.where(df[i] > (q3 + norm_range), q3 + norm_range, df[i])

columns = ['Price', 'SqFt']

for i in columns:
    q1 = np.percentile(df[i], 25)
    q3 = np.percentile(df[i], 75)
    norm_range = (q3 - q1) * 1.5

    lower_outliers = df[df[i] < (q1 - norm_range)]

    upper_outliers = df[df[i] > (q3 + norm_range)]

    outliers = len(lower_outliers) + len(upper_outliers)

    print(f"The number of outliers in {i}: {outliers}")

df.head(15)

# **✅CHECK DUPLICATES**

df.duplicated().sum()

# **✅CHECK CORRELATION:**

df.corr()

fig = plt.subplots(figsize=(20, 10))
sns.heatmap(df.corr(),annot = True)

high_corr = []
low_corr = []
bad_corr = []
for col in df.columns:
  relation = df['Price'].corr(df[col])
  if(relation > 0):
    if relation >= 0.7 and relation <= 1 :
      
      high_corr.append(col)
    elif relation >= 0.4 and relation < 0.7 :
      
      low_corr.append(col)
    else: bad_corr.append(col)
  else:
    if relation <= -0.7 and relation > -1 :
      
      high_corr.append(col)
    elif relation <= -0.4 and relation > -0.7 :
      
      low_corr.append(col)
    else: bad_corr.append(col)

print(f"the high corr are {high_corr}")
print(f"the low corr are {low_corr}")
print(f"the bad corr are {bad_corr}")

# **✅SCIKIT LEARN:**

from sklearn.linear_model import LinearRegression

x = df.drop(columns=['Price'])
y = df['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5, random_state = 0)

linear = LinearRegression()

linear.fit(x_train, y_train)

y_pred_train=linear.predict(x_train)

from sklearn import metrics
print("MSE:",metrics.mean_squared_error(y_pred_train,y_train))
print("MAE:",metrics.mean_absolute_error(y_pred_train,y_train))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_pred_train,y_train)))
print("r2_score:",metrics.r2_score(y_pred_train,y_train))

y_pred = linear.predict(x_test)

print("MSE:",metrics.mean_squared_error(y_pred,y_test))
print("MAE:",metrics.mean_absolute_error(y_pred,y_test))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
print("r2_score:",metrics.r2_score(y_pred,y_test))
