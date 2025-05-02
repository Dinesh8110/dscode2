import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,mean_squared_error,root_mean_squared_error

df = pd.read_csv('titanic_train.csv')
df = df.drop(['Name', 'Ticket','Cabin'], axis=1)

#fill missing values
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

#encode categorical values
df = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)
df = df.dropna()

x = df.drop('Survived',axis=1)
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#Linear Regression 

linReg = LinearRegression()
linReg.fit(x_train,y_train)
y_pred = linReg.predict(x_test)
y_pred_class = np.where(y_pred>0.5,1,0)

print(mean_squared_error(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))
print(classification_report(y_test,y_pred_class))


#Ridge Regression
# ridge = Ridge()
# ridge.fit(X_train, y_train)
# y_pred = ridge.predict(X_test)
# y_pred_class = np.where(y_pred > 0.5, 1, 0)


# #logistic Regression
# model = LogisticRegression(max_iter=1000)
# model.fit(x_train,y_train)

# y_pred = model.predict(x_test)

# print(accuracy_score(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# #Random ForestClassifier
# modelRF = RandomForestClassifier()
# modelRF.fit(x_train,y_train)
# y_pred1 = modelRF.predict(x_test)

# print(accuracy_score(y_test,y_pred1))
# print(confusion_matrix(y_test,y_pred1))
# print(classification_report(y_test,y_pred1))


# #knn
# modelknn = KNeighborsClassifier(n_neighbors=5)
# modelknn.fit(x_train,y_train)
# y_pred2 = modelknn.predict(x_test)

# print(accuracy_score(y_test,y_pred2))
# print(confusion_matrix(y_test,y_pred2))
# print(classification_report(y_test,y_pred2))


# #decision Tree
# dt = DecisionTreeClassifier()
# dt.fit(x_train, y_train)
# y_pred3 = dt.predict(x_test)

# print(accuracy_score(y_test,y_pred3))
# print(confusion_matrix(y_test,y_pred3))
# print(classification_report(y_test,y_pred3))


#Covariance and correlation analysis:
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("/bot_detection_data.csv")

# # Convert 'Verified' column to numeric (True -> 1, False -> 0)
# df['Verified'] = df['Verified'].astype(int)

# # Select only numeric columns for analysis
# num_cols = ['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Bot Label']
# df_numeric = df[num_cols]

# # Save cleaned dataset
# df_numeric.to_csv("/bot_detection_cleaned.csv", index=False)

# # Compute correlation and covariance matrices
# corr_matrix, cov_matrix = df_numeric.corr(), df_numeric.cov()

# # Plot correlation heatmap
# plt.figure(figsize=(16, 6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap")
# plt.show()

# # Plot covariance heatmap
# plt.figure(figsize=(16, 6))
# sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Covariance Heatmap")
# plt.show()

# # Violin plot for 'Bot Label' vs 'Follower Count'
# plt.figure(figsize=(10, 5))
# sns.violinplot(x="Bot Label", y="Follower Count", data=df_numeric)
# plt.title("Violin Plot: Follower Count vs Bot Label")
# plt.show()
