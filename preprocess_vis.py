import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic_train.csv')

#drop irrelevant features
df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

print("Before Handling missign values : ",df.isnull().sum())

#fill the missing values
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

print("After Handling missign values : ",df.isnull().sum())

#descriptive Statistics
print(df['Age'].describe())

#encode categorical values
df['Sex'].map({'male':0,'female':1})
df['Embarked'].map({'S':0,'C':1,'Q':2})

#survived count
sns.countplot(df['Survived'])
plt.title("Survived count")
plt.show()

#survival by Sex
sns.barplot(x=df['Sex'],y=df['Survived'])
plt.title("Survival by Sex")
plt.show()

#survival by Plcass
sns.barplot(x=df['Pclass'],y=df['Survived'])
plt.title("survival by Passenger class")
plt.show()


#Age distibution
sns.histplot(df['Age'],bins=20,kde=True)
plt.title("Age distribution")
plt.show()












#over sampling and under sampling

# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTEENN
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("/bot_detection_data.csv")

# # Features and Target
# X = df[['Retweet Count', 'Mention Count', 'Follower Count']]  # Use numerical columns
# y = df['Bot Label']

# # Check class distribution
# print("Original class distribution:", Counter(y))

# # Oversampling using SMOTEw
# smote = SMOTE(random_state=42)
# X_smote, y_smote = smote.fit_resample(X, y)
# print("After SMOTE:", Counter(y_smote))

# # Undersampling
# undersampler = RandomUnderSampler(random_state=42)
# X_under, y_under = undersampler.fit_resample(X, y)
# print("After Undersampling:", Counter(y_under))

# # Combine SMOTE and Undersampling using SMOTEENN
# smoteenn = SMOTEENN(random_state=42)
# X_smoteenn, y_smoteenn = smoteenn.fit_resample(X, y)
# print("After SMOTEENN:", Counter(y_smoteenn))

# # Visualization of the Resampled Data
# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
# axes[0].bar(Counter(y_smote).keys(), Counter(y_smote).values(), color='skyblue')
# axes[0].set_title("Oversampling (SMOTE)")

# axes[1].bar(Counter(y_under).keys(), Counter(y_under).values(), color='lightgreen')
# axes[1].set_title("Undersampling")

# axes[2].bar(Counter(y_smoteenn).keys(), Counter(y_smoteenn).values(), color='salmon')
# axes[2].set_title("Combination (SMOTEENN)")

# for ax in axes:
#     ax.set_xlabel("Class")
#     ax.set_ylabel("Frequency")

# plt.tight_layout()
# plt.show()