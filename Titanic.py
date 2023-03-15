import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('titanic.csv')

# Check for missing values
df.isnull().sum()

# Fill in missing values in the 'Age' column with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

import matplotlib.pyplot as plt

# Create a histogram of the ages
plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Create a bar chart of passenger class
class_counts = df['Pclass'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Create a scatterplot of age vs. fare
plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Create a stacked bar chart of survival rates
survival_counts = df.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count().unstack()
survival_counts.plot(kind='bar', stacked=True)
plt.xlabel('Sex, Passenger Class')
plt.ylabel('Count')
plt.legend(('Did not survive', 'Survived'))
plt.show()

