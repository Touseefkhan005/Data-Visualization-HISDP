# MODEL-TRAINING-HISDP
Model training on Diabetes Dataset.

1. Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pandas: For data manipulation and analysis, especially for working with structured data (like .csv files).
numpy: A library for numerical operations, but in this specific code, it's not used directly.
matplotlib.pyplot: A plotting library used to visualize the data.
2. Loading the Dataset

df = pd.read_csv("diabetes.csv")
df
pd.read_csv("diabetes.csv"): Reads the CSV file containing diabetes data into a DataFrame.
df: A variable that stores the DataFrame object, where df stands for DataFrame.
df: Simply displays the dataset.
3. Checking for Missing Values

df.isnull().sum()
This checks for missing values in each column of the dataset by summing up the number of null or missing entries in each column.
4. Getting the Shape of the Dataset

df.shape
Returns the shape of the dataset, i.e., the number of rows and columns. This helps in understanding the size of the data.
5. Identifying Unique Values in Outcome

df['Outcome'].unique()
Displays the unique values in the Outcome column. The Outcome column likely represents whether or not a person has diabetes (e.g., 1 for diabetes, 0 for non-diabetes).
6. Summary Statistics

df.describe()
Provides summary statistics of the numerical columns, including measures like mean, standard deviation, minimum, maximum, and percentiles.
7. Visualizing Data with Boxplots

plt.boxplot(data=df, x=df['Pregnancies'])
plt.show()
plt.boxplot(data=df, x=df['Glucose'])
plt.show()
Creates boxplots for the Pregnancies and Glucose columns.
Boxplots show the distribution of the data, highlighting median values, quartiles, and potential outliers.
plt.show(): Displays the plot.
8. Pie Chart for Outcome Proportions

plt.pie(df.Outcome.value_counts(), labels=['Diabetes', 'Not Diabetes'], autopct='%.f')
plt.title('Outcome proportionality')
plt.show()
Creates a pie chart showing the proportion of people with and without diabetes in the dataset.
df.Outcome.value_counts(): Counts the frequency of each unique value in the Outcome column.
labels=['Diabetes', 'Not Diabetes']: Labels the two parts of the pie chart.
autopct='%.f': Shows percentages with no decimal places.
plt.title(): Adds a title to the plot.
plt.show(): Displays the pie chart.
9. Defining Features (X) and Target (y)

X = df[['Pregnancies', 'Glucose', 'BloodPressure']]
X
y = df['Outcome']
y
X: Selects the features for model training (here, Pregnancies, Glucose, and BloodPressure are selected as independent variables).
y: The target variable Outcome which represents whether a person has diabetes or not (1 for diabetes, 0 for not).
10. Splitting the Data into Training and Test Sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
X_train
X_test
train_test_split: Splits the dataset into training and test sets.
X_train, y_train: Features and target for training.
X_test, y_test: Features and target for testing.
test_size=0.20: 20% of the data will be used for testing, and 80% for training.
random_state=5: Ensures reproducibility of the split.
11. Training the Logistic Regression Model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
model.coef_
LogisticRegression(): Initializes a Logistic Regression model.
max_iter=2000: Sets the maximum number of iterations to 2000 (to ensure convergence).
model.fit(X_train, y_train): Trains the Logistic Regression model using the training data.
model.coef_: Displays the coefficients (weights) of the trained model.
12. Making Predictions on the Test Data

predictions = model.predict(X_test)
predictions
model.predict(X_test): Uses the trained model to make predictions on the test set.
predictions: Stores the predicted values (whether each test instance is predicted as diabetic or not).
13. Confusion Matrix and Heatmap

import seaborn as sns
from sklearn.metrics import confusion_matrix
cof_matrix = confusion_matrix(y_test, predictions)
plt.figure(dpi=100)
sns.heatmap(cof_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
confusion_matrix(y_test, predictions): Computes the confusion matrix, which shows how well the model predicted the test data:
True Positives (correctly predicted diabetes).
True Negatives (correctly predicted non-diabetes).
False Positives (incorrectly predicted diabetes).
False Negatives (incorrectly predicted non-diabetes).
sns.heatmap(): Visualizes the confusion matrix as a heatmap.
annot=True: Annotates the heatmap with the actual numbers.
fmt='d': Formats the annotations as integers.
cmap='Blues': Specifies the color palette for the heatmap.
plt.xlabel(), plt.ylabel(), plt.title(): Label the axes and add a title.
14. Displaying Actual vs Predicted Outcomes

Result = pd.DataFrame({'Actual': y_test, 'Predictions': predictions})
Result
Creates a DataFrame called Result that compares the actual test values (y_test) with the model's predictions (predictions).
