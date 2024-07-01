import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Load your dataset
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1=pd.read_csv('Pos_Neg_Final.csv')
df = df.iloc[:40000]
df1=df1.iloc[:40000]
df['Summary_Tamil'].fillna('', inplace=True)

X = df['Summary_Tamil']
y = df1['Sentiment_1']

# Vectorize text data
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

# Train the MNB model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion matrix with heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Assuming y_test and y_score are your multi-class labels and scores
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # replace with your classes
y_score = clf.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for i in range(3):  # replace with your number of classes
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(3):  # replace with your number of classes
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver Operating Characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
