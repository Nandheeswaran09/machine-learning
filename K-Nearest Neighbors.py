import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  # Import KNN Classifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Load your review dataset (replace with your actual data)
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1=pd.read_csv('Pos_Neg_Final.csv')

df=df.iloc[:40000]
df1=df1.iloc[:40000]

df['Summary_Tamil'].fillna('', inplace=True)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df['Summary_Tamil'], df1['Sentiment_1'], test_size=0.2, random_state=42
)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Use KNN Classifier here

# Train the model
knn_classifier.fit(X_train_tfidf, y_train)

# Predictions on validation set
y_val_pred = knn_classifier.predict(X_val_tfidf)

# Evaluate model performance
print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

#author nandheeswaran,hemanth,dharshan 
# Heatmap of Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Binarize the output
y_val_bin = label_binarize(y_val, classes=[0, 1, 2])  # replace with your actual classes
n_classes = y_val_bin.shape[1]

# Compute AUC for each class
roc_auc_scores = dict()  # renamed variable
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_val_pred)
    roc_auc_scores[i] = roc_auc_score(y_val_bin[:, i], y_val_pred)  # use roc_auc_score function here
    print(f"AUC for class {i}: {roc_auc_scores[i]}")  # print the roc_auc_scores

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_val_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

#author nandheeswaran,hemanth,dharshan 
# Import necessary libraries
import numpy as np
from sklearn.metrics import accuracy_score

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Use KNN Classifier here

# Placeholder for accuracies
train_accuracies = []
test_accuracies = []

# Maximum depth range
max_depth_range = list(range(1, 6))

for max_depth in max_depth_range:
    knn_classifier = KNeighborsClassifier(n_neighbors=max_depth)  # Use KNN Classifier here
    knn_classifier.fit(X_train_tfidf, y_train)
    
    # Predictions on training set
    y_train_pred = knn_classifier.predict(X_train_tfidf)
    
    # Predictions on validation set
    y_val_pred = knn_classifier.predict(X_val_tfidf)
    
    # Compute accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Append to accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

#author nandheeswaran,hemanth,dharshan 
# Plot training and testing accuracies
plt.figure(figsize=(10, 5))
plt.plot(max_depth_range, train_accuracies, label='Train')
plt.plot(max_depth_range, test_accuracies, label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy for KNN')
plt.legend()
plt.show()
