import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load your labeled dataset (replace with your actual data)
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df=df.iloc[:40000]
df1 = pd.read_csv('Pos_Neg_Final.csv')
df1=df1.iloc[:40000]

# Fill any missing values in the 'Summary_Tamil' column with an empty string
df['Summary_Tamil'].fillna('', inplace=True)


#train the datasets for tamil
# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Extract features from the text data
X = tfidf_vectorizer.fit_transform(df['Summary_Tamil'])

# Sentiment scores (positive/negative) from your second dataset
sentiment_score = df1['Sentiment_1']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiment_score, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict sentiment scores on the train and test data
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Evaluate model performance (Mean Squared Error)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Train Mean Squared Error: {mse_train:.2f}")
print(f"Test Mean Squared Error: {mse_test:.2f}")

# Print classification report
print(classification_report(y_test, y_test_pred))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# Binarize the output
y = label_binarize(sentiment_score, classes=[0, 1, 2])
n_classes = y.shape[1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


#author hemanth,nandheeswaran,dharshan
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
plt.title('Receiver Operating Characteristic to Multi-Class')
plt.legend(loc="lower right")
plt.show()

from sklearn.preprocessing import LabelBinarizer

# Binarize the output
lb = LabelBinarizer()
y_test_lb = lb.fit_transform(y_test)
y_test_pred_lb = lb.transform(y_test_pred)

# Compute confusion matrix for each class
cm = {}
for i in range(n_classes):
    cm[i] = confusion_matrix(y_test_lb[:, i], y_test_pred_lb[:, i])

# Plot confusion matrix as a heatmap for each class
for i in range(n_classes):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm[i], annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix for Class {i}')
    plt.show()

#author hemanth,nandheeswaran,dharshan
# Placeholder for loss graph
# Note: RandomForestClassifier does not support tracking loss over epochs
train_loss = [] # replace with actual train loss if available
test_loss = [] # replace with actual test loss if available

# Create count of the number of epochs
epoch_count = range(1, len(train_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


#author hemanth,nandheeswaran,dharshan
