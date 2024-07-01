import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.svm import SVC

# train the datasets for tamil datasets
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

# Initialize SVC classifier
svc_classifier = SVC(
    kernel='linear',  # For multiclass classification
    probability=True,  # To enable predict_proba
    random_state=42
)

#author nandheeswaran,hemanth,dharshan
# Train the model
svc_classifier.fit(X_train_tfidf, y_train)

# Predictions on validation set
y_val_pred = svc_classifier.predict(X_val_tfidf)
y_val_pred_proba = svc_classifier.predict_proba(X_val_tfidf)

# Evaluate model performance
print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

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
roc_auc_values = dict()  # renamed variable
for i in range(n_classes):
    roc_auc_values[i] = roc_auc_score(y_val_bin[:, i], y_val_pred_proba[:, i])
    print(f"AUC for class {i}: {roc_auc_values[i]}")  # use the renamed variable

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_val_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#author nandheeswaran,hemanth,dharshan
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

from sklearn.metrics import log_loss

# Initialize lists to store losses
train_losses = []
val_losses = []

# Define the number of iterations
n_iterations = 10

for i in range(n_iterations):
    # Train the model incrementally
    svc_classifier.fit(X_train_tfidf[i::n_iterations], y_train[i::n_iterations])

    # Predict probabilities
    y_train_pred_proba = svc_classifier.predict_proba(X_train_tfidf)
    y_val_pred_proba = svc_classifier.predict_proba(X_val_tfidf)

    # Compute log loss
    train_loss = log_loss(y_train, y_train_pred_proba)
    val_loss = log_loss(y_val, y_val_pred_proba)

    # Store the losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
#author nandheeswaran,hemanth,dharshan
# Plot the losses
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.title('SVC Log Loss')
plt.show()



#author nandheeswaran,hemanth,dharshan
