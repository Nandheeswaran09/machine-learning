import pandas as pd
import xgboost as xgb
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
# Assume you have a CSV file with columns: 'review_text' and 'sentiment'
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1=pd.read_csv('Pos_Neg_Final.csv')

df=df.iloc[:40000]
df1=df1.iloc[:40000]

df['Summary_Tamil'].fillna('', inplace=True)
# Preprocessing: Clean text, tokenize, and convert to lowercase
# You can use NLTK or spaCy for tokenization and other preprocessing steps

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df['Summary_Tamil'], df1['Sentiment_1'], test_size=0.2, random_state=42
)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',  # For multiclass classification
    num_class=len(df1['Sentiment'].unique()),  # Number of sentiment classes
    max_depth=5,  # Adjust hyperparameters as needed
    learning_rate=0.1,
    n_estimators=100
)

# Train the model
eval_set = [(X_train_tfidf, y_train), (X_val_tfidf, y_val)]
xgb_classifier.fit(X_train_tfidf, y_train, eval_metric=["mlogloss"], eval_set=eval_set, verbose=False)

# Predictions on validation set
y_val_pred = xgb_classifier.predict(X_val_tfidf)
y_val_pred_proba = xgb_classifier.predict_proba(X_val_tfidf)

# Evaluate model performance
print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

#train and test the datasets for tamil
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
#train and test the datasets for tamil
for i in range(n_classes):
    auc = roc_auc_score(y_val_bin[:, i], y_val_pred_proba[:, i])
    print(f"AUC for class {i}: {auc}")

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Binarize the output
y_val_bin = label_binarize(y_val, classes=[0, 1, 2])  # replace with your actual classes
n_classes = y_val_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_val_pred_proba[:, i])
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

#author dharshan,hemanth,nandheeswaran
# Training and Testing Loss
results = xgb_classifier.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()


#author dharshan,hemanth,nandheeswaran
