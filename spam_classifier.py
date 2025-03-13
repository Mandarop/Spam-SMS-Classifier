#Step 1 - 

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset with proper column selection
df = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0, 1], names=["label", "message"], skiprows=1)

# Display first 5 rows
print(df.head())

# Show dataset shape (rows, columns)
print("Dataset shape:", df.shape)




# This code loads the SMS Spam dataset from 'spam.csv' using pandas.
# It selects only the first two columns (label and message) using 'usecols=[0, 1]'.
# The columns are renamed to 'label' and 'message' for better readability.
# 'skiprows=1' skips the first row if it contains incorrect headers.
# Finally, it prints the first 5 rows and dataset shape to verify the data structure.



#Step 2 - 

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Remove any missing values (if present)
df.dropna(inplace=True)

# Encode labels: 'ham' → 0, 'spam' → 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Verify changes
print("\nUpdated dataset:\n", df.head())

# This code checks for missing values in the dataset using 'df.isnull().sum()'.
# If any missing values are found, 'df.dropna(inplace=True)' removes them.
# It then encodes labels: 'ham' is mapped to 0 and 'spam' to 1 using 'map()'.
# Finally, it prints the updated dataset to verify the changes.


#Step 3 - 

# Download stopwords & wordnet if not already available
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]  # Remove stopwords & lemmatize
    return ' '.join(words)  # Convert list of words back to string

# Apply text preprocessing
df["message"] = df["message"].apply(preprocess_text)

# Display cleaned text
print("\nSample preprocessed messages:\n", df["message"].head())

# This code preprocesses the SMS text messages for machine learning.
# It converts text to lowercase, removes special characters, numbers, and extra spaces.
# It tokenizes the text into words and removes common stopwords (e.g., "the", "is").
# Lemmatization reduces words to their base form (e.g., "running" → "run").
# The cleaned text is stored back in the 'message' column.


# Save the cleaned dataset
df.to_csv("spam_cleaned.csv", index=False)

print("Cleaned dataset saved as 'spam_cleaned.csv'")

# This code saves the preprocessed dataset to 'spam_cleaned.csv'.
# It ensures that the cleaned text is stored for future use.
# 'index=False' prevents Pandas from writing row indices to the file.



#Step 4 -

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words

# Convert text messages into numerical TF-IDF vectors
X = tfidf_vectorizer.fit_transform(df["message"]).toarray()

# Extract labels (y)
y = df["label"].values

# Print shape of transformed data
print("TF-IDF matrix shape:", X.shape)


# This code converts SMS messages into numerical form using TF-IDF.
# 'TfidfVectorizer' transforms text into a matrix of TF-IDF features.
# 'max_features=5000' limits the vocabulary to the top 5000 words.
# The transformed data (X) is stored as an array, while labels (y) are extracted separately.
# Finally, it prints the shape of the TF-IDF matrix to verify the transformation.


# Step 5 -

from sklearn.model_selection import train_test_split

# Split data while maintaining spam-to-ham ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Print class distribution
print("Training set - Spam:", sum(y_train), "Ham:", len(y_train) - sum(y_train))
print("Testing set - Spam:", sum(y_test), "Ham:", len(y_test) - sum(y_test))

# This code splits data into training (80%) and testing (20%) sets while maintaining class balance.
# 'stratify=y' ensures both spam and ham are split in the same proportion.
# It prints the count of spam & ham messages in training and testing sets for verification.

#Step 6 -

from sklearn.naive_bayes import MultinomialNB
import joblib

# Initialize and train the Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Print confirmation
print("Model training completed!")


joblib.dump(nb_model, "spam_classifier_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")


# This code trains a Naïve Bayes classifier for spam detection.
# 'MultinomialNB()' is ideal for text-based classification problems.
# 'fit()' trains the model using TF-IDF features (X_train) and labels (y_train).

# This code saves the trained Naïve Bayes model to a file using joblib.
# 'dump()' stores the model in 'spam_classifier_model.pkl' for future use.
# This prevents retraining every time the script runs.


# Step 7 - 

from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = nb_model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Print classification report (Precision, Recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))
