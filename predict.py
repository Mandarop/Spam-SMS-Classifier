import joblib

# Load the trained model and vectorizer
nb_model = joblib.load("spam_classifier_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Example: Predict if a message is spam or ham
sample_message = ["Congratulations! You have won a lottery. Claim now."]
sample_vector = tfidf_vectorizer.transform(sample_message).toarray()

prediction = nb_model.predict(sample_vector)

# Print result
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")



# This code loads the trained model from 'spam_classifier_model.pkl'.
# It then takes a sample message, converts it using TF-IDF, and predicts if it's spam or ham.
