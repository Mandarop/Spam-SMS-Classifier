# 📩 Spam SMS Classifier

A machine learning project to classify SMS messages as **spam** or **ham** (not spam) using **Natural Language Processing (NLP)**.

---

## 🚀 Features
✔️ Preprocessing of text data  
✔️ TF-IDF Vectorization for feature extraction  
✔️ Classification using a trained Machine Learning model  
✔️ Model and vectorizer stored for future predictions  
✔️ Simple script to test predictions  

---

## 🔧 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Mandarop/Spam-SMS-Classifier.git
cd Spam-SMS-Classifier

2️⃣ Set Up a Virtual Environment (Recommended)

python -m venv myenv
source myenv/bin/activate   # macOS/Linux
myenv\Scripts\activate      # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

📂 Project Structure

📁 Spam-SMS-Classifier
│── spam_classifier.py       # Model training and classification
│── predict.py               # Script to test the model
│── spam.csv                 # Raw dataset
│── spam_cleaned.csv         # Preprocessed dataset
│── classifier_model.pkl     # Saved trained model
│── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
│── requirements.txt         # List of dependencies
│── README.md                # Project documentation

🛠 Usage :- 
python predict.py
Run the predict.py script to classify an SMS message You can modify predict.py to test different messages.

📊 Model Details :-
Vectorization: TF-IDF (Term Frequency - Inverse Document Frequency)
Model Used: Logistic Regression / Naive Bayes (whichever was best performing)
Accuracy: Achieved high accuracy on test data


