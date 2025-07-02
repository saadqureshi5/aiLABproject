import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import pickle
import joblib

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

def text_preprocess(text):
    """
    Clean the text by removing punctuation and stopwords
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

def load_model_and_vectorizer():
    """
    Load the trained model and TF-IDF vectorizer
    """
    try:
        # Load the SVM model
        model = joblib.load('spam_model.pkl')
        # Load the TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, tfidf_vectorizer
    except FileNotFoundError:
        print("Error: Model or vectorizer file not found. Make sure to train the model first.")
        return None, None

def test_email(email_text, model, tfidf_vectorizer):
    """
    Test if an email is spam or not
    """
    if model is None or tfidf_vectorizer is None:
        print("Cannot test email: Model or vectorizer not loaded.")
        return
    
    # Preprocess the sample email
    processed_email = text_preprocess(email_text)
    
    # Transform the sample email using the TF-IDF vectorizer
    email_vector = tfidf_vectorizer.transform([processed_email])
    
    # Use the trained model to predict
    prediction = model.predict(email_vector)
    
    if prediction[0] == 'spam':
        return "SPAM"
    else:
        return "NOT SPAM"

if __name__ == "__main__":
    # Load model and vectorizer
    model, tfidf_vectorizer = load_model_and_vectorizer()
    
    if model is not None and tfidf_vectorizer is not None:
        # Test with sample emails
        samples = [
            """07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile +
            free camcorder. Please call now 08000930705 for delivery tomorrow""",
            
            "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?",
            
            "Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child and"
        ]
        
        for i, sample in enumerate(samples):
            result = test_email(sample, model, tfidf_vectorizer)
            print(f"Sample {i+1} is classified as: {result}")
            print(f"Email text: {sample[:60]}...\n")
        
        # Interactive testing
        print("\nEnter your own email text to test (or type 'exit' to quit):")
        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                break
            result = test_email(user_input, model, tfidf_vectorizer)
            print(f"Classification: {result}\n")
    else:
        print("Please run the training notebook first to generate the model files.") 