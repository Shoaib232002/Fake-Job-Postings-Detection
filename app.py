from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text Cleaning Function (must match the one used in training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        job_text = request.form['job_text']
        cleaned_text = clean_text(job_text)
        vect_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vect_text)[0]
        result = "❌ Fake Job Posting" if prediction == 1 else "✅ Legitimate Job Posting"
        return render_template('index.html', prediction=result, job_text=job_text)

# Run App
if __name__ == '__main__':
    app.run(debug=True)
