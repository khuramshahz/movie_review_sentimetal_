from flask import Flask, render_template, request
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
cv = joblib.load('count_vectorizer.pkl')

# Download stopwords
nltk.download('stopwords')
ps = PorterStemmer()

def preprocess_text(text):
    # Clean HTML
    text = re.sub('<.*?>', '', text)
    # Remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove stopwords and stem
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stopwords.words('english')])
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        processed_review = preprocess_text(review)
        vectorized_review = cv.transform([processed_review]).toarray()
        prediction = model.predict(vectorized_review)[0]
        result = "Positive" if prediction == 1 else "Negative"
        return render_template('index.html', result=result, review=review)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)