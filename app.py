from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    vectorized_news = vectorizer.transform([news])
    prediction = model.predict(vectorized_news)[0]
    result = "Real News 📰 ✅ " if prediction == 1 else "Fake News 🚨❗ "
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
