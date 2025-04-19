from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load("sms_spam_detector.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        result = model.predict(data)[0]
        prediction = "not Spam" if result == 1 else "Spam"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

