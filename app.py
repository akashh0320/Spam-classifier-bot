from flask import Flask, render_template, request
import pickle


model = pickle.load(open("model/spam_model.pkl", "rb"))
cv = pickle.load(open("model/vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.form["message"]
    msg_vec = cv.transform([msg])
    result = model.predict(msg_vec)

    output = "Spam 🚫" if result[0] == 1 else "Ham ✅"
    return render_template("index.html", prediction=output)

if __name__ == "__main__":
    app.run(debug=True)