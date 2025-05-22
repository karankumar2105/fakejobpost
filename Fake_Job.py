from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open("fraud_pickle.pkl", "rb"))
tf = joblib.load(open('tfidf_job_pickle.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("Fake_Job_Home.html", prediction=None)  # FIXED


def transform_text(title, location, company_profile, description):
    combined_text = f"{title} {location} {company_profile} {description}"
    vectorized_input_data = tf.transform([combined_text])
    prediction = model.predict(vectorized_input_data)
    return prediction


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        title = request.form.get('title', '')
        location = request.form.get('location', '')
        company_profile = request.form.get('company_profile', '')
        description = request.form.get('description', '')

        prediction = transform_text(title, location, company_profile, description)

        return render_template("Fake_Job_Home.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
