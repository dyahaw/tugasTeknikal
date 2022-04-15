from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template("jenisKelamin.html")
    elif request.method == 'POST':
        gender = float(request.form['gender'])
        height = float(request.form['height'])
        input_array = [[height, gender]]
        weight_predictor_model = pickle.load(open('model-development/weight_predictor.pkl', 'rb'))
        result = round(weight_predictor_model.predict(input_array)[0], 2)
        return render_template('jenisKelamin.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)