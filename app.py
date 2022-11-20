import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #final_rf.predict_proba([[41,81,100000]])[0][1]

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Clients probability of developing a heart disease is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))]) #predict probaility and not class

    output = prediction[0] #as percentage
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)