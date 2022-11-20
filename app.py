import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)

    output = round(prediction[0][1], 4)

    return render_template('index.html', prediction_text='Clients probability of developing a heart disease is {}{}'.format(output*100,"%"))

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