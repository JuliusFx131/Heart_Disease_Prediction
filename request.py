import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'BMI':41, 'Age':27, 'Income':60000})

print(r.json())