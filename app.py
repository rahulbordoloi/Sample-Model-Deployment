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
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 1:
        text = "Yes, they'll purchase the item "
    elif output == 0:
        text = "No, they'll not purchase the item "

    return render_template('index.html', prediction_text= text + 'i.e. output is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)