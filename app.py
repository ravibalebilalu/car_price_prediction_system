from flask import Flask
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

encoder = pickle.load(open("./pickle_models/encoder.pkl","rb"))
scaler =  pickle.load(open("./pickle_models/scaler.pkl","rb"))
model = pickle.load(open("./pickle_models/model.pkl","rb"))

def prediction():
    data = ['Dodge' ,'other', 2017, 'premium unleaded (recommended)', 495.0, 8.0,'MANUAL', 'rear wheel drive' ,2.0 ,'Large' ,'Coupe', 21, 13 ,1851 ,1 ,1 ,0 ,0, 0, 0 ,1 ,0 ,0, 0]

    cat_values,num_val = [],[]

    for val in data:
        if isinstance(val,str):
            cat_values.append(val)

        else:
            num_val.append(val)

    cat = encoder.transform(np.array(cat_values).reshape(1,-1))
    num = np.array(num_val).reshape(1,-1)
    combiined = np.hstack((cat,num))
    scaled = scaler.transform(combiined)

app = Flask(__name__)
 
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=['POST'])
def predict():
    data =  [float(x) for x in request.form.values()]
    
    return render_template("home.html" )

if __name__ == "__main__":
    app.run(debug=True)