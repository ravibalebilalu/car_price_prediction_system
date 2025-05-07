from flask import Flask
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

encoder = pickle.load(open("./pickle_models/encoder.pkl","rb"))
scaler =  pickle.load(open("./pickle_models/scaler.pkl","rb"))
model = pickle.load(open("./pickle_models/model.pkl","rb"))

def prediction(data):
    cat_values = data[:7]
    num_val = data[7:]
    cat = encoder.transform(np.array(cat_values).reshape(1,-1))
    num = np.array(num_val).reshape(1,-1)
    combiined = np.hstack((cat,num))
    scaled = scaler.transform(combiined)
    return model.predict(scaled)

make = ['Dodge' , 'other', 'Volkswagen', 'Nissan', 'Toyota', 'Chevrolet','Mazda', 'Honda', 'Ford', 'GMC']
models = ['other', 'Tundra', 'Frontier', 'GTI', 'Silverado 1500', 'Tacoma','Sierra 1500', 'F-150', 'Beetle Convertible', 'Accord']
engine_type = ['premium unleaded (recommended)', 'regular unleaded', 'diesel','premium unleaded (required)', 'flex-fuel (unleaded/E85)','flex-fuel (premium unleaded recommended/E85)', 'electric','flex-fuel (premium unleaded required/E85)','flex-fuel (unleaded/natural gas)', 'natural gas']
transmission_type = ['MANUAL', 'AUTOMATIC', 'AUTOMATED_MANUAL', 'UNKNOWN','DIRECT_DRIVE']
driven_wheels = ['rear wheel drive', 'front wheel drive', 'all wheel drive','four wheel drive']
vehicle_style = ['Coupe', 'Sedan', '4dr SUV', '2dr Hatchback', 'Convertible','Wagon', 'other', 'Crew Cab Pickup', 'Extended Cab Pickup','4dr Hatchback']
vehicle_size = ['Large', 'Compact', 'Midsize']
year = [year for year in range(1990,2018)]
engine_cylinders = [ 1,3,4,5,6,8,9]

app = Flask(__name__)



 
@app.route("/")
def home():
    return render_template("home.html",make_list = make,model_list = models,engine_type_list = engine_type,transmission_type_list = transmission_type,driven_wheels_list = driven_wheels,vehicle_style_list=vehicle_style ,vehicle_size_list=vehicle_size,year_list = year,engine_cylinders_list=engine_cylinders)


@app.route("/predict",methods=['POST'])
def predict():
    data =  [x for x in request.form.values()]
    result = prediction(data)[0]
    return render_template("home.html" , result=round(result,2),make_list = make,model_list = models,engine_type_list = engine_type,transmission_type_list = transmission_type,driven_wheels_list = driven_wheels,vehicle_style_list=vehicle_style ,vehicle_size_list=vehicle_size,year_list = year,engine_cylinders_list=engine_cylinders)

if __name__ == "__main__":
    app.run(debug=True)