import pandas as pd 
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
import pickle

# read data
df = pd.read_csv("./data/clean_data.csv")
# split data
x = df.drop(columns="msrp")
y = df["msrp"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.33,random_state=42)

# encoding
cat_col = df.select_dtypes("object").columns.tolist()
oe  = OrdinalEncoder()
oe.fit(x_train[cat_col])
x_train[cat_col] = oe.transform(x_train[cat_col])
x_test[cat_col] = oe.transform(x_test[cat_col])

pickle.dump(oe,open("../car_price_prediction_system/pickle_models/encoder.pkl","wb"))
 

# scaling
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

pickle.dump(sc,open("../car_price_prediction_system/pickle_models/scaler.pkl","wb"))

# model building

model = RandomForestRegressor(
                                    criterion  = 'squared_error',
                                      max_depth  = None,
                                      max_features  = 'sqrt',
                                      min_samples_leaf  = 1,
                                      min_samples_split  = 5,
                                      n_estimators  = 100,
                                      random_state=42
)
model.fit(x_train,y_train)

pickle.dump(model,open("../car_price_prediction_system/pickle_models/model.pkl","wb"))

print(f"Training score : {model.score(x_train,y_train)}")
print(f"Testing score : {model.score(x_test,y_test)}")

