import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

missing_values = ["nan", "model", "other", "missing"]
data = pd.read_csv("data/skychallenge_car_data.csv", na_values=missing_values)

#Columns which could be unnecessary | What about city and paint_color?
drop_columns = ['Unnamed: 0', 'city', 'vin', 'paint_color', 'lat', 'long', 'county_fips',
                'county_name', 'state_fips', 'state_code', 'state_name', 'weather']

#Drop that columns
data = data.drop(columns=drop_columns, axis=1)
data = data.drop(columns=["size"], axis=1)  #more than 50% nan

#See price distribution
red_square = dict(markerfacecolor='r', marker='s')
boxplot = plt.boxplot(data.price, flierprops=red_square)
whiskers = [item.get_ydata()[1] for item in boxplot['whiskers']]

plt.scatter(x=[i for i in range(len(data['price']))], y=data['price'])

idx_to_drop  = data[(data['price'] < whiskers[0]) | (data['price'] > whiskers[1])].index
data.drop(idx_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)

#Visualization of missing values
#msno.matrix(data)
#msno.bar(data)
msno.heatmap(data)

#Dealing with 'manufacturer' and 'make' data
most_manu = data['manufacturer'].value_counts().index[0]
values_manu = {'manufacturer': most_manu}
data = data.fillna(value=values_manu)

most_manu_make = data.groupby('manufacturer')['make'].value_counts()[most_manu].index[0]
values_make = {'make': most_manu_make}
data = data.fillna(value=values_make)

#Dealing with 'year' data
year_unique_val = list(data['year'].unique()).sort()
data = data[data['year'] > 1900] 
median_year = int(data['year'].median())
values_year = {'year': median_year}
data = data.fillna(value=values_year)

#Dealing with 'condition' data
stat_cond = data.groupby('year')['condition'].value_counts()
fill_dict_cond = stat_cond.unstack().idxmax(axis=1).to_dict()
data['condition'] = data['condition'].fillna(data['year'].map(fill_dict_cond))

#Dealing with 'cylinders' data
most_clnd = data['cylinders'].value_counts().index[0]
values_clnd = {'cylinders': most_clnd}
data = data.fillna(value=values_clnd)
data['cylinders'] = pd.to_numeric(data.cylinders.str.split(expand=True)[0])


#Dealing with 'fuel' data
most_fuel = data['fuel'].value_counts().index[0] #gas 93%
values_fuel = {'fuel': most_fuel}
data = data.fillna(value=values_fuel)

#Dealing with 'odometer' data
data['odometer'].replace(np.nan, -1, inplace=True)
stat_odo = data.groupby('year')['odometer'].mean().to_dict()

#filling nan (-1) with odometer from previous year (mean)
value_prev = None
for key, value in stat_odo.items():
    
    if value != -1:
        stat_odo[key] = round(value)
        value_prev = value
    
    else:
        stat_odo[key] = round(value_prev)
        
data['odometer'].replace(-1, np.nan, inplace=True)    
data['odometer'] = data['odometer'].fillna(data['year'].map(stat_odo)) 
    
#Dealing with 'title_status' data
most_ts = data['title_status'].value_counts().index[0] 
values_ts = {'title_status': most_ts}
data = data.fillna(value=values_ts)

#Dealing with 'transmission' data
cond = data['year'] < 2010
data['transmission'] = data['transmission'].fillna(cond.map({True:'manual', False: 'automatic'}))

#Dealing with 'drive' data
stat_drive = data.groupby("manufacturer")["drive"].value_counts()
fill_dict_drive = stat_drive.unstack().idxmax(axis=1).to_dict()
data['drive'] = data['drive'].fillna(data['manufacturer'].map(fill_dict_drive))

#Dealing with 'type' data
stat_type = data.groupby("manufacturer")["type"].value_counts()
fill_dict_type = stat_type.unstack().idxmax(axis=1).to_dict()
data['type'] = data['type'].fillna(data['manufacturer'].map(fill_dict_type))


#Start to encoding categorical data
unique_val = {}
columns    = list(data.columns)

for col in columns:
    
    if data[col].dtype == "O":
        unique_val[col] = len(data[col].unique())
        
manu_freq_map = data.manufacturer.value_counts().to_dict()
make_freq_map = data.make.value_counts().to_dict()

#too many unique values to make dummy variables
data.manufacturer = data.manufacturer.map(manu_freq_map)
data.make         = data.make.map(make_freq_map)

#Encoding categorical data
data = pd.get_dummies(data, drop_first=True)

X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values

#Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)

y_train = y_train.reshape(-1, 1)

#Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train[:, :5] = sc_X.fit_transform(X_train[:, :5])
X_test[:, :5] = sc_X.transform(X_test[:, :5])

y_train = sc_y.fit_transform(y_train)


#Training
from sklearn.tree import DecisionTreeRegressor

dectree_regressor = DecisionTreeRegressor(random_state=0)
dectree_regressor.fit(X_train, y_train)
y_pred_dectree = sc_y.inverse_transform(dectree_regressor.predict(X_test))
diff_dectree = y_test - y_pred_dectree
stat_dectree = {"R-Squared": r2_score(y_test, y_pred_dectree),
                "Mean Squared Error":  mean_squared_error(y_test, y_pred_dectree),
                "Mean Absolute Error": mean_absolute_error(y_test, y_pred_dectree)} 

pickle.dump(dectree_regressor, open('car_prices_dectree_model.sav', 'wb'))

from sklearn.ensemble import RandomForestRegressor

rndm_frst_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rndm_frst_regressor.fit(X_train, np.squeeze(y_train))
y_pred_rndm_frst = sc_y.inverse_transform(rndm_frst_regressor.predict(X_test))
diff_rndm_frst = y_test - y_pred_rndm_frst
stat_rndm_frst = {"R-Squared": r2_score(y_test, y_pred_rndm_frst),
                "Mean Squared Error":  mean_squared_error(y_test, y_pred_rndm_frst),
                "Mean Absolute Error": mean_absolute_error(y_test, y_pred_rndm_frst)}

pickle.dump(rndm_frst_regressor, open('car_prices_rndm_frst_model.sav', 'wb'))


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


#Build the model
model = Sequential()
model.add(Dense(output_dim=30, init="uniform", activation="relu", input_dim=31))
model.add(Dropout(rate=0.4))
model.add(Dense(output_dim=30, init="uniform", activation="relu"))
model.add(Dropout(rate=0.4))
model.add(Dense(output_dim=30, init="uniform", activation="relu"))
model.add(Dropout(rate=0.4))
model.add(Dense(output_dim=30, init="uniform", activation="relu"))
model.add(Dense(output_dim=1, init="uniform", activation="linear"))

#Compile
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

#Fit, Training
y_valid = sc_y.transform(y_test.reshape(-1, 1))
history = model.fit(X_train, y_train, validation_data=[X_test, y_valid],
                    batch_size=1024, epochs=50)

model.save('car_prices_ann_model.h5') 

y_pred_neural_net = sc_y.inverse_transform(model.predict(X_test))
diff_neural_net = y_test.reshape(-1, 1) - y_pred_neural_net
stat_neural_net = {"R-Squared": r2_score(y_test, y_pred_neural_net),
                   "Mean Squared Error":  mean_squared_error(y_test, y_pred_neural_net),
                   "Mean Absolute Error": mean_absolute_error(y_test, y_pred_neural_net)}

#Summarize history for mae
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()