import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)
model = load_model('model1.h5')

'''
preprocessing
'''


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data1=pd.read_csv("data1.csv")
    df1=data1.reset_index()['Close/Last']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    datemax="24/06/2022"
    datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
    x_input=df1[:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    date1 = request.form.get("date")
    date1=str(date1)
    date1=dt.datetime.strptime(date1,"%d/%m/%Y")
    nDay=date1-datemax
    nDay=nDay.days
    lst_output=[]
    n_steps=150
    i=0
    while(i<=nDay):
    
        if(len(temp_input)>n_steps):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    res =scaler.inverse_transform(lst_output)
    output = res[nDay-1]

    return render_template('index.html', prediction_text='predicted_value on given date $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)