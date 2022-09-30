from flask import Flask,render_template,request
import numpy as np 
import pandas as pd 
import sklearn
import joblib


app=Flask(__name__)
finaltest=joblib.load('classifier.pkl')


@app.route('/')
@app.route('/main')
def main():
	return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
	int_features=[[i for i in request.form.values()]]


	print("#################################")
	print(int_features)
	print("***")
	features=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
	print(features)

	df = pd.DataFrame(int_features,columns=features)
	result=finaltest.predict(df)
	return render_template("index.html",prediction_text="Estimated crop is :{}".format(result))



if __name__ == "__main__":
 	app.debug=True 
 	app.run('127.0.0.4',port=7000)
