from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age = request.form.get('age'),
            gender = request.form.get('gender'),
            bmi = request.form.get('bmi'),
            children = request.form.get('children'),
            smoker = request.form.get('smoker'),
            region = request.form.get('region'),
            medical_history = request.form.get('medical_history'),
            family_medical_history = request.form.get('family_medical_history'),
            exercise_frequency = request.form.get('exercise_frequency'),
            occupation =  request.form.get('occupation'),
            coverage_level = request.form.get('coverage_level')
            )
        pred_df=data.get_data_as_data_frame()
        pred_df.to_csv("test.csv")

        predict_pipeline=PredictPipeline()#predict_pipeline is obj for PredictPipeline class
        
        predicted_result=predict_pipeline.predict(pred_df)
        
        return render_template('home.html',results=predicted_result[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        

