import flask
from flask import Flask,render_template,request,jsonify
from sklearn.externals import joblib
from treeinterpreter import treeinterpreter as ti
import pandas as pd
import json

app=Flask(__name__)

@app.route('/')
@app.route('/hello',methods=['POST'])
def hello():
    return flask.render_template('inded.html')

@app.route('/predict',methods=['POST'])
def make_prediction():
    if request.method=='POST':
        sample=[]
        data={}
        feature_names=['satisfaction', 'evaluation', 'projectCount', 'averageMonthlyHours', 'yearsAtCompany', 'workAccident', 'promotion', 'salary', 'department']

        #print request.form
        satisfaction=request.form.get('Satisfaction')
        sample.append(float(satisfaction))
        evaluation=request.form.get('Evaluation')
        sample.append(float(evaluation))
        projectCount=request.form.get('Project Count')
        sample.append(int(projectCount))
        averageMonthlyHours=request.form.get('Average Monthly Hours')
        sample.append(int(averageMonthlyHours))
        yearsAtCompany=request.form.get('Years At Company')
        sample.append(int(yearsAtCompany))
        workAccident=request.form.get('Work Accident')
        sample.append(int(workAccident))
        promotion=request.form.get('Promotion')
        sample.append(int(promotion))
        salary=request.form.get('Salary')
        sample.append(int(salary))
        department=request.form.get('Department')
        sample.append(department)
        sampleDf= pd.DataFrame(sample)
        if len(sample)<9:
            return render_template('result.html', label1="Missing data or incorrect data entered")
        # make prediction
        prediction = model.predict(sampleDf.T)
        pred, bias, contributions = ti.predict(model, sampleDf.T)
        confidence = model.predict_proba(sampleDf.T)
        data["confidence_0"]=confidence[0][0]
        data["confidence_0"]=confidence[0][1]
        data["Prediction"]=prediction[0]
        #json_cities = json.dumps(city_array)
        #return render (request, 'plot3/plot_page.html', {"city_array" : json_cities})

        data[0]={}
        data[1]={}
        for c in range(len(contributions[0])):

            data[0][feature_names[c]]=round(contributions[0][c][0],2)
            data[1][feature_names[c]]=round(contributions[0][c][1],2)

        json_data = json.dumps(data)
        jsonify(data=data)
        return render_template('result.html', data=data)
if __name__ =='__main__':
    model = joblib.load('model.pkl')
    app.run(host='127.0.0.1', port=8002, debug=True)
