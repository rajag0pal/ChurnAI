from flask import Flask, render_template,request
import os
import keras
import numpy as np
app = Flask(__name__)
model_path = os.path.dirname(os.path.realpath(__file__)) + '/saved_model/my_model'
# Loading the model
model = keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index.html')
def Home():
    return render_template("index.html")


@app.route('/About.html.html')
def Model():
    return render_template("About.html.html")

@app.route('/Data.html.html')
def Data():
    return render_template("Data.html.html")

@app.route('/Features.html.html')
def Features():
    return render_template("Features.html.html")

@app.route('/Stat.html.html')
def Statistics():
    return render_template("Stat.html.html")

@app.route('/Predict.html.html')
def Predict():
    return render_template("Predict.html.html")

@app.route('/predict',methods=['POST'])
def predict():
    def crscale(data):
        mean = 650.528800
        sd = 96.653299
        value = (data - mean) / sd
        return value
    def agescale(data):
        mean = 38.921800
        sd = 10.487806
        value1 = (data - mean) / sd
        return value1
    def Tenscale(data):
        mean = 5.012800
        sd = 2.892174
        value2 = (data - mean) / sd
        return value2
    def Balscale(data):
        mean = 76485.889288
        sd = 62397.405202
        value3 = (data - mean) / sd
        return value3

    def Salscale(data):
        mean = 100090.239881
        sd = 57510.492818
        value4 = (data - mean) / sd
        return value4

    features = [float(x) for x in request.form.values()]

    crscore = features[0]
    crscore = crscale(crscore)

    age = features[2]
    age = agescale(age)

    tenure = features[3]
    tenure = Tenscale(tenure)

    balance = features[4]
    balance = Balscale(balance)

    salary = features[7]
    salary = Salscale(salary)

    gender = features[1]
    crcard = features[5]
    active = features[6]
    pro1 = features[8]
    pro2 = features[9]
    pro3 = features[10]
    pro4 = features[11]
    coun1 = features[12]
    coun2 = features[13]
    coun3 = features[14]

    finalFeatures = [crscore, gender, age, tenure, balance, crcard, active, salary, pro1, pro2, pro3, pro4, coun1, coun2, coun3]

    finalvalues = np.array(finalFeatures)
    finalvalues = finalvalues.reshape(1, 15)

    res = model.predict(finalvalues)
    res = res.reshape(1)
    res = float(res)*100
    res = round(res)

    return render_template("Predict.html.html", predict_text="The probability of this type of Customer leaving the bank is {}%".format(res), sug_text="Explore more on ")


if __name__ == "__main__":
    app.run(debug=True)