from flask import Flask, render_template, request
import pickle
import numpy as np
import sys


# Load the Random Forest CLassifier model
filename = 'model.pkl'
model_tumor = pickle.load(open(filename, 'rb'))



app = Flask(__name__,static_url_path='', static_folder='./templates')

@app.route('/')
def home():
    try:
        return render_template('./worriors.html')
    except:
        ops = str(sys.exc_info())
        return('<h1>Oops!' + ops + 'occurred</h1>')


@app.route('/predict', methods=['POST'])
def predict():
    try:
       if request.method=="POST":
         age=request.form.get("age")
         Chemo_Therapy=request.form.get("Chemo_Therapy")
         Hormone_Therapy=request.form.get("Hormone_Therapy")
         Lymph_Nodes_Examied_Positive=request.form.get("Lymph_Nodes_Examied_Positive")
         Radio_Therapy=request.form.get("Radio_Therapy")
         Mutation_Count=request.form.get("Mutation_Count")
         Overall_Survival=request.form.get("Overall_Survival")
         Tumor_Size=request.form.get("Tumor_Size")
         data = np.array([[age ,Chemo_Therapy ,Hormone_Therapy ,Lymph_Nodes_Examied_Positive , Radio_Therapy ,Mutation_Count , Overall_Survival ,Tumor_Size]])
         my_prediction = model_tumor.predict(data)
         return render_template('./submit.html', prediction=my_prediction)
    except:
        ops = str(sys.exc_info())
        return('<h1>Oops!' + ops + 'occurred</h1>')


if __name__ == '__main__':
    app.run(debug=True)