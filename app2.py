import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model2 = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home2():
  return render_template('index2.html')

@app.route('/submit2',methods=['POST'])
def submit2():
     if request.method == "POST":
        name = request.form["username"]
        return render_template("submit2.html", n=name)
@app.route("/predict", methods=["POST"])
def predict2():
    input_features = [int(x) for x in request.form.values()]
    features_values = [np.array(input_features)]
    features_name = ['age_at_diagnosis', 'chemotherapy' , 'hormone_therapy' , 'lymph_nodes_examined_positive' , 'overall_survival' ,'radio_therapy' , 'tumor_size','tumor_stage' , 'primary_tumor_laterality_Left', 'primary_tumor_laterality_Right']
    df = pd.DataFrame(features_values, columns=features_name)
    output2 = model2.predict(df)
    return output2

if __name__ == "__main__":
  app.run()
