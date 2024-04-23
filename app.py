from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

API = os.getenv('API')

app = Flask(__name__)
CORS(app)

def process_and_train(inputdata):
    engine = create_engine(API)
    query = """
        SELECT * FROM final
    """
    df = pd.read_sql_query(query, engine)
    dis_sym_data = df
    columns_to_check = []
    for col in dis_sym_data.columns:
        if col != 'Disease':
            columns_to_check.append(col)
    symptoms = dis_sym_data.iloc[:, 1:].values.flatten()
    symptoms = list(set(symptoms))
    symptoms = dis_sym_data.drop(columns=['Disease']).columns
    dis_sym_data_v1 = dis_sym_data
    dis_sym_data_v1.columns = dis_sym_data_v1.columns.str.strip()
    var_mod = ['Disease']
    le = LabelEncoder()
    for i in var_mod:
        dis_sym_data_v1[i] = le.fit_transform(dis_sym_data_v1[i])
    X = dis_sym_data_v1.drop(columns="Disease")
    y = dis_sym_data_v1['Disease']
    def class_algo(model,independent,dependent):
        model.fit(independent,dependent)
        if(model_name == 'K-Nearest Neighbors'):
            pred = model.predict(independent.values)
        else:
            pred = model.predict(independent)
        accuracy = metrics.accuracy_score(pred,dependent)
        print(model_name,'Accuracy : %s' % '{0:.3%}'.format(accuracy))
    algorithms = {'Logistic Regression': 
              {"model": LogisticRegression()},
              
              'Decision Tree': 
              {"model": tree.DecisionTreeClassifier()},
              
              'K-Nearest Neighbors' :
              {"model": KNeighborsClassifier()},
             }

    for model_name, values in algorithms.items():
        class_algo(values["model"],X,y)
    doc_data = pd.read_sql_query("SELECT * FROM doctor_versus_disease", engine)

    doc_data['Specialist'] = np.where((doc_data['Disease'] == 'Tuberculosis'),'Pulmonologist', doc_data['Specialist'])

    des_data = pd.read_sql_query("SELECT * FROM disease_description", engine)
    test_col = []
    for col in dis_sym_data_v1.columns:
        if col != 'Disease':
            test_col.append(col)
    test_data = {}
    symptoms = []
    predicted = []
    def test_input(indata):
        symptoms = indata.copy()
        predicted.clear()
        print("Symptoms you have:", symptoms)
        for column in test_col:
            test_data[column] = 1 if column in symptoms else 0
            test_df = pd.DataFrame(test_data, index=[0])
        print("Predicting Disease based on 6 ML algorithms...")
        for model_name, values in algorithms.items():
            if(model_name == 'K-Nearest Neighbors'):
                predict_disease = values["model"].predict(test_df.values)
            else:
                predict_disease = values["model"].predict(test_df)
            predict_disease = le.inverse_transform(predict_disease)
            predicted.extend(predict_disease)
        disease_counts = Counter(predicted)
        percentage_per_disease = {disease: (count / 6) * 100 for disease, count in disease_counts.items()}
        result_df = pd.DataFrame({"Disease": list(percentage_per_disease.keys()),
                                "Chances": list(percentage_per_disease.values())})
        result_df = result_df.merge(doc_data, on='Disease', how='left')
        result_df = result_df.merge(des_data, on='Disease', how='left')
        return result_df
    output = test_input(inputdata)
    result = { 'name': output.loc[0, 'Disease'], 'chances': output.loc[0, 'Chances'], 'doctor': output.loc[0, 'Specialist'], 'description': output.loc[0, 'Description'] }
    return result


@app.route('/', methods=['GET'])
def main():
    return 'hello'

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    
    print(data['input_data'])
    # Make predictions
    predictions = process_and_train(data['input_data'])
    
    # Return predictions as JSON
    return jsonify(predictions)


