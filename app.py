from sanic import Sanic
from sanic import response
import json

# for different versions of sklearn:
# from sklearn.externals import joblib
import joblib
import os
import numpy as np
import os.path
from os import path
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
import pandas as pd

app = Sanic("App Name")

from sanic_cors import CORS, cross_origin

# app = Sanic(__name__)
CORS(app)

# parameters: data, model_name
@app.route("/evaluate", methods=["POST", "GET"])
async def evaluate(request):
    return response.json({"TODO": "evaluate model route"})

# parameters: data, model_name, col_to_predict
@app.route("/train", methods=["POST", "GET"])
async def train(request):
    DATASET_PATH_DEFAULT = 'https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt'
    COLUMN_NAMES_DEFAULT = ['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']
    MODEL_NAME_DEFAULT = 'multiple_regression_model.sav'
    COL_TO_PREDICT_DEFAULT = 'medianCompexValue'
        
    print("received parameters:", request.form)

    model_name = MODEL_NAME_DEFAULT
    col_to_predict = COL_TO_PREDICT_DEFAULT
    dataset_path = DATASET_PATH_DEFAULT
    column_names = COLUMN_NAMES_DEFAULT

    if 'model_name' in request.form:
        model_name = request.form.get('model_name')

    if 'col_to_predict' in request.form:
        col_to_predict = request.form.get('col_to_predict')    
    
    if 'dataset_path' in request.form:
        dataset_path = request.form.get('dataset_path')
    
    if 'column_names' in request.form:
        # column_names = json.loads(request.form.get('column_names'))
        column_names = request.form.get('column_names').replace("[", "").replace("]", "").replace(" ", "").split(",")


        print("column names:", column_names)
    
    df = pd.read_csv(dataset_path, names=column_names)
    
    X = df.copy()
    X = X.drop(columns=[col_to_predict])
    y = df[col_to_predict].values.flatten()

    reg = LinearRegression().fit(X, y)
    joblib.dump(reg, model_name)

    input_params = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "column_names": column_names,
        "col_to_predict": col_to_predict
    }

    return response.json({"status": "succes", "model_name_trained": model_name, "input_parameters": input_params})


# parameters: data, model_name
@app.route("/predict", methods=['POST', "GET"])
async def predict(request):

    # COLUMN_NAMES_DEFAULT = ['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']
    MODEL_NAME_DEFAULT = 'multiple_regression_model.sav'
        
    print("received parameters:", request.form)

    model_name = MODEL_NAME_DEFAULT
    data = []

    if 'model_name' in request.form:
        model_name = request.form.get('model_name')

    if 'data' in request.form:
        data = request.form.get('data').replace("[", "").replace("]", "").replace(" ", "").split(",")
        try:
            data = [float(x) for x in data]
        except Exception as e:
            return response.json({"status": "error. " + str(e), "message": "wrong data format"}, status=400)

        print("data:", data)
    
    model = joblib.load(model_name)
    # print(model)

    X = np.array(data)

    if len(X.shape)==1:
        X = np.array([data])

    
    y_predicted = model.predict(X)

    return response.json({"status": "success. predict route", "prediction": str(y_predicted)})


@app.route("/")
async def index(request):
    available_routes = {
        "for making predictions": "POST /predict (data, model_name)",
        "for training the model": "POST /train (data, model_name)",
        "for evaluating the model": "POST /evaluate (data, model_name)"
    }

    return response.json({"available_routes": available_routes})

if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True)
    port1 = int(os.environ.get('PORT', 5000))
    # uvicorn.run(app, host='0.0.0.0', port=port1)
    app.run(host="0.0.0.0", port=port1)