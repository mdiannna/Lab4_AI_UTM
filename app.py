from sanic import Sanic
from sanic import response

import joblib
import os
import numpy as np
import pandas as pd
import os.path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


app = Sanic("App Name")

from sanic_cors import CORS, cross_origin

CORS(app)


@app.route("/evaluate", methods=["POST", "GET"])
async def evaluate(request):
    """ evaluate model route """

    if request.method=='GET':
        return response.json("EVALUATE ROUTE. Please make a POST request!")


    MODEL_NAME_DEFAULT = 'multiple_regression_model.sav'
    COLUMN_NAMES_DEFAULT = ['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']
    COL_TO_PREDICT_DEFAULT = 'medianCompexValue'
    TEST_DATASET_PATH_DEFAULT = 'https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt'

        
    print("received parameters:", request.form)

    model_name = MODEL_NAME_DEFAULT
    col_to_predict = COL_TO_PREDICT_DEFAULT
    test_dataset_path = TEST_DATASET_PATH_DEFAULT
    column_names = COLUMN_NAMES_DEFAULT

    try:
        model = joblib.load(model_name)
    except Exception as e:
        return response.json({"status": "error", "message": "Could not load model " + str(model_name)}, status=500)


    test_data = []

    if 'model_name' in request.form:
        model_name = request.form.get('model_name')
    
    if 'test_dataset_path' in request.form:
        test_dataset_path = request.form.get('test_dataset_path')

    if 'col_to_predict' in request.form:
        col_to_predict = request.form.get('col_to_predict') 
    

    if 'column_names' in request.form:
        column_names = request.form.get('column_names').replace("[", "").replace("]", "").replace(" ", "").split(",")

    try:
        df = pd.read_csv(test_dataset_path, names=column_names)
        
        X_test = df.copy()
        X_test = X_test.drop(columns=[col_to_predict])
        y_real = df[col_to_predict].values.flatten()

        
        print("Score on test (r-squared score):", model.score(X_test, y_real))
        print("Explained variance score:", explained_variance_score(y_real, model.predict(X_test)))
        print("MSE:", mean_squared_error(y_real, model.predict(X_test)))
        cv_score_test = cross_val_score(model, X_test, y_real, cv=5)
        print("cross val score:", cv_score_test)
        print("average of cross val scores:", np.average(cv_score_test))
    except Exception as e:
        return response.json({"status": "error", "message": str(e)}, status=500)


    scores = {
        'R-squared': model.score(X_test, y_real),
        'Explained variance score': explained_variance_score(y_real, model.predict(X_test)),
        'MSE': mean_squared_error(y_real, model.predict(X_test)),
        'Cross vall score average': np.average(cv_score_test)
    }

    return response.json({"status": "success", "scores": str(scores)})


@app.route("/train", methods=["POST", "GET"])
async def train(request):
    """ train model route """
    if request.method=='GET':
        return response.json("TRAIN ROUTE. Please make a POST request!")

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
        column_names = request.form.get('column_names').replace("[", "").replace("]", "").replace(" ", "").split(",")

        print("column names:", column_names)
    
    try:
        df = pd.read_csv(dataset_path, names=column_names)
        
        X = df.copy()
        X = X.drop(columns=[col_to_predict])
        y = df[col_to_predict].values.flatten()

        reg = LinearRegression().fit(X, y)
        joblib.dump(reg, model_name)
    except Exception as e:
        return response.json({"status": "error", "message": str(e)}, status=500)


    input_params = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "column_names": column_names,
        "col_to_predict": col_to_predict
    }

    return response.json({"status": "succes", "model_name_trained": model_name, "input_parameters": input_params})


@app.route("/predict", methods=['POST', "GET"])
async def predict(request):
    """ make a prediction using model route """

    if request.method=='GET':
        return response.json("PREDICT ROUTE. Please make a POST request!")

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
    
    try:
        model = joblib.load(model_name)
    except Exception as e:
        return response.json({"status": "error", "message": "Could not load model " + str(model_name)}, status=500)

    try:
        X = np.array(data)

        if len(X.shape)==1:
            X = np.array([data])
        
        y_predicted = model.predict(X)
    except Exception as e:
        return response.json({"status": "error", "message": str(e)}, status=500)

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
    port1 = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port1)