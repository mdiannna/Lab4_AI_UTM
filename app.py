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

app = Sanic("App Name")

from sanic_cors import CORS, cross_origin

# app = Sanic(__name__)
CORS(app)

# parameters: data, model_name
@app.route("/evaluate", methods=["POST"])
async def evaluate(request):
    return response.json({"TODO": "evaluate model route"})

# parameters: data, model_name
@app.route("/train", methods=["POST"])
async def predict(request):
    return response.json({"TODO": "train route"})

# parameters: data, model_name
@app.route("/predict", methods=['POST'])
async def predict(request):
    return response.json({"TODO": "predict route"})


@app.route("/")
async def test(request):
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