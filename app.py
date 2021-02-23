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

@app.route("/evaluate")
async def evaluate(request):
    return response.json({"TODO": "evaluate model route"})

@app.route("/train")
async def predict(request):
    return response.json({"TODO": "train route"})

@app.route("/predict")
async def predict(request):
    return response.json({"TODO": "predict route"})


@app.route("/")
async def test(request):
    return response.json("Hello world!")

if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True)
    port1 = int(os.environ.get('PORT', 5000))
    # uvicorn.run(app, host='0.0.0.0', port=port1)
    app.run(host="0.0.0.0", port=port1)