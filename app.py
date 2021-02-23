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

@app.route("/")
async def test(request):
    return response.json({"hello": "world"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)