# python3 -m pytest -v tests/test_app.py

import unittest
from app import app

app.testing = True


def test_index_returns_200():
    request, response = app.test_client.get('/')
    assert response.status == 200

def test_get_train_returns_200():
    request, response = app.test_client.get('/train')
    assert response.status == 200

def test_post_train_returns_200():
    request, response = app.test_client.post('/train')
    assert response.status == 200


def test_get_predict_returns_200():
    request, response = app.test_client.get('/predict')
    assert response.status == 200

def test_train_and_predict():
    model_name = 'test_model.sav'
    request, response = app.test_client.post('/train', data={'model_name':model_name})
    assert response.status == 200

    test_data = [-122.250000,37.850000,52.000000,1274.000000,235.000000,558.000000,219.000000,5.643100]
    request, response = app.test_client.post('/predict', data={'model_name':model_name, 'data':str(test_data)})
    assert response.status == 200

def test_get_evaluate_returns_200():
    request, response = app.test_client.get('/evaluate')
    assert response.status == 200

def test_post_evaluate_returns_200():
    request, response = app.test_client.post('/evaluate')
    assert response.status == 200


def test_index_put_not_allowed():
    request, response = app.test_client.put('/')
    assert response.status == 405
