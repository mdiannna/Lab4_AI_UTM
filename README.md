# Lab4_AI_UTM

Lab 4 for the Fundamentals of Artificial Ingelligence Course at Technical University of Moldova. 

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [How to use](#how-to-use)
* [Troubleshooting](#troubleshooting)
* [Bibliography and resources](#bibliography-and-resources)

## Introduction
This laboratory work is based on using LinearRegression model to predict apartment prices, based on a dataset of numeric data, and deploying it to cloud.
This project implements a continuous integration/delivery pipeline, using Heroku and Github actions.


## Technologies
- Python
- Scikit-learn
- Pandas
- Numpy
- Sanic framework
- Heroku

## How to use

The system is available as a HTTP REST API, with 3 routes available:

- POST /train
- POST /evaluate
- POST /predict
 
#### The system can be used with curl requests - on the online server:
- Example of POST /train request with curl on local server:

``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'col_to_predict=medianCompexValue' -F 'dataset_path=https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt' -F 'column_names=['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']' https://linear-regression-diana-utm.herokuapp.com/train```

- Example of POST /predict request with curl:

``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'data=[-122.250000,37.850000,52.000000,1274.000000,235.000000,558.000000,219.000000,5.643100] ' https://linear-regression-diana-utm.herokuapp.com/predict```

- Example of POST /evaluate request with curl:

``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'col_to_predict=medianCompexValue' -F 'dataset_path=https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt' -F 'column_names=['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']' https://linear-regression-diana-utm.herokuapp.com/evaluate```


#### The system can be used with curl requests - on local server:

- Example of POST /train request with curl on local server:

``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'col_to_predict=medianCompexValue' -F 'dataset_path=https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt' -F 'column_names=['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']' http://localhost:5000/train```

- Example of POST /predict request with curl:

``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'data=[-122.250000,37.850000,52.000000,1274.000000,235.000000,558.000000,219.000000,5.643100] ' http://localhost:5000/predict```

- Example of POST /evaluate request with curl:

``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'col_to_predict=medianCompexValue' -F 'dataset_path=https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt' -F 'column_names=['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']' http://localhost:5000/evaluate```

## Troubleshooting
If there are some problems with heroku deploy, login and try the following command:

``` $heroku ps:scale web=1 --app=linear-regression-diana-utm ```

## Bibliography and resources 
- GitHub  Student  Developer  Pack.  Accessed  February  25,  2021. https://education.github.com/pack.
- Github Actions. Accessed February 25, 2021.https://github.com/mdiannna/Lab4_AI_UTM/actions/new.
- GitHub Integration (Heroku GitHub Deploys) by Heroku Dev Center. Accessed February 25, 2021. https://devcenter.heroku.com/articles/github-integration.
- Building a Flask App with CI/CD using Travis CI and Heroku. Accessed February 25,2021.https://srujandeshpande.medium.com/building-a-python3-flask-web-app-with-ci-cd-using-travis-ci-and-heroku-10529f9d27a.
- CI/CD: Flask App + GitHub, Travis,  Heroku by Samuel ATULE. Accessed February 25,2021.https://sweetcode.io/flask-app-github-travis-heroku/.
- Sanic official documentation. Accessed February 25, 2021.https://sanic.readthedocs.io/en/latest/.10
- https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
- https://devcenter.heroku.com/articles/heroku-ci
- https://towardsdatascience.com/how-to-deploy-your-fastapi-app-on-heroku-for-free-8d4271a4ab9
- https://devcenter.heroku.com/articles/error-codes#r10-boot-timeout
- https://stackoverflow.com/questions/40356197/python-error-r10-boot-timeout-web-process-failed-to-bind-to-port-within
- https://stackoverflow.com/questions/59391560/how-to-run-uvicorn-in-heroku
