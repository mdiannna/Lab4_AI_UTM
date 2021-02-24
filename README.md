# Lab4_AI_UTM

Lab 4 for the Fundamentals of Artificial Ingelligence Course at Technical University of Moldova

## How to use
#### Make a post request with curl:
``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'col_to_predict=medianCompexValue' -F 'dataset_path=https://raw.githubusercontent.com/mdiannna/Labs_UTM_AI/main/Lab3/apartmentComplexData.txt' -F 'column_names=['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']' http://localhost:5000/train```


``` $ curl -X POST -F 'model_name=my_reg_model.sav' -F 'data=[-122.250000,37.850000,52.000000,1274.000000,235.000000,558.000000,219.000000,5.643100] ' http://localhost:5000/predict```


## Troubleshooting
If there are some problems with heroku deploy, login and try the following command:

`` $heroku ps:scale web=1 --app=linear-regression-diana-utm ```

## Bibliography and resources 
- https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
- https://education.github.com/pack/offers
- https://docs.github.com/en/actions/guides/deploying-to-azure-app-service
- https://devcenter.heroku.com/articles/heroku-ci
- https://devcenter.heroku.com/articles/github-integration
- https://dashboard.heroku.com/apps/linear-regression-diana-utm/deploy/github
- https://towardsdatascience.com/mlops-a-tale-of-two-azure-pipelines-4135b954997
- https://towardsdatascience.com/how-to-deploy-your-fastapi-app-on-heroku-for-free-8d4271a4ab9
- https://devcenter.heroku.com/articles/error-codes#r10-boot-timeout
- https://stackoverflow.com/questions/40356197/python-error-r10-boot-timeout-web-process-failed-to-bind-to-port-within
- https://stackoverflow.com/questions/59391560/how-to-run-uvicorn-in-heroku