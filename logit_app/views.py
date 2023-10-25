# views.py
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import statsmodels.api as sm

def index(request):
    if request.method == 'POST':
        # get the uploaded csv file
        csv_file = request.FILES['csv_file']
        # read the csv file into a pandas dataframe
        df = pd.read_csv(csv_file,encoding="utf-8")
        # save the dataframe to the database using bulk_create
        
        # get the column names
        column_names = list(df)
        object = column_names[0]
        # set the dependent and independent variables
        y = df[column_names[0]]
        X = df.drop([column_names[0]], axis=1)
        # estimate the logistic regression model
        logit_model = sm.Logit(y, X)
        logit_model_result = logit_model.fit()
        # convert the model summary to a dataframe
        result_df = pd.read_html(logit_model_result.summary().tables[1].as_html(), header=0, index_col=0)[0]
        # pass the result dataframe to the template
        result = pd.read_html(logit_model_result.summary().tables[0].as_html(), header=0, index_col=0)[0]
        context = {
            "result_df":result_df ,
            "object":object,
            "result":result,
            
        }
        return render(request, 'logit_app/result.html', context) 
    else:
        # render the upload form
        return render(request, 'logit_app/index.html')