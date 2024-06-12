#!/usr/bin/env python3

import time
import os
import smabbler
from smabbler.api.client.models.algorithm_versions_response_model import AlgorithmVersionsResponseModel
from pprint import pprint
import json

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def analyse_smbb(text, alg_version, api_instance, n_retries=5):
    ''' send query to API, wait for the result, retrieve the result '''

    # init analysis
    initialize_operation_request_model = smabbler.api.client.InitializeOperationRequestModel()
    initialize_operation_request_model.algorithm_version = alg_version
    initialize_operation_request_model.text = text

    for n in range(n_retries+1):
        try:
            api_response_init = api_instance.analyze_initialize_post(
                initialize_operation_request_model)

        except Exception as e:
            print("Exception when calling DefaultApi->analyze_initialize_post: %s\n" % e)
            time.sleep(2)
            continue
        
        else:
            break

    # check if results available and get them or timeout
    operation_status_model = smabbler.api.client.OperationStatusModel()
    operation_status_model.operation_id = api_response_init.operation_id

    for n in range(n_retries+1):
        # check if results are available
        try:
            api_response = api_instance.analyze_status_post(operation_status_model)
        except Exception as e:
            print("Exception when calling DefaultApi->analyze_status_post: %s\n" % e) 
            time.sleep(2)
            continue

        
        if api_response.status == 'processed':
            # get the results
            try:
                api_response = api_instance.analyze_result_post(operation_status_model)
                return api_response
                
            except Exception as e:
                print("Exception when calling DefaultApi->analyze_result_post: %s\n" % e)
                return
                
        time.sleep(2)
    
    print('Timeout')
    return


def extract_results(res):
    ''' extracts annotations from smabbler output '''
    return [i.result for i in res.result.items]


def reshape_to_features(df, col_res, col_id, col_label):
    ''' reshapes the results dataframe (df) to the format usable with
        sklearn, treating smabbler annotations as features (columns)

        df - pandas DataFrame containing smabbler annotation, IDs and labels
        col_res - column name containing smabbler annotations
        col_id - column name containing IDs
        col_label - column name containing labels (y)

        returns:
        pandas DataFrame
    '''
    to_df = []
    
    for idx, gdf in df.groupby(by=col_id):
        tmp = {k:1 for k in gdf[col_res].iloc[0]}
        tmp[col_label] = gdf.iloc[0][col_label]
        tmp[col_id] = idx
        to_df.append(tmp)
    
    df_r = pd.DataFrame(to_df)
    
    df_r = df_r.fillna(0)

    df_r = df_r.set_index(col_id)

    return df_r



def main():
    # load input data
    df_data = pd.read_csv('data_red.csv', index_col='ID')
   
    df_data = df_data.sample(frac=1)

    # configuration: endpoint and authorization key
    configuration = smabbler.api.client.Configuration(
        host="https://beta.api.smabbler.com"
    )
    
    configuration.api_key['api_key'] = '<<YOUR_API_KEY_HERE>>'
    
    # in this example we are using algorithm that extracts symptoms and diseases
    # other off the shelf algorithms can be found by calling:
    # api_instance.algorithm_versions_get()
    # custom algorithms can be created here:
    # https://www.smabbler.com/alpha 
    algo = 'medicalconditionsanddiseases'
    
    # create api instance and get annotations for the input text
    with smabbler.api.client.ApiClient(configuration) as api_client: 
        api_instance = smabbler.api.client.DefaultApi(api_client)

        df_data['res_raw'] = df_data['Text'].map(
            lambda text: analyse_smbb(text, algo, api_instance, n_retries=5))

    # extract results
    df_data['res'] = df_data['res_raw'].map(extract_results)
    
    # prepare results for usage in sklearn and split into train and test
    col_res, col_id, col_label = 'res', 'ID', 'LabelID'
    
    df_tmp = reshape_to_features(df_data, col_res, col_id, col_label)
    
    cols_X = list(set(df_tmp.columns) - set([col_label]))

    X_train, X_test, y_train, y_test = train_test_split(
        df_tmp[cols_X], df_tmp[col_label],
        shuffle=True, stratify=df_tmp[col_label],
        test_size=0.4, random_state=42,
    ) 
    
    # train and test a model
    clf = RandomForestClassifier()
    
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_test)
  
    print('\n\n---------------')
    print('confusion matrix:')
    print(confusion_matrix(pred, y_test))


if __name__ == '__main__':
    main()
