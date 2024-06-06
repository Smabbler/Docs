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


def analyse_smbb(text, alg_version, configuration, timeout=10):
    ''' send query to API, wait for the result, retrieve the result '''
    
    with smabbler.api.client.ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = smabbler.api.client.DefaultApi(api_client)

        # init analysis
        initialize_operation_request_model = smabbler.api.client.InitializeOperationRequestModel()
        initialize_operation_request_model.algorithm_version = alg_version
        initialize_operation_request_model.text = text

        try:
            api_response_init = api_instance.analyze_initialize_post(
                initialize_operation_request_model)
        except Exception as e:
            print("Exception when calling DefaultApi->analyze_initialize_post: %s\n" % e)
            return

        # check if results available and get them or timeout
        api_instance = smabbler.api.client.DefaultApi(api_client)
        operation_status_model = smabbler.api.client.OperationStatusModel()
        operation_status_model.operation_id = api_response_init.operation_id

        n_tries = timeout // 2
        for n in range(n_tries):
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
   return [i.result for i in res.result.items]


def reshape_to_features(df, col_res, col_id, col_label):
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


def main(offline=False):
    df_tr = pd.read_csv('train.csv', index_col='ID')
    df_tt = pd.read_csv('test.csv', index_col='ID')
    
    df_tr = df_tr[df_tr['LabelID'].isin(['ASC99', 'ASC165'])].copy()
    df_tt = df_tt[df_tt['LabelID'].isin(['ASC99', 'ASC165'])].copy()
    
    
    configuration = smabbler.api.client.Configuration(
        host="<<SMABBLER API URL>>"
    )
    
    # Configure API key authorization: api_key
    configuration.api_key['api_key'] = '<<YOUR API KEY HERE>>'
    
    # we are using algorithm for extraction of symptoms and diseases
    algo = 'medicalconditionsanddiseases'
    
    # process input text
    if not offline:
        df_tr['res_raw'] = df_tr['Text'].map(
            lambda text: analyse_smbb(text, algo, configuration, timeout=10))
        df_tr = df_tr.dropna(subset='res_raw')
        df_tr['res'] = df_tr['res_raw'].map(extract_results)
    
        df_tt['res_raw'] = df_tt['Text'].map(
            lambda text: analyse_smbb(text, algo, configuration, timeout=10))
        df_tt = df_tt.dropna(subset='res_raw')
        df_tt['res'] = df_tt['res_raw'].map(extract_results)

    else:
        df_tt = pd.read_csv('test_ready.csv', index_col='ID')
        df_tr = pd.read_csv('train_ready.csv', index_col='ID')

    # prepare results for usage in sklearn
    col_res, col_id, col_label = 'res', 'ID', 'LabelID'
    
    df_tr_ = reshape_to_features(df_tr, col_res, col_id, col_label)
    df_tt_ = reshape_to_features(df_tt, col_res, col_id, col_label)
    
    df_tmp = pd.concat([df_tr_, df_tt_]).fillna(0.0).sort_index(axis=1)
    
    df_train = df_tmp.iloc[:df_tr_.shape[0]]
    df_test = df_tmp.iloc[df_tr_.shape[0]:]
    
    cols_X = list(set(df_tmp.columns) - set([col_label]))
    
    
    # train and test model
    clf = RandomForestClassifier()
    
    clf.fit(df_train[cols_X], df_train[col_label])
    
    pred = clf.predict(df_test[cols_X])
  
    print('\n\n---------------')
    print('confusion matrix:')
    print(confusion_matrix(pred, df_test[col_label]))


if __name__ == '__main__':
    offline = False
    main(offline=offline)
