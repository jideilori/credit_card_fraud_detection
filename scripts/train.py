import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc, recall_score, precision_score,f1_score,roc_auc_score
import xgboost as xgb
import mlflow
import joblib
save_model = True
def remove_outlier(col):
  sorted(col)
  Q1,Q3 = col.quantile([0.25,0.75])
  IQR = Q3-Q1
  lower_range = Q1 - (1.5 * IQR)
  upper_range = Q3 + (1.5 * IQR)
  return lower_range,upper_range

def strat_split(df,target,test_size,seed):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in split.split(df, df[f"{target}"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set,strat_test_set


data = pd.read_csv('./data/raw/creditcard.csv.zip')
# Take out validation data
train_data,val_df = strat_split(data,'Class',0.2,11)

# Split into train and test set
train_data = train_data.reset_index(drop=True)


def drop_cols(df):
    df = df.drop(['Class','Amount'],axis=1)
    return df

# Replacing outliers using median values in all columns except amount and class
def outlier_mean_remover(train_data):
    train_data_cols = [i for i in train_data.columns]
    for i in train_data_cols:
        low, upp = remove_outlier(train_data[f'{i}'])
        train_data[f'{i}'] = np.where(train_data[f'{i}']>upp,
                                    upp,train_data[f'{i}'])
        train_data[f'{i}'] = np.where(train_data[f'{i}']<low ,
                                    low,train_data[f'{i}'])


    return train_data

def add_cols(final_df,df):
     # Add back amount and class to dataframe
    final_df['Amount'] = df['Amount']
    final_df['Class'] = df['Class']
    return final_df

train_data = train_data.reset_index(drop=True)
final_df = drop_cols(train_data)
final_df =  outlier_mean_remover(final_df)
train_data = add_cols(final_df,train_data)

test_split_seed = 25
test_split_ratio =0.2
train_df,test_df = strat_split(train_data,'Class',test_split_ratio,test_split_seed)



x_train = train_df.drop('Class',axis=1)
y_train = train_df['Class']

x_test = test_df.drop('Class',axis=1)
y_test = test_df['Class']

x_val = val_df.drop('Class',axis=1)
y_val = val_df['Class']

run_name ='xgb_fraud_clf'
with mlflow.start_run(run_name=run_name) as run:
    # get current run and experiment id
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id

    xgbclf=xgb.XGBClassifier(
        max_depth=4,
        random_state=42,
        learning_rate=0.5,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='mlogloss',
        )


    # train and predict
    xgbclf.fit(x_train, y_train)
    # Testing
    print('---------------------------------------------------------------')
    print('----------------------Testing---------------------------------')

    y_pred = xgbclf.predict(x_test)
    test_conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(test_conf_matrix)
    print(classification_report(y_test, y_pred))
    print(recall_score(y_test,y_pred))

    # Validation

    print('-----------------------------------------------------------------')
    print('----------------------Validation---------------------------------')
    y_val_pred = xgbclf.predict(x_val)
    if save_model:
        joblib.dump(final_model, './model_checkpoints/xgb_fraud_model.pkl')
        print('model saved')
    val_conf_matrix = pd.crosstab(y_val, y_val_pred, rownames=['Actual'], colnames=['Predicted'])
    print(val_conf_matrix)
    print(classification_report(y_val, y_val_pred))
    print(recall_score(y_val,y_val_pred))


    # Log mlflow attributes for mlflow UI

    mlflow.log_param("max_depth", xgbclf.max_depth)
    mlflow.log_param("random_state", xgbclf.random_state)
    mlflow.log_param("learning_rate", xgbclf.learning_rate)
    mlflow.log_param("n_estimators", xgbclf.n_estimators)
    mlflow.log_param("test_split_seed", test_split_seed)
    mlflow.log_param("test_split_ratio",test_split_ratio)

    test_conf_matrix = confusion_matrix(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_pred)

    # confusion matrix values
    test_tp = test_conf_matrix[0][0]
    test_tn = test_conf_matrix[1][1]
    test_fp = test_conf_matrix[0][1]
    test_fn = test_conf_matrix[1][0]

    # actually FRAUD but model says its not fraud
    mlflow.log_metric("test_false_negative", test_fn)
    # actually NOT fraud but model says it is fraud
    mlflow.log_metric("test_false_positive", test_fp) 

    val_conf_matrix = confusion_matrix(y_val, y_val_pred)

    # confusion matrix values
    val_tp = val_conf_matrix[0][0]
    val_tn = val_conf_matrix[1][1]
    val_fp = val_conf_matrix[0][1]
    val_fn = val_conf_matrix[1][0]

    # actually FRAUD but model says its not fraud
    mlflow.log_metric("val_false_negative", val_fn)
    # actually NOT fraud but model says it is fraud
    mlflow.log_metric("val_false_positive", val_fp) 








