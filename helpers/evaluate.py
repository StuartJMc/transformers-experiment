
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,f1_score,roc_auc_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
import pickle
import uuid

def eval_metrics(actual, pred,pred_proba):
    score_dict={
        'roc_auc':roc_auc_score(actual,pred_proba),
        'accuracy':accuracy_score(actual,pred),
        'recall':recall_score(actual,pred),
        'precision':precision_score(actual,pred)
    }
    return score_dict



def auto_log_params(params):

    for param in params:
        mlflow.log_param(param,params[param])


def auto_log_metrics(metrics):

    for metric in metrics:
        mlflow.log_metric(metric,metrics[metric])


def train_and_test(model,X,y,mlflow_exp,params={},desc='model'):  
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    model.fit(X_train,y_train)

    myuuid = uuid.uuid4()

    with mlflow.start_run(experiment_id=mlflow_exp) as run:

        pickle.dump(model,open(f'models/{desc}-{myuuid}.pkl', 'wb'))

        mlflow.log_artifact(f'models/{desc}-{myuuid}.pkl')

        predicted_labels = model.predict(X_test)
        predicted_probabilities=model.predict_proba(X_test)[:,1]

        metrics = eval_metrics(y_test, predicted_labels,predicted_probabilities)
        mlflow.log_param('model type',desc)
        auto_log_params(params)
        auto_log_metrics(metrics)
    
        print(metrics)