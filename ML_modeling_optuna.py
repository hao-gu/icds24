import sys
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from joblib import dump
from joblib import load
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras import Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
import optuna
import pickle
from sklearn.metrics import mean_squared_error
import os
import plotly

model_id = 40


# customize ann model
def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 5, 10) #5, 10 / 10, 15 / 15, 20 ... 25, 30
    in_features = 8
    layer_name = 'Flux_intensity' 
    input_part1 = Input(shape=(in_features,), name=f'{layer_name}_Input_9')
    
    #define hyperparameters
    activation_f = trial.suggest_categorical("activation", ['relu', 'softplus','sigmoid'])
    output_activation = trial.suggest_categorical("out_activation", ['softplus', 'linear', 'relu','sigmoid'])
    
    #create model
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 5, 15) # 5,15 / 15, 30 / 30, 45
        if i == 0:
            ann_part1 = Dense(units=out_features, activation=activation_f,  name=f'{layer_name}_layer_{i+1}_{out_features}')(input_part1)
        else:
            ann_part1 = Dense(units=out_features, activation=activation_f,  name=f'{layer_name}_layer_{i+1}_{out_features}')(ann_part1)
    ann_part1 = Dense(units=1, activation=output_activation,  name=f'{layer_name}_layer_{i+1}_1')(ann_part1)
    
    #layer_name = 'Inundation'
    #input_part2 = Input(shape=(1,), name=f'{layer_name}_Input_1')
    #combined = tf.keras.layers.Multiply()([ann_part1, input_part2])
    
    #model = tf.keras.Model(inputs=[input_part1, input_part2], outputs=combined)
    model = tf.keras.Model(inputs=input_part1, outputs=ann_part1)
    
    return model

def create_optimizer(trial):
    
    kwargs = {}
    optimizer_options = ["Adam", "SGD", 'RMSprop']
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    
    if optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("Adam_learning_rate", 1e-5, 1e-3, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float("SGD_learning_rate", 1e-5, 1e-3, log=True)
        #kwargs["momentum"] = trial.suggest_float("sgd_momentum", 0.01, 0.5, log=True)
    elif optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float("RMSprop_learning_rate", 1e-5, 1e-3, log=True)
        #kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 0.01, 0.5, log=True)
    
    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    
    return optimizer

# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):
    
    X_train = pd.read_csv(mydir+'model'+str(model_id+1)+'/Xtrain.csv').values
    X_test = pd.read_csv(mydir+'model'+str(model_id+1)+'/Xtest.csv').values
    y_train = pd.read_csv(mydir+'model'+str(model_id+1)+'/ytrain.csv').values
    y_test = pd.read_csv(mydir+'model'+str(model_id+1)+'/ytest.csv').values
    
    model = create_model(trial)
    opt = create_optimizer(trial)
    
    mybatch_size = 10 
    myepochs = 500 #1000, 1500, 2000
    
    model.compile(optimizer = opt,loss = 'mse')

    history = model.fit(X_train, y_train,\
              batch_size=mybatch_size,epochs=myepochs,\
              validation_data=(X_test,y_test))
    
    error = np.mean(history.history['val_loss'][-10:])
    
    return error

if __name__ == "__main__":

    #session_config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    mydir = "../../GEDI-biomass_all_models/"
    #study_name = 'GCP_CH4_model'
    #storage_name = mydir+'model'+str(model_id+1)+'/'+study_name+'.db'

    # read GCP-CH4 model samples
    samples = pd.read_csv('all_data.csv').values #loadmat(mydir+'model'+str(model_id+1)+'/model_samples_monthly.mat')['model_samples_monthly']
    #samples = np.transpose(model_samples_monthly)#.reshape(9,-1)) # samples x features
    
    # get X, y
    X = samples[:,:8] # tempearture, rain, pressure, wind, radiation, lai, lat, lon
    y = samples[:,8] # FCH4 at gridcell level
    
    # rescaling
    sc_X = MinMaxScaler(feature_range=(0,1))
    sc_y = MinMaxScaler(feature_range=(0,1))
    
    fitted_X_scaled = sc_X.fit_transform(X)
    fitted_y_scaled = sc_y.fit_transform(y.reshape(-1,1))
    
    os.mkdir(mydir+'model'+str(model_id+1))
    
    scaler_filename = mydir+'model'+str(model_id+1)+'/sc_X.mat'
    dump(sc_X, scaler_filename)
    scaler_filename = mydir+'model'+str(model_id+1)+'/sc_y.mat'
    dump(sc_y, scaler_filename)
    
    
    savemat(mydir+'model'+str(model_id+1)+'/fitted_X_scaled.mat', {'fitted_X_scaled':fitted_X_scaled[:,:],'y':y})
    #? why
    # stratified random sampling
    n_strata = 20 # cover 0-99% percentiles, the rest is assigned to the last strata bin
    strata_bins = np.full([n_strata],np.nan)
    for j in range(n_strata):
        strata_bins[j] = np.percentile(fitted_y_scaled,99/n_strata*(j+1))
    strata_y = np.full([len(fitted_y_scaled),1],0)
    for j in range(len(fitted_y_scaled)):
        if fitted_y_scaled[j] <= strata_bins[0]:
            strata_y[j] = 1
        elif fitted_y_scaled[j] <= strata_bins[n_strata-1]:
            for k in range(n_strata-1):
                if fitted_y_scaled[j] <= strata_bins[k+1] and fitted_y_scaled[j] > strata_bins[k]:
                    strata_y[j] = k+1
        else:
            strata_y[j] = n_strata-1
    
    X_train,X_test,y_train,y_test = train_test_split(fitted_X_scaled,fitted_y_scaled,test_size=0.2,stratify=strata_y,random_state=0)
    
    pd.DataFrame(X_train).to_csv(mydir+'model'+str(model_id+1)+'/Xtrain.csv',index=False)
    pd.DataFrame(X_test).to_csv(mydir+'model'+str(model_id+1)+'/Xtest.csv',index=False)
    pd.DataFrame(y_train).to_csv(mydir+'model'+str(model_id+1)+'/ytrain.csv',index=False)
    pd.DataFrame(y_test).to_csv(mydir+'model'+str(model_id+1)+'/ytest.csv',index=False)
    
    # create optuna case
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    # optimize hyper parameters
    study.optimize(objective, n_trials=200)
    
    # output results
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value) # loss value returned from objective
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(mydir+'model'+str(model_id+1)+'/optuna_trials.csv',index=False)

    # save sampler
    with open(mydir+'model'+str(model_id+1)+'/sampler.pkl', 'wb') as fout:
        pickle.dump(study.sampler, fout)
    
    #visualize history
    optuna.visualization.plot_optimization_history(study).show(renderer="browser")
    optuna.visualization.plot_param_importances(study).show(renderer="browser")
    optuna.visualization.plot_parallel_coordinate(study).show(renderer="browser")
    #optuna.visualization.plot_slice(study, params = ['n_layers','optimizer','adam_learning_rate'])
    #FIX THIS PLEASEEEEE optuna.visualization.plot_param_importances(study)
    sys.exit()
    
    '''
    # viaualization
    optuna.visualization.matplotlib.plot_optimization_history(study)
    optuna.visualization.matplotlib.plot_param_importances(study)
    optuna.visualization.matplotlib.plot_rank(study, params=["optimizer", "n_units_l2"])

    # load previous existing sampler
    restored_sampler = pickle.load(open(mydir+'model'+str(model_id+1)+'/sampler.pkl', 'rb'))
    #study = optuna.create_study(study_name='test', storage=storage_name,load_if_exists=True,sampler=restored_sampler)
    study.optimize(objective, n_trials=100)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)
    print("Trials: ", study.trials)
'''


