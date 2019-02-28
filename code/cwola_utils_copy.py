from __future__ import print_function
import numpy as np
import numpy.random as rand
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle as pickle
import tensorflow as tf
import time
import glob
import numpy.ma as ma


#################################################################
####################       model_ensemble      ##################
#################################################################
"""
Helper class for managed trained models and their dataset.
Assumes nested cross-validation.
"""
#################################################################


class model_ensemble:

    def __init__(self, dataset, bin_i=8, kfolds=5, eff_for_thresh = 0.02, batch_size = 5000, preprocess = None):
        self.kfolds = kfolds
        self.dataset_split = [self.get_kth_fold(dataset, k) for k in range(kfolds)]
        self.bin_i = bin_i
        self.batch_size = batch_size
        def null_preprocess(data):
            return data
        if preprocess == None:
            self.preprocess = null_preprocess
        else:
            self.preprocess = preprocess
        self.eff_for_thresh = eff_for_thresh
        
        self.models = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.model_filenames = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.model_hists = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.effs_valid = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.effs_train = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.aucs_valid = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.aucs_train = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        
        self.predictions_valid = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.predictions_train = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.thresholds = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.ensemble_models = [None for k in range(kfolds)]
        self.ensemble_model_names = [None for k in range(kfolds)]
        
        self.current_trainval_data = None
        self.current_k = None
        self.current_l = None

    def get_kth_fold(self,data,k):
        return [databin[int(len(databin)*k/self.kfolds):int(len(databin)*(k+1)/self.kfolds)] for databin in data]

    def get_binned_train_data(self, k, l):
        #Get all data except the ones corresponding to k and l
        bins = np.arange(self.bin_i-3,self.bin_i+4)
        train_folds = [i for i in range(self.kfolds)]
        train_folds.remove(k)
        train_folds.remove(l)

        dataset = [self.dataset_split[train_folds[0]][bin_num] for bin_num in bins]
        for fold in train_folds[1:]:
            dataset = [np.append(dataset[i],
                                 self.dataset_split[fold][bin_num],
                                 axis=0)
                       for i, bin_num in enumerate(bins)]

        return dataset

    def get_kth_signalregion_data(self,k):
        binned_data = self.dataset_split[k]
        to_return = np.concatenate((binned_data[self.bin_i-1],
                                    binned_data[self.bin_i],
                                    binned_data[self.bin_i+1]),
                                   axis=0)
        return to_return

    def get_kth_sidebandregion_data(self,k):
        binned_data = self.dataset_split[k]
        to_return = np.concatenate((binned_data[self.bin_i-3],
                                    binned_data[self.bin_i-2],
                                    binned_data[self.bin_i+2],
                                    binned_data[self.bin_i+3]),
                                   axis=0)
        return to_return
    
    def get_trainval_data(self,k,l):

        if (k != self.current_k) or (l != self.current_l) or (self.current_trainval_data == None):
        
            train_dataset = self.get_binned_train_data(k,l)
            low_sideband_train = np.append(train_dataset[0],
                                           train_dataset[1],
                                           axis=0)
            signal_train = np.concatenate((train_dataset[2],
                                           train_dataset[3],
                                           train_dataset[4]),
                                          axis=0)
            high_sideband_train = np.append(train_dataset[5],
                                            train_dataset[6],
                                            axis=0)
            
            valid_dataset = [self.dataset_split[l][i] for i in range(self.bin_i-3,self.bin_i+4)]
            low_sideband_valid = np.append(valid_dataset[0],
                                           valid_dataset[1],
                                           axis=0)
            signal_valid = np.concatenate((valid_dataset[2],
                                           valid_dataset[3],
                                           valid_dataset[4]),
                                          axis=0)
            high_sideband_valid = np.append(valid_dataset[5],
                                            valid_dataset[6],
                                            axis=0)
            
            n_low = len(low_sideband_train)
            n_high = len(high_sideband_train)
            n_signal = len(signal_train)
            n_total = n_low + n_high + n_signal
            
            train_data = np.append(low_sideband_train,
                                   signal_train,
                                   axis=0)
            train_data = np.append(train_data,
                                   high_sideband_train,
                                   axis=0)
            
            valid_data = np.append(low_sideband_valid,
                                   signal_valid,
                                   axis=0)
            valid_data = np.append(valid_data,
                                   high_sideband_valid,
                                   axis=0)
            
            categories_train = np.append(np.zeros(n_low,dtype=np.int8), np.ones(n_signal,dtype=np.int8))
            categories_train = np.append(categories_train, np.zeros(n_high,dtype=np.int8))
            
            categories_valid = np.append(np.zeros(len(low_sideband_valid),dtype=np.int8), np.ones(len(signal_valid),dtype=np.int8))
            categories_valid = np.append(categories_valid, np.zeros(len(high_sideband_valid),dtype=np.int8))
            
            weights_train = np.append(np.ones(n_low)*0.25*n_total/n_low,
                                      np.ones(n_signal)*0.5*n_total/n_signal)
            weights_train = np.append(weights_train,
                                      np.ones(n_high)*0.25*n_total/n_high)
            
            weights_valid = np.append(np.ones(len(low_sideband_valid))*0.25*n_total/n_low,
                                      np.ones(len(signal_valid))*0.5*n_total/n_signal)
            weights_valid = np.append(weights_valid,
                                      np.ones(len(high_sideband_valid))*0.25*n_total/n_high)

            perms_train = np.random.permutation(len(train_data))
            perms_valid = np.random.permutation(len(valid_data))
            
            self.current_trainval_data = [train_data[perms_train], valid_data[perms_valid],
                                          categories_train[perms_train], categories_valid[perms_valid],
                                          weights_train[perms_train], weights_valid[perms_valid]]
            self.current_k = k
            self.current_l = l
            
        return self.current_trainval_data
    

    def reload_model(self,k,l,i):
        if l > k:
            l = l - 1
        self.models[k][l][i] = keras.models.load_model(self.model_filenames[k][l][i])

    def reload_models(self,k,l):
        if l > k:
            l = l - 1
        for i in range(len(self.models[k][l])):
            self.models[k][l][i] = keras.models.load_model(self.model_filenames[k][l][i])

    def reload_best_model(self,k,l):
        if l > k:
            l = l - 1
        i = np.argmax(self.effs_valid[k][l])
        self.models[k][l][i] = keras.models.load_model(self.model_filenames[k][l][i])
        
    def makeandsave_ensemble_model(self,k,modelname,load=True):
        modelset = []
        for l in range(self.kfolds-1):
            i = np.argmax(self.effs_valid[k][l])
            if load:
                modelset.append(keras.models.load_model(self.model_filenames[k][l][i]))
            else:
                modelset.append(self.models[k][l][i])
        numvars = modelset[0].layers[0].input.get_shape().as_list()[1]
        singleInput = keras.layers.Input((numvars,))
        outs = [one_model(singleInput) for one_model in modelset]
        outmerge = keras.layers.Average()(outs)
        merged_model = keras.Model(singleInput,outmerge)
        merged_model.save(modelname)
        self.ensemble_model_names[k] = modelname
        self.ensemble_models[k] = merged_model
        return self.ensemble_models[k]

    def load_all_ensemble_models(self, model_names = None):
        if model_names == None:
            model_names = self.ensemble_model_names
        for k, model_name in enumerate(model_names):
            self.ensemble_models[k] = keras.models.load_model(model_name)

            
    def add_model(self, model, model_hist, kfold, lfold, model_filename):

        train_data, valid_data, categories_train, categories_valid, weights_train, weights_valid = self.get_trainval_data(kfold,lfold)

        if lfold > kfold:
            lfold = lfold - 1
        self.models[kfold][lfold].append(model)
        self.model_hists[kfold][lfold].append(model_hist)
        self.model_filenames[kfold][lfold].append(model_filename)
        
        model_pred_valid = model.predict(self.preprocess(valid_data),batch_size = self.batch_size).flatten()
        model_pred_train = model.predict(self.preprocess(train_data),batch_size = self.batch_size).flatten()

        self.predictions_valid[kfold][lfold].append(model_pred_valid)
        self.predictions_train[kfold][lfold].append(model_pred_train)
        #print(categories_valid[:100])
        ordered_bg_list = np.flip(np.sort(model_pred_valid[categories_valid<0.5]),axis=0)
        #Find the threshold above which only 2% of bg survives
        thresh = ordered_bg_list[int(self.eff_for_thresh*len(ordered_bg_list))]
        self.thresholds[kfold][lfold].append(thresh)
        
        #Now get entries in category 1 ('sig')
        #Order these by NN output
        ordered_sig_list_valid = np.sort(model_pred_valid[categories_valid>0.5])
        #Find fraction of signal events which survive a cut on the above threshold
        sig_eff_valid = 1.0 - 1.0*np.searchsorted(ordered_sig_list_valid,thresh)/len(ordered_sig_list_valid)
        self.effs_valid[kfold][lfold].append(sig_eff_valid)

        
        ordered_sig_list_train = np.sort(model_pred_train[categories_train>0.5])
        #Find fraction of signal events which survive a cut on the above threshold
        sig_eff_train = 1.0 - 1.0*np.searchsorted(ordered_sig_list_train,thresh)/len(ordered_sig_list_train)
        self.effs_train[kfold][lfold].append(sig_eff_train)
        
        self.aucs_valid[kfold][lfold].append(roc_auc_score(categories_valid,model_pred_valid))
        self.aucs_train[kfold][lfold].append(roc_auc_score(categories_train,model_pred_train))

    def return_modelset_onek(self,k):
        model_indices = np.array([0 for i in range(self.kfolds-1)])
        for l in range(self.kfolds-1):
            model_indices[l] = np.argmax(self.effs_valid[k][l])
        return [self.models[k][l][model_indices[l]] for l in range(self.kfolds-1)]
            
    def avg_model_predict_onek(self, data, k):
        if self.ensemble_models[k] == None:
            model_indices = np.array([0 for i in range(self.kfolds-1)])
            for l in range(self.kfolds-1):
                model_indices[l] = np.argmax(self.effs_valid[k][l])
            to_return = np.average(np.array([self.models[k][l][index].predict(self.preprocess(data),batch_size=self.batch_size).flatten() for l, index in enumerate(model_indices)]),axis=0)
        else:
            to_return = self.ensemble_models[k].predict(self.preprocess(data),batch_size=self.batch_size).flatten()
        return to_return

    def avg_model_predict_lset(self, data, k):
        model_indices = np.array([0 for i in range(self.kfolds-1)])
        for l in range(self.kfolds-1):
            model_indices[l] = np.argmax(self.effs_valid[k,l])
        to_return = np.array([self.models[k][l][index].predict(self.preprocess(data),batch_size=self.batch_size).flatten() for l, index in enumerate(model_indices)])
        return np.average(to_return,axis=0), to_return

    def avg_model_predict_kset(self):
        datasets = [self.get_concat_kfold(k) for k in range(self.kfolds)]
        return datasets, [self.avg_model_predict_onek(datasets[k],k) for k in range(self.kfolds)]

    def get_concat_kfold(self,k):
        concat_set = self.dataset_split[k][0]
        for i in range(1,len(self.dataset_split[k])):
            concat_set = np.append(concat_set,self.dataset_split[k][i],axis=0)
        return concat_set
    
    def get_bin_cut_counts_individual(self, k, eff = None):
        if eff is None:
            eff = self.eff_for_thresh
        binned_predictions = [self.avg_model_predict_onek(data_bin,k).flatten() for data_bin in self.dataset_split[k]]
        concat_predictions = binned_predictions[0]
        for bin_no in range(1,len(binned_predictions)):
            concat_predictions = np.append(concat_predictions,binned_predictions[bin_no])
        concat_predictions = np.sort(1-concat_predictions)
        threshold = 1-concat_predictions[min(int(eff*len(concat_predictions)),len(concat_predictions)-1)]
        return np.array([np.sum(pred_bin > threshold) for pred_bin in binned_predictions])

    def get_bin_cut_counts_all(self, eff = None):
        if eff is None:
            eff = self.eff_for_thresh
        pass_set = np.array([self.get_bin_cut_counts_individual(k,eff) for k in range(self.kfolds)])
        pass_sum = np.sum(pass_set,axis=0)
        return pass_sum, pass_set

    def print_scatter_avg_onek_signalregion(self,k,axes_list=[[0,1]],axes_labels=None,
                                            rates = np.array([0.5,0.95,0.98,0.99]),
                                            colors=['silver','grey','khaki','goldenrod','firebrick']):
        data = self.get_kth_signalregion_data(k)
        predictions = self.avg_model_predict_onek(data, k)
        AddPredictionsToScatter(data, predictions,axes_list=axes_list,axes_labels=axes_labels,
                                rates=rates,colors=colors)

    def print_scatter_avg_allk_signalregion(self,axes_list=[[0,1]],axes_labels=None,
                                            rates = np.array([0.5,0.95,0.98,0.99]),
                                            colors=['silver','grey','khaki','goldenrod','firebrick']):
        data = []
        predictions = []
        for k in range(self.kfolds):
            data.append(self.get_kth_signalregion_data(k))
            predictions.append(self.avg_model_predict_onek(data[-1], k))
        AddPredictionsToScatter_nestedcrossval(data, predictions,axes_list=axes_list,axes_labels=axes_labels,
                                               rates=rates,colors=colors)

    def print_scatter_onemodel_signalregion(self,k,l,i,axes_list=[[0,1]],axes_labels=None,
                                            rates = np.array([0.5,0.95,0.98,0.99]),
                                            colors=['silver','grey','khaki','goldenrod','firebrick']):
        if l > k:
            l = l-1
        data = self.get_kth_signalregion_data(k)
        sideband_data = self.get_kth_sidebandregion_data(k)
        
        predictions = self.models[k][l][i].predict(self.preprocess(data)).flatten()
        predictions_sideband = self.models[k][l][i].predict(self.preprocess(sideband_data)).flatten()
        
        AddPredictionsToScatter(data, predictions,axes_list=axes_list,axes_labels=axes_labels,
                                rates=rates,colors=colors,threshold_predictions = predictions_sideband)
    
    
    def print_scatter_avg_onek_signalplussidebandregion(self,k,axes_list=[[0,1]],axes_labels=None,
                                            rates = np.array([0.5,0.95,0.98,0.99]),
                                            colors=['silver','grey','khaki','goldenrod','firebrick']):
        
        data = self.get_kth_signalregion_data(k)
        sideband_data = self.get_kth_sidebandregion_data(k)
        
        predictions = self.avg_model_predict_onek(data, k)
        predictions_sideband = self.avg_model_predict_onek(sideband_data, k)
        
        AddPredictionsToScatter(data, predictions,
                                axes_list=axes_list,axes_labels=axes_labels,
                                rates=rates,colors=colors,
                                threshold_predictions = predictions_sideband)
    
    
    def print_scatter_onemodel_signalplussidebandregion(self,k,l,i,axes_list=[[0,1]],axes_labels=None,
                                            rates = np.array([0.5,0.95,0.98,0.99]),
                                            colors=['silver','grey','khaki','goldenrod','firebrick']):
        if l > k:
            l = l-1
        data = self.get_kth_signalregion_data(k)
        sideband_data = self.get_kth_sidebandregion_data(k)
        
        predictions = self.models[k][l][i].predict(self.preprocess(data)).flatten()
        predictions_sideband = self.models[k][l][i].predict(self.preprocess(sideband_data)).flatten()
        
        AddPredictionsToScatter(np.append(data,sideband_data,axis=0),
                                np.append(predictions,predictions_sideband,axis=0),
                                axes_list=axes_list,axes_labels=axes_labels,
                                rates=rates,colors=colors,
                                threshold_predictions = predictions_sideband)

    def get_thresh_individual(self, k, eff = None):
        if eff is None:
            eff = self.eff_for_thresh
        binned_predictions = [self.avg_model_predict_onek(data_bin,k).flatten() for data_bin in self.dataset_split[k]]
        concat_predictions = binned_predictions[0]
        for bin_no in range(1,len(binned_predictions)):
            concat_predictions = np.append(concat_predictions,binned_predictions[bin_no])
        concat_predictions = np.sort(1-concat_predictions)
        return 1-concat_predictions[min(int(eff*len(concat_predictions)),len(concat_predictions)-1)]

    def get_thresh_all(self, eff = None):
        return np.array([self.get_thresh_individual(k, eff) for k in range(self.kfolds)])


        
#################################################################
#######       AddPredictionsToScatter_nestedcrossval      #######
#################################################################
"""
Creates scatter plots of all events in selected planes,
highlighting those events passing thresholds on NN output.
"""
#################################################################

def AddPredictionsToScatter(data, predictions,axes_list=[[0,1]],axes_labels=None,
                            rates = np.array([0.5,0.95,0.98,0.99]),
                            colors=['silver','grey','khaki','goldenrod','firebrick'],
                           threshold_predictions = None,
                           plotmode="show"):
    
    if threshold_predictions is None:
        threshold_predictions = predictions

    if axes_labels == None:
        axes_labels = [[None,None] for axes in axes_list]
        
    extended_rates = np.insert(rates,0,0.0)
    extended_rates = np.append(extended_rates,1.0)
    
    sorted_threshold_predictions = np.sort(threshold_predictions)
    threshold_indices = [max(int(rate*len(threshold_predictions))-1,0) for rate in extended_rates]
    thresholds = [sorted_threshold_predictions[int(index)] for index in threshold_indices]
    
    points_list = np.array([data[(predictions > thresholds[i])*(predictions < thresholds[i+1])] for i in range(len(thresholds)-1)])
    
#     sorted_args = np.argsort(predictions)
#     total_num = len(sorted_args)
#     points_list = np.array([data[sorted_args[int(extended_rates[i] * total_num):int(extended_rates[i+1] * total_num)]]
#                             for i in range(0,len(extended_rates)-1)])
    
    plt.figure(figsize=(5*len(axes_list),5))
    size = 0.1
    for h, axes in enumerate(axes_list):
        plt.subplot(1, len(axes_list), h+1)
        plt.xlabel(axes_labels[h][0])
        plt.ylabel(axes_labels[h][1])
        for i, points in enumerate(points_list):
            size = 0.1
            if i == len(points_list)-1:
                size = 1.0
            plt.scatter(points[:,axes[0]],points[:,axes[1]],
                        s=size, color=colors[i])
    if plotmode == "show":
        plt.show()
                
    return [rates]


#################################################################
#######       AddPredictionsToScatter_nestedcrossval      #######
#################################################################
"""
Creates scatter plots of all events in selected planes,
highlighting those events passing thresholds on NN output.

This version of the function is used in the case that there
are k models and k datasets for k-fold cross testing.
"""
#################################################################


def AddPredictionsToScatter_nestedcrossval(data_set,
                                           predictions_set,
                                           axes_list=[[0,1]],
                                           rates = np.array([0.5,0.95,0.98,0.99]),
                                           colors=['silver','grey','khaki','goldenrod','firebrick'],
                                           axes_labels=None):

    if axes_labels == None:
        axes_labels = [[None,None] for axes in axes_list]
    points_list_init = False
    for data_i, predictions in enumerate(predictions_set):
        extended_rates = np.insert(rates,0,0.0)
        extended_rates = np.append(extended_rates,1.0)
        sorted_args = np.argsort(predictions)
        total_num = len(sorted_args)
        
        points_list_temp = [data_set[data_i][sorted_args[int(extended_rates[i] * total_num):int(extended_rates[i+1] * total_num)]] for i in range(0,len(extended_rates)-1)]

        if points_list_init:
            points_list = [np.append(points_list[i], points_list_temp[i],axis=0) for i in range(len(points_list_temp))]
        else:
            points_list = points_list_temp
            points_list_init = True
    
    
    plt.figure(figsize=(5*len(axes_list),5))
    size = 0.1
    for h, axes in enumerate(axes_list):
        plt.subplot(1, len(axes_list), h+1)
        for i, points in enumerate(points_list):
            size = 0.1
            plt.xlabel(axes_labels[h][0])
            plt.ylabel(axes_labels[h][1])
            if i == len(points_list)-1:
                size = 1.0
            plt.scatter(points[:,axes[0]],points[:,axes[1]],
                        s=size, color=colors[i])
    plt.show()
    
    return [rates]


#################################################################
################       check_eff      ###########################
#################################################################
"""
This class is a Keras custom callback which will keep track
of the signal region efficiency evaluated at fixed sideband
region efficiency. The class has two functions:

1) At the end of every epoch, it checks if this performance
   metric (evaluated on validation data) is better than the
   previous record. If so, it  will save the model, overwriting 
   the previous version. Only the current record-holder is kept.

2) It keeps track of this performance metric as well as loss,
   and plots these if requested.

"""
#################################################################

class check_eff(keras.callbacks.Callback):

    def __init__(self,
                 verbose=0,                        # Verbose = 1: print data af each epoch. Verbose = 2: Additionally save plots.
                 filename='checkpoint_best.h5',    # Model save filename
                 preprocessed_training_data=[],
                 period=1,                         # Epoch-frequency for checking performance
                 min_epoch=10,                     # Wait this no of epochs before saving best model
                 avg_length=20,                     # Period for plotting moving average of performance metric
                 preprocessor = None,              # Preprocessor for data before feeding to NN
                 eff_rate=0.02,                    # Threshold used to evaluate performance metric
                 patience=70,                      # Epochs to wait since last performance increase
                 plot_period=1,                    # How frequently to plot performance metric
                 batch_size=5000,                  # Batch size for NN prediction
                 max_epochs=2000,
                plotmode="save"):                 
        self.verbose = verbose
        self.filename = filename
        if avg_length%2 == 0:
            avg_length = avg_length+1
        self.avg_length = avg_length
        self.temp_weights = [None for i in range(int(round((self.avg_length-1)/2+1)))]
        self.training_data = preprocessed_training_data
        self.period = period
        self.min_epoch = min_epoch

        self.eff_rate = eff_rate
        self.patience = patience
        self.plot_period=plot_period
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.plotmode = plotmode
    
    def moving_average(self, a):
        n = self.avg_length
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.append(np.ones(int(round((n-1)/2)))*(ret[n - 1] / n),ret[n - 1:] / n)
    
    def on_train_begin(self, logs={}):
        self.effs_val = []
        self.effs_val_avg = []
        self.effs_train = []
        self.effs_train_avg = []
        self.loss = []
        self.val_loss = []
        self.n_wait = 0

        
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])

        if epoch%self.period==0:
            
            del self.temp_weights[0]
            self.temp_weights.append(self.model.get_weights())
            
            data = self.validation_data[0]
            my_pred = self.model.predict(data,batch_size=self.batch_size).flatten()
            my_true = self.validation_data[1].flatten()
            #Get the entries in category 0 ('bg')
            #Order these entries by NN prediction
            ordered_bg_list = np.flip(np.sort(my_pred[my_true<0.5]),axis=0)
            #Find the threshold above which only 2% of bg survives
            thresh = ordered_bg_list[int(self.eff_rate*len(ordered_bg_list))]
        
            #Now get entries in category 1 ('sig')
            #Order these by NN output
            ordered_sig_list = np.sort(my_pred[my_true>0.5]).flatten()
            #Find fraction of signal events which survive a cut on the above threshold
            sig_eff = 1.0 - 1.0*np.searchsorted(ordered_sig_list,thresh)/len(ordered_sig_list)

            if epoch > self.min_epoch:
                #Increase patience timer by one epoch
                self.n_wait = self.n_wait + 1
                
            if len(self.effs_val) == 0:
                self.model.save(self.filename)


            self.effs_val.append(sig_eff)
            if(len(self.effs_val) >= self.avg_length):
                self.effs_val_avg = self.moving_average(self.effs_val)
                
            # If this model is the best so far, save the model and reset the patience timer.
            if len(self.effs_val) > self.min_epoch and len(self.effs_val_avg)>self.min_epoch+1:
                if (self.effs_val_avg[-1] > self.effs_val_avg[self.min_epoch:-1].max()):
                    self.model.set_weights(self.temp_weights[0])
                    self.model.save(self.filename)
                    self.model.set_weights(self.temp_weights[-1])
                    self.n_wait = 0
               
            if(self.verbose>1):
                print("sig eff = ", sig_eff)

                
            if (self.verbose > 0) & (epoch % self.plot_period == 0):
                plt.figure(figsize=(14,5))
                plt.subplot(1, 2, 1)
                plt.plot(self.effs_val,color='C0',linestyle='--',alpha=0.5)
                if len(self.effs_val_avg) > 0:
                    plt.plot(self.moving_average(self.effs_val),color='C0',linestyle='-')

            if len(self.training_data) > 0:
                data = self.training_data[0]
                my_pred = self.model.predict(data,batch_size=self.batch_size).flatten()
                my_true = self.training_data[1].flatten()
                #Get the entries in category 0 ('bg')
                #Order these entries by NN prediction
                ordered_bg_list = np.flip(np.sort(my_pred[my_true<0.5]),axis=0)
                #Find the threshold above which only 2% of bg survives
                thresh = ordered_bg_list[int(self.eff_rate*len(ordered_bg_list))]
                
                #Now get entries in category 1 ('sig')
                #Order these by NN output
                ordered_sig_list = np.sort(my_pred[my_true>0.5]).flatten()
                #Find fraction of signal events which survive a cut on the above threshold
                sig_eff = 1.0 - 1.0*np.searchsorted(ordered_sig_list,thresh)/len(ordered_sig_list)
                self.effs_train.append(sig_eff)

                if(len(self.effs_train) <= self.avg_length):
                    self.effs_train_avg.append(np.mean(self.effs_train))
                else:
                    self.effs_train_avg.append(np.mean(np.array(self.effs_train)[-self.avg_length:]))
                
                if(self.verbose):
                    print("sig eff train = ", sig_eff)
                    
                if (self.verbose > 0) & (epoch % self.plot_period == 0) & epoch>0:
                    plt.plot(self.effs_train,color='C0')
                    if (self.avg_length > 1) and (len(self.effs_train) >= self.avg_length):
                        plt.plot(self.effs_train_avg,color='C0',linestyle='--')
                    plt.grid(b=True)

            # If we have waited too long with no improvement, halt training.
            if ((self.patience > 0) & (self.n_wait > self.patience)) or (epoch == self.max_epochs-1):
                if self.verbose > -1:
                    plt.close('all')
                    plt.figure(figsize=(14,5))
                    plt.subplot(1, 2, 1)
                    plt.plot(self.effs_val,color='C0',linestyle='--',alpha=0.5)
                    plt.plot(self.effs_train,color='C1',linestyle='--',alpha=0.5)
                    if(self.avg_length > 1):
                        plt.plot(self.effs_val_avg,color='C0')
                        plt.plot(self.effs_train_avg,color='C1')
                    plt.grid(b=True)
                    plt.xlabel('Epoch')
                    plt.ylabel('Sig. reg. eff at fixed sideband eff')
                    plt.subplot(1, 2, 2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.plot(self.val_loss,color='C0')
                    plt.plot(self.loss,color='C1')
                    plt.grid(b=True)
                    plt.tight_layout()
                    if self.plotmode == "save":
                        print("Saving fig:", self.filename[:-3] + "_losseffplots.png")
                        plt.savefig(self.filename[:-3] + "_losseffplots.png")
                    else:
                        plt.show()
                    self.verbose = 0
                self.model.stop_training = True


                
        if (self.verbose > 0) & (epoch % self.plot_period == 0) & (epoch > 0):
            plt.grid(b=True)
            plt.xlabel('Epoch')
            plt.ylabel('Sig. reg. eff at fixed sideband eff')
            plt.subplot(1, 2, 2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(self.val_loss,color='C0')
            plt.plot(self.loss,color='C1')
            plt.grid(b=True)
            plt.tight_layout()
            if self.plotmode == "save":
                plt.savefig(self.filename[:-3] + "_losseffplots.png")
            else:
                plt.show()




#################################################################
###########       print_scatter_checkpoint      #################
#################################################################
"""
This class is a Keras custom callback which will print scatter
plots of the most signal-like datapoints during training. Useful
monitor.

"""
#################################################################

class print_scatter_checkpoint(keras.callbacks.Callback):

    def __init__(self, verbose=0,
                 filename='epoch',
                 axes_list = [[0,1]],
                 axes_labels=None,
                 period=5,
                 batch_size = 5000,
                 training_data=[],
                 training_labels=[],
                 preprocess = None,
                 mode="print",
                 fig=None,
                 rates = np.array([0.5,0.95,0.99,0.998]),
                 colors=['silver','grey','khaki','goldenrod','firebrick']):
        self.verbose = verbose
        self.filename = filename
        self.axes_list = axes_list
        self.period = period
        self.training_data = training_data
        self.training_labels = training_labels
        self.mode=mode
        def null_preprocess(data):
            return data
        if preprocess == None:
            self.preprocess = null_preprocess
        else:
            self.preprocess = preprocess
        self.axes_labels=axes_labels
        self.batch_size = batch_size
        self.fig=fig
        self.rates=rates
        self.colors=colors
        
    def on_train_begin(self, logs={}):
        self.effs_val = []
            
    def on_epoch_end(self, epoch, logs={}):
        if epoch%self.period == 0:
            if len(self.training_data) > 0:
                data = self.training_data
                truth = self.training_labels
            else:
                data = self.validation_data[0]
                truth = self.validation_data[1].flatten()
                
            
            
            predictions = self.model.predict(self.preprocess(data),batch_size=self.batch_size).flatten()
            predictions_sideband = self.model.predict(self.preprocess(data[truth < 0.5]),batch_size=self.batch_size).flatten()
            
            plt.close('all')
            AddPredictionsToScatter(data, predictions,
                                    axes_list=self.axes_list,axes_labels=self.axes_labels,
                                    threshold_predictions = predictions_sideband,
                                   rates=self.rates, colors=self.colors,plotmode=None)
            plt.title("Epoch = " + str(epoch))
            if self.mode == "print":
                plt.savefig(self.filename + '_' + str(epoch) + '.png')
            else:
                plt.show()

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, norm, kstest
import numdifftools
from numpy.linalg import inv


#################################################################
###################       get_p_value      ######################
#################################################################

def get_p_value(ydata,binvals,mask=[],verbose=0,plotfile=None,yerr=None,return_teststat = False):
    
    ydata = np.array(ydata)
    #Assume poisson is gaussian with N+1 variance
    if not yerr:
        yerr = np.sqrt(ydata+1)
    else:
        yerr=np.array(yerr)
        
    def fit_func(x,p1,p2,p3):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        xi = 0.
        y = x/13000.
        return p1*(1.-y)**(p2-xi*p3)*y**-p3
    
    xdata = np.array([0.5*(binvals[i]+binvals[i+1]) for i in range(0,len(binvals)-1)])
    xwidths = np.array([-binvals[i]+binvals[i+1] for i in range(0,len(binvals)-1)])
    
    #Assuming inputs are bin counts, this is needed to get densities. Important for variable-width bins
    ydata = np.array(ydata) * 100 / xwidths
    yerr = np.array(yerr)*100/ np.array(xwidths)
    
    #Least square fit, masking out the signal region
    if len(mask) > 0:
        try:
            popt, pcov = curve_fit(fit_func, np.delete(xdata,mask),
                                   np.delete(ydata,mask),sigma=np.delete(yerr,mask),
                                   maxfev=3000,absolute_sigma=True)
        except:
            print("Failure")
            return
    else:
        try:
            popt, pcov = curve_fit(fit_func, xdata, ydata, sigma=yerr,maxfev=3000)
        except:
            print("Failure")
            return
    if verbose:
        print('fit params: ', popt)
        print('\n')
        
    ydata_fit = np.array([fit_func(x,popt[0],popt[1],popt[2]) for x in xdata])
    
    #Check that the function is a good fit to the sideband
    if verbose > 0:
        if len(mask) > 0:
            residuals = np.delete((ydata - ydata_fit)/yerr,mask)
        else:
            residuals = np.delete((ydata - ydata_fit)/yerr,mask)
        print("Goodness: ",kstest(residuals, norm(loc=0,scale=1).cdf))
        print('\n')
    
    if len(mask) == 0:
        pass
    
    #The following code is used to get the bin errors by propagating the errors on the fit params
    def fit_func_array(parr):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        p1, p2, p3 = parr
        xi = 0.
        return np.array([p1*(1.-(x/13000.))**(p2-xi*p3)*(x/13000.)**-p3 for x in xdata])
    
    jac=numdifftools.core.Jacobian(fit_func_array)
    x_cov=np.dot(np.dot(jac(popt),pcov),jac(popt).T)
    #For plot, take systematic error band as the diagonal of the covariance matrix
    y_unc=np.sqrt([row[i] for i, row in enumerate(x_cov)])
    
    if plotfile:
        lower_bound = ydata_fit-y_unc
        lower_bound[lower_bound<10**-4] = 10**-4
        plt.fill_between(xdata,ydata_fit+y_unc,lower_bound,color='gray',alpha=0.4)
        plt.errorbar(xdata, ydata,yerr,None, 'bo', label='data',markersize=4)
        plt.plot(xdata, ydata_fit, 'r--', label='data')
        plt.gca().set_yscale("log", nonposy='clip')
        plt.ylabel('Num events / 100 GeV')
        plt.xlabel('mJJ / GeV')
        
    #Now, let's compute some statistics.
    #  Will use asymptotic formulae for p0 from Cowan et al arXiv:1007.1727
    #  and systematics procedure from https://cds.cern.ch/record/2242860/files/NOTE2017_001.pdf

    # Note that in an eariler of this code, we were doing a 3-bin signal region shape fit.
    # We shifted to doing a single counting experiment, adding the three signal region bins together into a scalar value.
    # As a result of this shift, some scalars are confusingly represented as 1-dim arrays. It was
    # the quickest way to make the new code work without accidentally breaking something.
    
    #First get systematics in the signal region
    
    #This function returns array of signal predictions in the signal region
    def signal_fit_func_array(parr):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        p1, p2, p3 = parr
        xi = 0.
        return np.array([np.sum([p1*(1.-(x/13000.))**(p2-xi*p3)*(x/13000.)**-p3*xwidths[mask[i]]/100 for i, x in enumerate(xdata[mask])])])
    #Get covariance matrix of prediction uncertainties in the signal region
    jac=numdifftools.core.Jacobian(signal_fit_func_array)
    x_signal_cov=np.dot(np.dot(jac(popt),pcov),jac(popt).T)
    #Inverse signal region covariance matrix:
    inv_x_signal_cov = inv(x_signal_cov)
    
    #Get observed and predicted event counts in the signal region
    obs = np.array([np.sum(np.array(ydata)[mask]*np.array(xwidths)[mask]/100)])
    expected = np.array([np.sum([fit_func(xdata[targetbin],popt[0],popt[1],popt[2])*xwidths[targetbin]/100 for targetbin in mask])])
    
    #Negative numerator of log likelihood ratio, for signal rate mu = 0
    def min_log_numerator(expected_nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(expected_nuis_arr)
        to_return = 0
        #Poisson terms
        for i, expected_nuis in enumerate(expected_nuis_arr):
            #Poisson lambda. Have to rescale nuisance constribution by bin width
            my_lambda = expected[i]+expected_nuis_arr[i]
            #Prevent negative predicted rates
            if my_lambda < 10**-10:
                my_lambda = 10**-10
            #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
            to_return = to_return + (obs[i]*np.log(my_lambda) - my_lambda)
            
        #Gaussian nuisance term
        nuisance_term = -0.5*np.dot(np.dot(expected_nuis_arr,inv_x_signal_cov),expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return

    def jac_min_log_numerator(expected_nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(expected_nuis_arr)
        to_return = np.array([0.])
        #Poisson terms
        #Poisson lambda. Have to rescale nuisance constribution by bin width
        my_lambda = expected+expected_nuis_arr
        dmy_lambda = np.array([1.])
        #Prevent negative predicted rates
        my_lambda[my_lambda < 10**-10] = np.ones(len(my_lambda[my_lambda < 10**-10])) * 10**-10
        dmy_lambda[my_lambda < 10**-10] = 0
        #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
        to_return = to_return + (obs*dmy_lambda/my_lambda - dmy_lambda)
        #Gaussian nuisance term
        nuisance_term = -np.dot(inv_x_signal_cov,expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return
    
    #Initialization of nuisance params
    expected_nuis_array_init = [0.02]
    
    #shift log likelihood to heklp minimization algo
    def rescaled_min_log_numerator(expected_nuis_arr):
        return min_log_numerator(expected_nuis_arr) - min_log_numerator(expected_nuis_array_init)
    
    #Perform minimization over nuisance parameters. Set bounds for bg nuisance at around 8 sigma.
    bnds=[[-8*y_unc[mask[0]],8*y_unc[mask[0]]]]
    minimize_log_numerator = minimize(rescaled_min_log_numerator,
                                      expected_nuis_array_init,
                                      #jac=jac_min_log_numerator,
                                      bounds=bnds)
    
    if verbose:
        print("numerator: ",  minimize_log_numerator.items(),'\n')
        
    #Now get likelihood ratio denominator
    def min_log_denom(nuis_arr):
        #nuis_arr contains the bg systematics and also the signal rate
        expected_nuis_arr = np.array(nuis_arr)[:1]
        #print(expected_nuis_arr)
        mu = nuis_arr[1]
        #Signal prediction
        pred = [mu]
        to_return = 0
        #Poisson terms
        for i, expected_nuis in enumerate(expected_nuis_arr):
            #Poisson lambda
            my_lambda = expected[i]+expected_nuis_arr[i] + pred[i]
            #Prevent prediction from going negative
            if my_lambda < 10**-10:
                my_lambda = 10**-10
            #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
            to_return = to_return + (obs[i]*np.log(my_lambda) - my_lambda)

        #Gaussian nuisance term
        nuisance_term = -0.5*np.dot(np.dot(expected_nuis_arr,inv_x_signal_cov),expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return

    def jac_min_log_denom(nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(nuis_arr)[:1]
        mu = nuis_arr[1]
        pred = [mu]
        to_return_first = np.array([0.])
        #Poisson terms
        #Poisson lambda. Have to rescale nuisance constribution by bin width
        my_lambda = expected+expected_nuis_arr+pred
        dmy_lambda = np.array([1.])
        #Prevent prediction from going negative
        my_lambda[my_lambda < 10**-10] = np.ones(len(my_lambda[my_lambda < 10**-10])) * 10**-10
        dmy_lambda[my_lambda < 10**-10] = 0
        #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
        to_return_first = to_return_first + (obs*dmy_lambda/my_lambda - dmy_lambda)
        #Gaussian nuisance term
        nuisance_term = -np.dot(inv_x_signal_cov,expected_nuis_arr)
        to_return_first = to_return_first + nuisance_term
        
        to_return_last = np.array([0.])
        
        dpred = np.array([[1.]])
        
        my_lambda = expected+expected_nuis_arr+pred
        dmy_lambda = dpred
        to_return_last = np.dot((obs/my_lambda),dmy_lambda.T) - np.sum(dmy_lambda,axis=1)
        
        return -np.append(to_return_first, to_return_last)
    
    #initizalization for minimization
    nuis_array_init = [0.01,1.]
    
    #Shift log likelihood for helping minimization algo.
    def rescaled_min_log_denom(nuis_arr):
        return min_log_denom(nuis_arr) - min_log_denom(nuis_array_init)
    
    bnds = ((None,None),(None,None))
    minimize_log_denominator = minimize(rescaled_min_log_denom,nuis_array_init,
                                        #jac=jac_min_log_denom,
                                        bounds=bnds)
    
    if verbose:
        print("Denominator: ",  minimize_log_denominator.items(),'\n')
        
    if minimize_log_denominator.x[-1] < 0:
        Zval = 0
        neglognum = 0
        neglogden = 0
    else:
        neglognum = min_log_numerator(minimize_log_numerator.x)
        neglogden = min_log_denom(minimize_log_denominator.x)
        Zval = np.sqrt(2*(neglognum - neglogden))
      
    
    p0 = 1-norm.cdf(Zval)
    
    if verbose:
        print("z = ", Zval)
        print("p0 = ", p0)

    plt.title(str(p0))
    if plotfile == 'show':
        plt.show()
    elif plotfile:
        plt.savefig(plotfile)

    if return_teststat:
        return p0, 2*(neglognum - neglogden)
    else:
        return p0

