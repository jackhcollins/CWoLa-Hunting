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

class model_ensemble_nested_crossval:

    def __init__(self, dataset, bin_i=8, kfolds=5,eff_for_thresh = 0.02, no_mass=False, preprocess = None):
        self.kfolds = kfolds
        self.dataset_split = [self.get_kth_fold(dataset, k) for k in range(kfolds)]
        self.bin_i = bin_i
        def null_preprocess(data):
            return data
        if preprocess == None:
            self.preprocess = null_preprocess
        else:
            self.preprocess =preprocess
        self.eff_for_thresh = eff_for_thresh
        
        self.models = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.model_hists = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.effs_valid = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.effs_train = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.aucs_valid = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.aucs_train = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        
        self.predictions_valid = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.predictions_train = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        self.thresholds = [[[] for l in range(kfolds-1)] for k in range(kfolds)]
        
        self.no_mass = no_mass

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
    
    
    def add_model(self, model, model_hist, kfold, lfold):

        train_data, valid_data, categories_train, categories_valid, weights_train, weights_valid = self.get_trainval_data(kfold,lfold)

        if lfold > kfold:
            lfold = lfold - 1
        self.models[kfold][lfold].append(model)
        self.model_hists[kfold][lfold].append(model_hist)
        
        if self.no_mass:
            if self.preprocess != None:
                model_pred_valid = model.predict(self.preprocess(valid_data[:,2:]),batch_size=5000).flatten()
                model_pred_train = model.predict(self.preprocess(train_data[:,2:]),batch_size=5000).flatten()
            else:
                model_pred_valid = model.predict(valid_data[:,2:],batch_size=5000).flatten()
                model_pred_train = model.predict(train_data[:,2:],batch_size=5000).flatten()
        else:
            if self.preprocess != None:
                model_pred_valid = model.predict(self.preprocess(valid_data),batch_size=5000).flatten()
                model_pred_train = model.predict(self.preprocess(train_data),batch_size=5000).flatten()
            else:
                model_pred_valid = model.predict(valid_data,batch_size=5000).flatten()
                model_pred_train = model.predict(train_data,batch_size=5000).flatten()

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
        
        fpr, tpr, thresholds = roc_curve(categories_valid, model_pred_valid)
        #self.effs_valid[kfold][lfold].append(np.interp(np.log10(self.eff_for_thresh), np.log10(fpr), tpr))
        #self.thresholds[kfold][lfold].append(np.interp(np.log10(self.eff_for_thresh), np.log10(fpr), thresholds))
        #self.effs_train[kfold][lfold].append(np.sum((model_pred_train.T > self.thresholds[kfold][lfold][-1])*categories_train)/
        #                                      np.sum(categories_train))
        self.aucs_valid[kfold][lfold].append(roc_auc_score(categories_valid,model_pred_valid))
        self.aucs_train[kfold][lfold].append(roc_auc_score(categories_train,model_pred_train))

    def return_modelset_onek(self,k):
        model_indices = np.array([0 for i in range(self.kfolds-1)])
        for l in range(self.kfolds-1):
            model_indices[l] = np.argmax(self.effs_valid[k][l])
        return [self.models[k][l][model_indices[l]] for l in range(self.kfolds-1)]
            
    def avg_model_predict_onek(self, data, k):
        model_indices = np.array([0 for i in range(self.kfolds-1)])
        for l in range(self.kfolds-1):
            model_indices[l] = np.argmax(self.effs_valid[k][l])
        if self.no_mass:
            to_return = np.average(np.array([self.models[k][l][index].predict(self.preprocess(data[:,2:]),batch_size=5000).flatten() for l, index in enumerate(model_indices)]),axis=0)
        else:
            to_return = np.average(np.array([self.models[k][l][index].predict(self.preprocess(data),batch_size=5000).flatten() for l, index in enumerate(model_indices)]),axis=0)
        return to_return

    def avg_model_predict_lset(self, data, k):
        model_indices = np.array([0 for i in range(self.kfolds-1)])
        for l in range(self.kfolds-1):
            model_indices[l] = np.argmax(self.effs_valid[k,l])
        if self.no_mass:
            to_return = np.array([self.models[k][l][index].predict(self.preprocess(data[:,2:]),batch_size=5000).flatten() for l, index in enumerate(model_indices)])
        else:
            to_return = np.array([self.models[k][l][index].predict(self.preprocess(data),batch_size=5000).flatten() for l, index in enumerate(model_indices)])
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
    
class model_ensemble_crossval:

    def __init__(self, data, labels, eff_for_thresh = 0.02, no_mass=False):
        self.data = data
        self.labels = labels
        
        self.eff_for_thresh = eff_for_thresh

        self.train_labels = []
        self.valid_labels = []
        
        self.models = []
        self.model_hists = []
        self.effs_valid = []
        self.effs_train = []
        self.aucs_valid = []
        self.aucs_train = []
        
        self.predictions_valid = []
        self.predictions_train = []
        self.thresholds = []

        self.no_mass = no_mass

    def add_kfold(self, train_labels, valid_labels):
        self.train_labels.append(train_labels)
        self.valid_labels.append(valid_labels)
        self.models.append([])
        self.model_hists.append([])
        self.effs_valid.append([])
        self.effs_train.append([])
        self.aucs_valid.append([])
        self.aucs_train.append([])
        
        self.predictions_valid.append([])
        self.predictions_train.append([])
        self.thresholds.append([])
        
    def add_model(self, model, model_hist, kfold):
            
        self.models[kfold].append(model)
        self.model_hists[kfold].append(model_hist)

        if self.no_mass:
            model_pred_valid = model.predict(self.data[self.valid_labels[kfold]][:,2:],batch_size=5000)
            model_pred_train = model.predict(self.data[self.train_labels[kfold]][:,2:],batch_size=5000)
        else:
            model_pred_valid = model.predict(self.data[self.valid_labels[kfold]],batch_size=5000)
            model_pred_train = model.predict(self.data[self.train_labels[kfold]],batch_size=5000)

        self.predictions_valid[kfold].append(model_pred_valid)
        self.predictions_train[kfold].append(model_pred_train)
        
        fpr, tpr, thresholds = roc_curve(self.labels[self.valid_labels[kfold]], model_pred_valid)
        self.effs_valid[kfold].append(np.interp(np.log10(self.eff_for_thresh), np.log10(fpr), tpr))
        self.thresholds[kfold].append(np.interp(np.log10(self.eff_for_thresh), np.log10(fpr), thresholds))
        self.effs_train[kfold].append(np.sum((model_pred_train.T > self.thresholds[kfold][-1])*self.labels[self.train_labels[kfold]])/
                               np.sum(self.labels[self.train_labels[kfold]]))
        self.aucs_valid[kfold].append(roc_auc_score(self.labels[self.valid_labels[kfold]],model_pred_valid))
        self.aucs_train[kfold].append(roc_auc_score(self.labels[self.train_labels[kfold]],model_pred_train))

class avg_model_crossval:
    def __init__(self, model_ensemble):
        self.model_ensemble = model_ensemble
        self.no_mass = model_ensemble.no_mass
        self.kfolds = len(model_ensemble.models)
        print("Averaging over ", self.kfolds, " models.")
        model_list = []
        indices = []
        for kfold in range(0,self.kfolds):
            argbest = np.argmax(model_ensemble.effs_valid[kfold])
            model_list.append(model_ensemble.models[kfold][argbest])
            indices.append([kfold,argbest])
            
        self.indices = np.array(indices)
        self.model_list = np.array(model_list)
        
    def predict(self, data,batch_size=5000):
        if self.no_mass:
            to_return = np.average(np.array([model.predict(data[:,2:],batch_size=batch_size) for model in self.model_list]),axis=0)
        else:
            to_return = np.average(np.array([model.predict(data,batch_size=batch_size) for model in self.model_list]),axis=0)
        return to_return
    
    def predict_set(self, data,batch_size=5000):
        if self.no_mass:
            to_return = np.array([model.predict(data[:,2:],batch_size=batch_size) for model in self.model_list])
        else:
            to_return = np.array([model.predict(data,batch_size=batch_size) for model in self.model_list])
        return to_return

    
class model_ensemble:

    def __init__(self, data_train, labels_train, data_valid, labels_valid, eff_for_thresh = 0.02):
        self.data_train = data_train
        self.data_valid = data_valid
        self.labels_train = labels_train
        self.labels_valid = labels_valid
        
        self.eff_for_thresh = eff_for_thresh
        
        self.ntries = 0
        self.models = list()
        self.model_hists = list()
        self.effs_valid = list()
        self.effs_train = list()
        self.aucs_valid = list()
        self.aucs_train = list()
        
        self.predictions_valid = list()
        self.predictions_train = list()
        self.thresholds = list()
        
    def add_model(self, model, model_hist):
        self.ntries = self.ntries + 1
        self.models.append(model)
        self.model_hists.append(model_hist)
        
        model_pred_valid = model.predict(self.data_valid,batch_size=5000)
        self.predictions_valid.append(model_pred_valid)
        model_pred_train = model.predict(self.data_train,batch_size=5000)
        
        fpr, tpr, thresholds = roc_curve(self.labels_valid, model_pred_valid)
        self.effs_valid.append(np.interp(np.log10(self.eff_for_thresh), np.log10(fpr), tpr))
        self.thresholds.append(np.interp(np.log10(self.eff_for_thresh), np.log10(fpr), thresholds))
        self.effs_train.append(np.sum((model_pred_train.T > self.thresholds[-1])*self.labels_train)/
                               np.sum(self.labels_train))
        self.aucs_valid.append(roc_auc_score(self.labels_valid,model_pred_valid))
        self.aucs_train.append(roc_auc_score(self.labels_train,model_pred_train))

def sideband_signal_split(data, bin_no, valid_frac = 0.2):

    train_frac = 1 - valid_frac

    low_sideband = np.append(data[bin_no-2],data[bin_no-3],axis=0)
    high_sideband = np.append(data[bin_no+2],data[bin_no+3],axis=0)
    signal = np.append(data[bin_no-1],data[bin_no], axis=0)
    signal = np.append(signal,data[bin_no+1],axis=0)
    
    n_low = len(low_sideband)
    n_high = len(high_sideband)
    n_signal = len(signal)
    n_total = n_low + n_high + n_signal
    
    all_data = np.append(low_sideband,
                         signal,
                         axis=0)
    all_data = np.append(all_data,
                         high_sideband,
                         axis=0)
    
    strat_categories = np.append(np.zeros(n_low), np.ones(n_signal))
    strat_categories = np.append(strat_categories, np.ones(n_high)*2)
    
    categories = np.append(np.zeros(n_low), np.ones(n_signal))
    categories = np.append(categories, np.zeros(n_high))
    
    weights = np.append(np.ones(n_low)*0.25*n_total/n_low,
                        np.ones(n_signal)*0.5*n_total/n_signal)
    weights = np.append(weights,
                        np.ones(n_high)*0.25*n_total/n_high)
    
    return train_test_split(all_data[:,1:], categories, weights, test_size=0.25, random_state=44, stratify=strat_categories, shuffle=True)

from sklearn.model_selection import StratifiedKFold
def sideband_signal_split_crossval(data, bin_no, kfolds=4):

    low_sideband = np.append(data[bin_no-2],data[bin_no-3],axis=0)
    high_sideband = np.append(data[bin_no+2],data[bin_no+3],axis=0)
    signal = np.append(data[bin_no-1],data[bin_no], axis=0)
    signal = np.append(signal,data[bin_no+1],axis=0)
    
    n_low = len(low_sideband)
    n_high = len(high_sideband)
    n_signal = len(signal)
    n_total = n_low + n_high + n_signal
    
    all_data = np.append(low_sideband,
                         signal,
                         axis=0)
    all_data = np.append(all_data,
                         high_sideband,
                         axis=0)
    
    strat_categories = np.append(np.zeros(n_low), np.ones(n_signal))
    strat_categories = np.append(strat_categories, np.ones(n_high)*2)

    categories = np.append(np.zeros(n_low), np.ones(n_signal))
    categories = np.append(categories, np.zeros(n_high))
    
    weights = np.append(np.ones(n_low)*0.25*n_total/n_low,
                        np.ones(n_signal)*0.5*n_total/n_signal)
    weights = np.append(weights,
                        np.ones(n_high)*0.25*n_total/n_high)
    
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
    
    return all_data[:,1:], categories, weights, skf.split(all_data, strat_categories)

def get_kth_fold(binned_data,k,kfolds=5,bins=None):
    if bins:
        return [binned_data[i][int(len(binned_data[i])*k/kfolds):int(len(binned_data[i])*(k+1)/kfolds)] for i in bins]
    else:
        return [databin[int(len(databin)*k/kfolds):int(len(databin)*(k+1)/kfolds)] for databin in binned_data]

def get_train_data(binned_data, k, l, kfolds=5,bins=None):
    #Get all data except the ones corresponding to k and l
    if bins:
        dataset = [binned_data[i][0:int(min(k,l)*len(binned_data[i])/kfolds)] for i in bins]
        dataset = [np.append(dataset[j],binned_data[i][int(min(k,l)*len(binned_data[i])/kfolds):int(max(k,l)*len(binned_data[i]/kfolds))],axis=0)
                   for j, i in enumerate(bins)]
        dataset = [np.append(dataset[j],binned_data[i][int(max(k,l)*len(binned_data[i])/kfolds):],axis=0)
                   for j, i in enumerate(bins)]
    else:
        dataset = [databin[0:int(min(k,l)*len(databin)/kfolds)] for databin in binned_data]
        dataset = [np.append(dataset[i],databin[int(min(k,l)*len(databin)/kfolds):int(max(k,l)*len(databin)/kfolds)],axis=0)
                   for i, databin in enumerate(binned_data)]
        dataset = [np.append(dataset[i],databin[int(max(k,l)*len(databin)/kfolds):],axis=0)
                   for i, databin in enumerate(binned_data)]
    return dataset
    
def sideband_signal_split_nested_crossval(data, bin_no, k, lfold, kfolds=5):
                   
    low_sideband_train_bins = get_train_data(data,k,lfold,kfolds,bins=[bin_no-3,bin_no-2])
    low_sideband_train = np.append(low_sideband_train_bins[0],
                                   low_sideband_train_bins[1],
                                   axis=0)
    high_sideband_train_bins = get_train_data(data,k,lfold,kfolds,bins=[bin_no+3,bin_no+2])
    high_sideband_train = np.append(high_sideband_train_bins[0],
                                    high_sideband_train_bins[1],
                                    axis=0)
    signal_train_bins = get_train_data(data,k,lfold,kfolds,bins=[bin_no-1,bin_no, bin_no+1])
    signal_train = np.concatenate((signal_train_bins[0],
                                   signal_train_bins[1],
                                   signal_train_bins[2]),
                                  axis=0)

    low_sideband_valid_bins = get_kth_fold(data,lfold,kfolds,bins=[bin_no-3,bin_no-2])
    low_sideband_valid = np.append(low_sideband_valid_bins[0],
                                   low_sideband_valid_bins[1],
                                   axis=0)
    high_sideband_valid_bins = get_kth_fold(data,lfold,kfolds,bins=[bin_no+3,bin_no+2])
    high_sideband_valid = np.append(high_sideband_valid_bins[0],
                                    high_sideband_valid_bins[1],
                                    axis=0)
    signal_valid_bins = get_kth_fold(data,lfold,kfolds,bins=[bin_no-1,bin_no, bin_no+1])
    signal_valid = np.concatenate((signal_valid_bins[0],
                                   signal_valid_bins[1],
                                   signal_valid_bins[2]),
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
    weights_valid = np.append(weights_train,
                              np.ones(len(high_sideband_valid))*0.25*n_total/n_high)
    
    return train_data, valid_data, categories_train, categories_valid, weights_train, weights_valid
"""
def AddPredictionsToScatter(data, predictions,axes_list=[[0,1]],axes_labels=None):

    if axes_labels == None:
        axes_labels = [[None,None] for axes in axes_list]

    fpr, tpr, thresholds = roc_curve(np.append(np.zeros(len(predictions)), np.ones(len(predictions))),
                                     np.append(predictions, predictions))
    
    rates = np.log10(np.array([0.5,0.05,0.02,0.01]))
    threshold_list = np.array([np.interp(rate, np.log10(fpr), thresholds) for rate in rates])
    
    extended_threshold_list = np.append(threshold_list,1.01)
    extended_threshold_list = np.insert(extended_threshold_list,0,0)
    
    points_list = np.array([data[(predictions > extended_threshold_list[i]) & (predictions <= extended_threshold_list[i+1])]
                            for i in range(0,len(extended_threshold_list)-1)])
    
    colorlist = np.linspace(0,1,len(rates)+1)
    colors = [plt.get_cmap('YlOrRd')(colornum) for colornum in colorlist]
    colors = ['silver','grey','khaki','goldenrod','firebrick']
    
    plt.figure(figsize=(7*len(axes_list),5))
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
    plt.show()
        
    return [rates, threshold_list]
"""
def AddPredictionsToScatter(data, predictions,axes_list=[[0,1]],axes_labels=None,
                            rates = np.array([0.5,0.95,0.98,0.99]),
                            colors=['silver','grey','khaki','goldenrod','firebrick']):

    if axes_labels == None:
        axes_labels = [[None,None] for axes in axes_list]
        
    extended_rates = np.insert(rates,0,0.0)
    extended_rates = np.append(extended_rates,1.0)
    sorted_args = np.argsort(predictions)
    total_num = len(sorted_args)
    points_list = np.array([data[sorted_args[int(extended_rates[i] * total_num):int(extended_rates[i+1] * total_num)]]
                            for i in range(0,len(extended_rates)-1)])
    
    plt.figure(figsize=(7*len(axes_list),5))
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
    plt.show()
                
    return [rates]


def AddPredictionsToScatter_nestedcrossval(data_set, predictions_set, axes_list=[[0,1]],
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
    
    
    plt.figure(figsize=(7*len(axes_list),5))
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

class check_eff(keras.callbacks.Callback):

    def __init__(self, verbose=0, filename='checkpoint_best.h5',patience=0,training_data=[],period=1,min_epoch=0,avg_length=5,
                 moving_avg_mode = False, no_mass = False, eff_rate=0.02,plot_period=1):
        self.verbose=verbose
        self.filename=filename
        self.patience=patience
        self.training_data=training_data
        self.period=period
        self.min_epoch=min_epoch
        self.avg_length=avg_length
        self.moving_avg_mode=moving_avg_mode
        self.no_mass = no_mass
        self.eff_rate = eff_rate
        self.plot_period=plot_period
        
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
            data = self.validation_data[0]
            my_pred = self.model.predict(data,batch_size=5000).flatten()
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
                self.n_wait = self.n_wait + 1
            if len(self.effs_val) == 0:
                self.model.save(self.filename)
            elif len(self.effs_val) > self.min_epoch:
                if (sig_eff >= np.array(self.effs_val)[self.min_epoch:].max()):
                    self.model.save(self.filename)
                    if not self.moving_avg_mode:
                        self.n_wait=0

            self.effs_val.append(sig_eff)
            if(len(self.effs_val) <= self.avg_length):
                self.effs_val_avg.append(np.mean(self.effs_val))
            else:
                self.effs_val_avg.append(np.mean(np.array(self.effs_val)[-self.avg_length:]))
                if (self.effs_val_avg[-1] > np.array(self.effs_val_avg[:-1]).max()) & self.moving_avg_mode:
                    self.n_wait=0
               
            if(self.verbose):
                print("sig eff = ", sig_eff)

            if (self.verbose > 1) & (epoch % self.plot_period == 0):
                plt.figure(figsize=(14,5))
                plt.subplot(1, 2, 1)
                plt.plot(self.effs_val,color='C1')
                if(self.avg_length > 1):
                    plt.plot(self.effs_val_avg,color='C1',linestyle='--')

            if len(self.training_data) > 0:
                if self.no_mass:
                    data = self.training_data[0][:,2:]
                else:
                    data = self.training_data[0]
                my_pred = self.model.predict(data,batch_size=5000).flatten()
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
                    
                if (self.verbose > 1) & (epoch % self.plot_period == 0):
                    plt.plot(self.effs_train,color='C0')
                    if(self.avg_length > 1):
                        plt.plot(self.effs_train_avg,color='C0',linestyle='--')
                    plt.grid(b=True)
                    
            if (self.patience > 0) & (self.n_wait > self.patience):
                if self.moving_avg_mode:
                    self.model.save(self.filename)
                self.model.stop_training = True
            
        if (self.verbose > 1) & (epoch % self.plot_period == 0):
            plt.subplot(1, 2, 2)
            plt.plot(self.val_loss,color='C1')
            plt.plot(self.loss,color='C0')
            plt.grid(b=True)
            plt.savefig(self.filename[:-3] + "_losseffplots.png")

                
class print_scatter_checkpoint(keras.callbacks.Callback):

    def __init__(self, verbose=0, filename='epoch',axes_list = [[0,1]],axes_labels=None,period=5,training_data=[], no_mass = False, preprocess = None):
        self.verbose=verbose
        self.filename=filename
        self.axes_list = axes_list
        self.period = period
        self.training_data=training_data
        self.no_mass = no_mass
        self.preprocess = preprocess
        self.axes_labels=axes_labels
        
    def on_train_begin(self, logs={}):
        self.effs_val = []
            
    def on_epoch_end(self, epoch, logs={}):
        if epoch%self.period == 0:
            if len(self.training_data) > 0:
                data = self.training_data
            else:
                data = self.validation_data[0]
            if self.no_mass:
                if self.preprocess != None:
                    predictions = self.model.predict(self.preprocess(data[:,2:],batch_size=5000)).flatten()
                else:
                    predictions = self.model.predict(data[:,2:],batch_size=5000).flatten()
            else:
                if self.preprocess != None:
                    predictions = self.model.predict(self.preprocess(data),batch_size=5000).flatten()
                else:
                    predictions = self.model.predict(data,batch_size=5000).flatten()
            
            plt.close('all')
            AddPredictionsToScatter(data, predictions,axes_list=self.axes_list,axes_labels=self.axes_labels)
            #plt.title(epoch)
            plt.savefig(self.filename + '_' + str(epoch) + '.png')

class avg_model:
    def __init__(self, model_ensemble, navg = None):
        if navg == None:
            print("No navg set. Will set to half of ", model_ensemble.ntries)
            navg = int(round(model_ensemble.ntries/2))
        print("Averaging over ", navg, " models.")
        self.ordering = np.flip(np.argsort(model_ensemble.effs_valid),axis=0)
        self.model_list = np.array(model_ensemble.models)[self.ordering[:navg]]
            
    def predict(self, data,batch_size=5000):
        return np.average(np.array([model.predict(data,batch_size=batch_size) for model in self.model_list]),axis=0)
            
    def predict_set(self, data,batch_size=5000):
        return np.array([model.predict(data,batch_size=batch_size) for model in self.model_list])
            
    def return_ordering(self):
        return np.argsort(self.model_ensemble.effs_valid)
            

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, norm, kstest
import numdifftools
from numpy.linalg import inv

def get_p_value(ydata,binvals,mask=[],verbose=0,plotfile=None,yerr=None):
    
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
    popt, pcov = curve_fit(fit_func, np.delete(xdata,mask), np.delete(ydata,mask),sigma=np.delete(yerr,mask),maxfev=3000)
    if verbose:
        print('fit params: ', popt)
        
    ydata_fit = np.array([fit_func(x,popt[0],popt[1],popt[2]) for x in xdata])
    
    #Check that the function is a good fit to the sideband
    residuals = np.delete((ydata - ydata_fit)/yerr,mask)
    print("Goodness: ",kstest(residuals, norm(loc=0,scale=1).cdf))
    
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
        plt.fill_between(xdata,ydata_fit+y_unc,ydata_fit-y_unc,color='gray',alpha=0.4)
        plt.errorbar(xdata, ydata,yerr,None, 'bo', label='data',markersize=4)
        plt.plot(xdata, ydata_fit, 'r--', label='data')
        plt.semilogy()
        plt.ylabel('Num events / 100 GeV')
        plt.xlabel('mJJ / GeV')
        
    #Now, let's compute some statistics.
    #  Will use asymptotic formulae for p0 from Cowan et al arXiv:1007.1727
    #  and systematics procedure from https://cds.cern.ch/record/2242860/files/NOTE2017_001.pdf
    
    #First get systematics in the signal region
    
    #This function returns array of signal predictions in the signal region
    def signal_fit_func_array(parr):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        p1, p2, p3 = parr
        xi = 0.
        return np.array([p1*(1.-(x/13000.))**(p2-xi*p3)*(x/13000.)**-p3 for x in xdata[mask]])
    #Get covariance matrix of prediction uncertainties in the signal region
    jac=numdifftools.core.Jacobian(signal_fit_func_array)
    x_signal_cov=np.dot(np.dot(jac(popt),pcov),jac(popt).T)
    #Inverse signal region covariance matrix:
    inv_x_signal_cov = inv(x_signal_cov)
    
    #Get observed and predicted event counts in the signal region
    obs = np.array(ydata)[mask]*np.array(xwidths)[mask]/100
    expected = np.array([fit_func(xdata[targetbin],popt[0],popt[1],popt[2])*xwidths[targetbin]/100 for targetbin in mask])
    
    myvec=xwidths[mask]/100 * (1 + obs/expected)
    optimal_nuis = np.dot(x_signal_cov,myvec)
    print("Optimal nuisance = ", optimal_nuis)
    
    #Negative numerator of log likelihood ratio, for signal rate mu = 0
    def min_log_numerator(expected_nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(expected_nuis_arr)
        to_return = 0
        #Poisson terms
        for i, expected_nuis in enumerate(expected_nuis_arr):
            #Poisson lambda. Have to rescale nuisance constribution by bin width
            my_lambda = expected[i]+expected_nuis_arr[i]*xwidths[mask][i]/100
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
        to_return = np.array([0.,0.,0.])
        #Poisson terms
        #Poisson lambda. Have to rescale nuisance constribution by bin width
        my_lambda = expected+expected_nuis_arr*xwidths[mask]/100
        dmy_lambda = xwidths[mask]/100
        #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
        to_return = to_return + (obs*dmy_lambda/my_lambda - dmy_lambda)
        #Gaussian nuisance term
        nuisance_term = -np.dot(inv_x_signal_cov,expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return
    
    #Initialization of nuisance params
    expected_nuis_array_init = [0.01,-0.01,0.02]
    
    #shift log likelihood to heklp minimization algo
    def rescaled_min_log_numerator(expected_nuis_arr):
        return min_log_numerator(expected_nuis_arr) - min_log_numerator(expected_nuis_array_init)
    
    #Perform minimization over nuisance parameters
    bnds=((-10*y_unc[mask[0]],10*y_unc[mask[0]]),(-10*y_unc[mask[1]],10*y_unc[mask[1]]),(-10*y_unc[mask[2]],10*y_unc[mask[2]]))
    minimize_log_numerator = minimize(rescaled_min_log_numerator,expected_nuis_array_init,
                                      jac=jac_min_log_numerator,
                                      bounds=bnds)
    #bounds=bnds)
    
    if verbose:
        print("numerator: ",  minimize_log_numerator.items(),'\n')
        
    #Now get likelihood ratio denominator
    def min_log_denom(nuis_arr):
        #nuis_arr contains the bg systematics and also the signal rate
        expected_nuis_arr = np.array(nuis_arr)[:3]
        #print(expected_nuis_arr)
        f1 = nuis_arr[3]
        f2 = nuis_arr[4]
        mu = nuis_arr[5]
        #Signal prediction
        pred = [f1*mu,f2*(1-f1)*mu,(1-f1-f2*(1-f1))*mu]
        to_return = 0
        #Poisson terms
        for i, expected_nuis in enumerate(expected_nuis_arr):
            #Poisson lambda
            my_lambda = expected[i]+expected_nuis_arr[i]*xwidths[mask][i]/100+pred[i]
            #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
            to_return = to_return + (obs[i]*np.log(my_lambda) - my_lambda)

        #Gaussian nuisance term
        nuisance_term = -0.5*np.dot(np.dot(expected_nuis_arr,inv_x_signal_cov),expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return

    def jac_min_log_denom(nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(nuis_arr)[:3]
        f1 = nuis_arr[3]
        f2 = nuis_arr[4]
        mu = nuis_arr[5]
        pred = [f1*mu,f2*(1-f1)*mu,(1-f1-f2*(1-f1))*mu]
        to_return_first = np.array([0.,0.,0.])
        #Poisson terms
        #Poisson lambda. Have to rescale nuisance constribution by bin width
        my_lambda = expected+expected_nuis_arr*xwidths[mask]/100+pred
        dmy_lambda = xwidths[mask]/100
        #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
        to_return_first = to_return_first + (obs*dmy_lambda/my_lambda - dmy_lambda)
        #Gaussian nuisance term
        nuisance_term = -np.dot(inv_x_signal_cov,expected_nuis_arr)
        to_return_first = to_return_first + nuisance_term
        
        to_return_last = np.array([0.,0.,0.])
        
        dpred = np.array([[mu,-f2*mu,(-1+f2)*mu],
                          [0,(1-f1)*mu,(-1+f1)*mu],
                          [f1,f2*(1-f1),1-f1-f2*(1-f1)]])
        
        my_lambda = expected+expected_nuis_arr*xwidths[mask]/100+pred
        dmy_lambda = dpred
        to_return_last = np.dot((obs/my_lambda),dmy_lambda.T) - np.sum(dmy_lambda,axis=1)
        
        return -np.append(to_return_first, to_return_last)
    
    #initizalization for minimization
    nuis_array_init = [0.01,0.02,0.01,0.33,0.5,1]
    
    #Shift log likelihood for helping minimization algo.
    def rescaled_min_log_denom(nuis_arr):
        return min_log_denom(nuis_arr) - min_log_denom(nuis_array_init)
    
    bnds = ((None,None),(None,None),(None,None),(0,1),(0,1),(None,None))
    minimize_log_denominator = minimize(rescaled_min_log_denom,nuis_array_init,bounds=bnds,
                                        jac=jac_min_log_denom)
    
    if verbose:
        print("Denominator: ",  minimize_log_denominator.items(),'\n')
        
    if minimize_log_denominator.x[-1] < 0:
        Zval = 0
    else:
        neglognum = min_log_numerator(minimize_log_numerator.x)
        neglogden = min_log_denom(minimize_log_denominator.x)
        Zval = np.sqrt(2*(neglognum - neglogden))
        
    print("z = ", Zval)
    p0 = 1-norm.cdf(Zval)
    print("p0 = ", p0)

    plt.title(str(p0))
    if plotfile == 'show':
        plt.show()
    elif plotfile:
        plt.savefig(plotfile)

    return p0

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
