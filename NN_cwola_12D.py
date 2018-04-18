from __future__ import print_function
#import os
import gc
import sys
import numpy as np
import numpy.random as rand
import math
import matplotlib
matplotlib.use('Agg')
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
#import pickle as pickle
import tensorflow as tf
import time
import glob
import numpy.ma as ma
from keras import backend as K

from cwola_utils import AddPredictionsToScatter_nestedcrossval
from cwola_utils import model_ensemble
from cwola_utils import check_eff
from cwola_utils import print_scatter_checkpoint
from cwola_utils import get_p_value

rand.seed(0)
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

from tensorflow.python.client import device_lib
print("Devices: ", device_lib.list_local_devices())

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

myargs = getopts(sys.argv)
print("Command line arguments: ", myargs)
if '-o' not in myargs:
    print("Setting default output name '-o': '~/model'")
    myargs['-o'] = "~/model"
if '-in' not in myargs:
    print("Setting default input directory to '/data1/users/jcollins/'")
    myargs['-in'] = '/data1/users/jcollins/'
if '-bin' not in myargs:
    myargs['-bin'] = 7
    print("Setting default bin '-bin':", myargs['-bin'])
else:
    myargs['-bin'] = int(myargs['-bin'])
if '-it' not in myargs:
    myargs['-it'] = 3
    print("Setting default  '-it':", myargs['-it'])
else:
    myargs['-it'] = int(myargs['-it'])
    
if '-kfold' not in myargs:
    myargs['-kfold'] = 5
    print("Setting default  '-kfold':",myargs['-kfold'])
else:
    myargs['-kfold'] = int(myargs['-kfold'])
    
if '-signal' not in myargs:
    myargs['-signal'] = 1
    print("Setting default  '-signal':",myargs['-signal'])
else:
    myargs['-signal'] = int(myargs['-signal'])
    
if '-sigevnts' not in myargs:
    myargs['-sigevnts'] = 1125
    print("Setting default  '-sigevnts':",myargs['-sigevnts'])
else:
    myargs['-sigevnts'] = int(myargs['-sigevnts'])
if '-bgevnts' not in myargs:
    myargs['-bgevnts'] = 1184900
    print("Setting default  '-bgevnts':", myargs['-bgevnts'])
else:
    myargs['-bgevnts'] = int(myargs['-bgevnts'])
if '-bgoffset' not in myargs:
    myargs['-bgoffset'] = 0
    print("Setting default  '-bgoffset':", myargs['-bgoffset'])
else:
    myargs['-bgoffset'] = int(myargs['-bgoffset'])

    
if '-checkeff' not in myargs:
    myargs['-checkeff'] = 0.01
    print("Setting default  '-checkeff':", myargs['-checkeff'])
else:
    myargs['-checkeff'] = float(myargs['-checkeff'])

if '-loadonly' not in myargs:
    myargs['-loadonly'] = 0
    print("Setting default  '-loadonly':", myargs['-loadonly'])
else:
    myargs['-loadonly'] = float(myargs['-loadonly'])
        
bin_i = myargs['-bin']    
data_prefix = myargs['-in']
output_prefix = myargs['-o']

selected_vars = [6,33,10,37,12,39,15,42,18,45,32,59]
selected_vars_plus = np.append([0],selected_vars)

#Which 2d planes to make scatter plots in
axes_list = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[0,2],[0,4],[0,6],[0,8],[0,10],[1,3],[1,5],[1,7],[1,9],[1,11]]
axes_labels = [['mJA','mJB'],['tau_1211A','tau1211B'],['tau_21A','tau21B'],['tau_32A','tau32B'],['tau_43A','tau43B'],['ntrkA','ntrkB'],
               ['mJA','tau1211A'],['mJA','tau21A'],['mJA','tau32A'],['mJA','tau43A'],['mJA','ntrkA'],
               ['mJB','tau1211B'],['mJB','tau21B'],['mJB','tau32B'],['mJB','tau43B'],['mJB','ntrkB']]

#Data binning in mJJ
mjjmin = 2001
mjjmax = 4350
mybinboundaries = np.round(np.logspace(np.log10(mjjmin), np.log10(mjjmax), num=16))
mybincenters = np.array([0.5*(mybinboundaries[i+1] + mybinboundaries[i]) for i in range(0,len(mybinboundaries)-1)])
mybinwidths = np.array([mybinboundaries[i+1] - mybinboundaries[i] for i in range(0,len(mybinboundaries)-1)])
mybincenterandwidth = np.vstack((mybincenters,mybinwidths)).T

def bin_data(data, binboundaries = mybinboundaries):
    databinned = []
    for i in range(0,len(binboundaries)-1):
        databinned.append(
            np.array([np.delete(myrow,0) for myrow in data if (myrow[0] < binboundaries[i+1] and myrow[0] >= binboundaries[i])])
        )
    return databinned

#Load bg data
print('\n')
numfiles = int((myargs['-bgoffset'] + myargs['-bgevnts'])/100000) + 1
print("Loading ", myargs['-in'] + 'new_dataset_0.dat')
bgdata = np.loadtxt(myargs['-in'] + 'new_dataset_0.dat')
for i in range(1,numfiles):
    filename = myargs['-in'] + 'new_dataset_' + str(i) + '.dat'
    print("Loading ", filename)
    bgdata = np.append(bgdata,np.loadtxt(filename),axis=0)
bgdata = bgdata[myargs['-bgoffset']:myargs['-bgoffset']+myargs['-bgevnts']]
bgdata = bgdata[bgdata[:,0]>=mjjmin]
rand.shuffle(bgdata)

#Load signal data
if myargs['-signal']:
    print('Adding signal data')
    signaldata = np.loadtxt(myargs['-in'] + 'W_WW_3000_400.dat')[:myargs['-sigevnts']]
    signaldata = signaldata[:,1:]
    signaldata = signaldata[signaldata[:,0]>=mjjmin]
    bg_plus_signal = np.nan_to_num(np.append(signaldata,bgdata,axis=0))
    rand.shuffle(bg_plus_signal)

    bghist = np.histogram(bgdata[:,0],bins=mybinboundaries)[0]
    sighist = np.histogram(signaldata[:,0],bins=mybinboundaries)[0]

    B = bghist[bin_i-1] + bghist[bin_i] + bghist[bin_i+1]
    S = sighist[bin_i-1] + sighist[bin_i] + sighist[bin_i+1]
    print("\nIn Signal Region:")
    print("S = ", S)
    print("B = ", B)
    print("S/B =", '%.3f' % (float(S)/float(B)))
    print("S/sqrt(B)", '%.3f' % (float(S)/np.sqrt(float(B))))
    print("\n")
else:
    print('Not adding signal data')
    bg_plus_signal = np.nan_to_num(bgdata)

#Create tau ratios
bg_plus_signal[:,[18,45]] = np.nan_to_num(bg_plus_signal[:,[18,45]]/bg_plus_signal[:,[15,42]])
bg_plus_signal[:,[15,42]] = np.nan_to_num(bg_plus_signal[:,[15,42]]/bg_plus_signal[:,[12,39]])
bg_plus_signal[:,[12,39]] = np.nan_to_num(bg_plus_signal[:,[12,39]]/bg_plus_signal[:,[9,36]])
bg_plus_signal[:,[10,37]] = np.nan_to_num(bg_plus_signal[:,[10,37]]/bg_plus_signal[:,[9,36]])
#Cut mJJ outliers
bg_plus_signal = bg_plus_signal[(bg_plus_signal[:,6] < 500) & (bg_plus_signal[:,33] < 300)]


#Preprocessor for data before feeding it to NN
def myprepreprocessor(predata, massvars=None):
    if massvars is None:
        massvars = [0,1]
    newdata = np.copy(predata)
    newdata[:,massvars] = np.nan_to_num(np.log10(newdata[:,massvars]+40))
    return newdata


bg_plus_signal=bg_plus_signal[:,selected_vars_plus]
myscaler = preprocessing.StandardScaler().fit(myprepreprocessor(bg_plus_signal,massvars=[1,2])[:,1:])
def preprocess(data):
    return np.clip(myscaler.transform(myprepreprocessor(data,massvars=[0,1])),-3,3)
rand.shuffle(bg_plus_signal)

bg_plus_signal_binned = bin_data(bg_plus_signal)
del bg_plus_signal

import warnings
warnings.filterwarnings('ignore')

ntries = myargs['-it']
times = list()
print("Working on bin ", bin_i)

no_mass = False
numvars=len(selected_vars)
if no_mass:
    numvars = numvars-2

#I have found that large batch sizes work best for our problem. I have not done a systematic test. But my intuition is that because signal events are so rare, even in the signal region, you want batch sizes large enough so that most batches will contain at least a few true signal events.
batch_size =5000

model_utils = {}
model_utils[bin_i] = model_ensemble(bg_plus_signal_binned, bin_i = bin_i, kfolds=myargs['-kfold'], preprocess = preprocess, eff_for_thresh = myargs['-checkeff'])

#Loop over kfolds
for k in range(myargs['-kfold']):
    print('Starting kfold', k, 'of', myargs['-kfold'])
    #Loop over validation sets
    for l in range(myargs['-kfold']):
        if l == k:
            continue
        print('Starting lfold', l, 'of', myargs['-kfold'])
        
        data_train, data_valid, labels_train, labels_valid, weights_train, weights_valid = model_utils[bin_i].get_trainval_data(k,l)
        
        for i in range(ntries):
            print(" k =", k, "l =", l)
            #Naming convention for model files.
            checkpoint_name = output_prefix + "_" + str(bin_i) + "_[" + str(k) + "," + str(l) + "]_" + str(i)
            start = time.time()

            #The flag myargs['-loadonly'] allows to skip training step and just load pre-saved models.
            if not myargs['-loadonly']:
                print("Now training model ", i + 1, " of ", ntries)
                #Following hyperparams seem to work well. Not done systematic optimization. Maybe something else works much better.
                myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

                #Custom callback to record tpr at fixed fpr (set by eff_rate), where tpr and fpr refer to signal and sideband regions rather than truth-labels.
                my_check_eff = check_eff(verbose=0, filename = checkpoint_name + '_best.h5', patience = 250,
                                         preprocessed_training_data=(preprocess(data_train),labels_train),
                                         min_epoch=10,
                                         plot_period=2,eff_rate=myargs['-checkeff'])
                #Custom callback for printing scatter plots every few epochs. Useful for troubleshooting, but slows down training considerably.
                my_print = print_scatter_checkpoint(filename = checkpoint_name,
                                                    axes_list = axes_list,
                                                    axes_labels = axes_labels,
                                                    period=20,
                                                    training_data=np.append(data_train,data_valid,axis=0),
                                                    preprocess=preprocess)
                
                mycallbacks=[#my_print,
                    my_check_eff]

                #Following seems to work well for benchmarks. Not systematically optimized. I basically just played around until something worked.
                #However, bias initialization is critical. Keras relu by default initializes to 0 bias, and especially in the first layer will not move from that initialization during training. This is very suboptimal.
                model = Sequential()
                model.add(Dense(500, input_dim=numvars,use_bias=True,
                                activation='relu',
                                bias_initializer = keras.initializers.TruncatedNormal(mean=0., stddev=0.04)))
                model.add(Dropout(0.5))
                model.add(Dense(500, use_bias=True, activation='elu',
                                bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)))
                model.add(Dropout(0.5))
                model.add(Dense(50, use_bias=True, activation='elu',
                                bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))
                
                model.compile(optimizer=myoptimizer,
                              loss='binary_crossentropy')
                
                model_hist = model.fit(preprocess(data_train), labels_train, epochs=1000, batch_size=batch_size,
                                       validation_data=(preprocess(data_valid), labels_valid, weights_valid),
                                       callbacks=mycallbacks,verbose=0,
                                       sample_weight = weights_train)
                
                del model
                K.clear_session()           #Otherwise TensorFlow eats up all GPU memory with previous models.
            else:
                print("Now loading model ", i + 1, " of ", ntries)
                
            model = keras.models.load_model(checkpoint_name + "_best.h5")
            model_utils[bin_i].add_model(model, None, k, l,checkpoint_name + "_best.h5")
            plt.close('all')
            model_utils[bin_i].print_scatter_onemodel_signalregion(k,l,i,axes_list=axes_list,axes_labels=axes_labels,
                                                                   rates = np.array([0.99]),
                                                                   colors=['silver','firebrick'])
            figfile=checkpoint_name + '_scatterk_' + str(k) + '.png'
            print('Saving fig', figfile)
            plt.savefig(figfile)
            for i in range(5):
                gc.collect()
            end = time.time()
            
            times.append(end-start)
            print("Elapsed Time = ", times[-1])


    print("Bin = ", bin_i)
    print("aucs valid: ", model_utils[bin_i].aucs_valid)
    print("Effs valid: ", model_utils[bin_i].effs_valid)
    print("aucs train: ", model_utils[bin_i].aucs_train)
    print("Effs train: ", model_utils[bin_i].effs_train)
    print("\n")

    #Make an ensemble model using the average of the best models trained using the (k-1) training-validation splits. Save this as a single model.
    ensemble_model = model_utils[bin_i].makeandsave_ensemble_model(k,output_prefix + "_" + str(bin_i) + "_ensemble_k" + str(k) + ".h5")
    plt.close('all')
    model_utils[bin_i].print_scatter_avg_onek_signalregion(k,axes_list=axes_list,axes_labels=axes_labels,
                                                           rates = np.array([0.99]),
                                                           colors=['silver','firebrick'])
    figfile=output_prefix + '_scatterk_' + str(k) + '.png'
    print('Saving fig', figfile)
    plt.savefig(figfile)
    del ensemble_model
    K.clear_session()
    
model_utils[bin_i].load_all_ensemble_models()
plt.close('all')
model_utils[bin_i].print_scatter_avg_allk_signalregion(axes_list=axes_list,axes_labels=axes_labels,
                                                       rates = np.array([0.99]),
                                                       colors=['silver','firebrick'])

figfile=output_prefix + '_scatter_allk.png'
print('Saving fig', figfile)
plt.savefig(figfile)
    
for key in model_utils.keys():
    print("Now working on bin: ", key)

    kset_data, kset_prediction = model_utils[key].avg_model_predict_kset()
    plt.close('all')
    AddPredictionsToScatter_nestedcrossval(kset_data,kset_prediction,axes_list = axes_list)
    plt.title(key)
    plt.savefig(output_prefix + '_avg_model_' + str(key) + '.png')
    
    chosen_effs = [1.0,0.1,0.05,0.02,0.01,0.005]

    ymin=1E8
    ymax=0
    plt.close('all')

    file = open(output_prefix + '_bincounts.dat','w')
    
    for eff in chosen_effs:
        print("Setting eff to ", eff)
        bincutcounts, bincutcountsset = model_utils[key].get_bin_cut_counts_all(eff)
        
        bindensities = bincutcounts / mybinwidths

        print("Counts after cut: ", bincutcounts)
        #print("St. Dev. after cut: ", binstd)
        file.write(str(eff))
        file.write('\t')
        for entry in bincutcounts:
            file.write(str(entry))
            file.write('\t')
        file.write('\n')

        pplotname = output_prefix + '_pplot_' + str(eff) + '.png'
        get_p_value(bincutcounts,mybinboundaries,mask=[bin_i-1,bin_i,bin_i+1],verbose=1,
                    plotfile=pplotname)

    file.close()
