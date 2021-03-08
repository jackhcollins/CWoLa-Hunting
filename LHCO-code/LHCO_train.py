from __future__ import print_function
#import os
import gc
import sys
import numpy as np
import numpy.random as rand
import math
import matplotlib
matplotlib.use('Agg')
fontsize=18
smfontsize=14
matplotlib.rcParams['xtick.labelsize'] = smfontsize 
matplotlib.rcParams['ytick.labelsize'] = smfontsize 

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
# import keras as keras
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Activation, Dropout
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras import regularizers
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
#import pickle as pickle

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
import time
import glob
import numpy.ma as ma
from tensorflow.keras import backend as K
# K.set_session(sess)

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
    myargs['-bgevnts'] = 553388
    print("Setting default  '-bgevnts':", myargs['-bgevnts'])
else:
    myargs['-bgevnts'] = int(myargs['-bgevnts'])
if '-bgoffset' not in myargs:
    myargs['-bgoffset'] = 0
    print("Setting default  '-bgoffset':", myargs['-bgoffset'])
else:
    myargs['-bgoffset'] = int(myargs['-bgoffset'])

if '-batchsize' not in myargs:
    myargs['-batchsize'] = 5000
    print("Setting default  '-batchsize':", myargs['-batchsize'])
else:
    myargs['-batchsize'] = int(myargs['-batchsize'])
if '-patience' not in myargs:
    myargs['-patience'] = 250
    print("Setting default  '-patience':", myargs['-patience'])
else:
    myargs['-patience'] = int(myargs['-patience'])
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
if '-trainonly' not in myargs:
    myargs['-trainonly'] = 0
    print("Setting default  '-trainonly':", myargs['-trainonly'])
else:
    myargs['-trainonly'] = int(myargs['-trainonly'])
if '-makeensemble' not in myargs:
    myargs['-makeensemble'] = 0
    print("Setting default  '-makeensemble':", myargs['-makeensemble'])
else:
    myargs['-makeensemble'] = int(myargs['-makeensemble'])
    
if '-startk' not in myargs:
    myargs['-startk'] = 0
    print("Setting default  '-startk':", myargs['-startk'])
else:
    myargs['-startk'] = int(myargs['-startk'])
if '-startl' not in myargs:
    myargs['-startl'] = 0
    print("Setting default  '-startl':", myargs['-startl'])
else:
    myargs['-startl'] = int(myargs['-startl'])
if '-endk' not in myargs:
    myargs['-endk'] = 0
    print("Setting default  '-endk':", myargs['-endk'])
else:
    myargs['-endk'] = int(myargs['-endk'])
if '-endl' not in myargs:
    myargs['-endl'] = 0
    print("Setting default  '-endl':", myargs['-endl'])
else:
    myargs['-endl'] = int(myargs['-endl'])
    
bin_i = myargs['-bin']    
data_prefix = myargs['-in']
output_prefix = myargs['-o']

#Which auxilliary variables to use
selected_vars = [1,4,2,5,3,6]
#Also use mJJ.
selected_vars_plus = np.append([0],selected_vars)

#Which 2d planes to make scatter plots in
axes_list = [[1,2],[3,4],[5,6]]
axes_labels = [[r'$m_{J,A}$',r'$m_{J,B}$'],[r'$\tau_{21,A}$',r'$\tau_{21,B}$'],[r'$\tau_{32,A}$',r'$\tau_{32,B}$']]

#Data binning in mJJ
mjjmin = 2500
mjjmax = 6000
mybinboundaries = np.round(np.logspace(np.log10(mjjmin), np.log10(mjjmax), num=18))
mybincenters = np.array([0.5*(mybinboundaries[i+1] + mybinboundaries[i]) for i in range(0,len(mybinboundaries)-1)])
mybinwidths = np.array([mybinboundaries[i+1] - mybinboundaries[i] for i in range(0,len(mybinboundaries)-1)])
mybincenterandwidth = np.vstack((mybincenters,mybinwidths)).T

def bin_data(data, binboundaries = mybinboundaries):
    databinned = []
    for i in range(0,len(binboundaries)-1):
        databinned.append(
            np.array([myrow for myrow in data if (myrow[0] < binboundaries[i+1] and myrow[0] >= binboundaries[i])])
        )
    return databinned

import pandas as pd
#load data
print('\n')
print("Loading E:\projects\CWoLa-Hunting\LHCO_data\events_anomalydetection_v3.features.h5")
f = pd.read_hdf("E:/projects/CWoLa-Hunting/LHCO_data/events_anomalydetection_v3.features.h5")
data = f.values
E1 = np.sqrt(np.square(np.linalg.norm(data[:,:3],axis=-1))+np.square(data[:,3]))
E2 = np.sqrt(np.square(np.linalg.norm(data[:,7:10],axis=-1))+np.square(data[:,10]))
mjj = np.sqrt(np.square(E1+E2)-np.square(np.linalg.norm(data[:,:3]+data[:,7:10],axis=-1)))
bg_plus_signal = np.concatenate(
    (mjj[:,None],data[:,3:4],data[:,5:6]/data[:,4:5],data[:,6:7]/data[:,5:6],
    data[:,10:11],data[:,12:13]/data[:,11:12],data[:,13:14]/data[:,12:13]), axis=-1
)
bg_plus_signal = np.nan_to_num(bg_plus_signal)
bg_plus_signal = bg_plus_signal[np.greater(mjj,mjjmin) & np.less(mjj,mjjmax)]

# Preprocessor for data before feeding it to NN
#   1) Take log off jet mass variables
#   2) Standard scale all auxilliary variables
def myprepreprocessor(predata, massvars=None):
    if massvars is None:
        massvars = [0,1]
    newdata = np.copy(predata)
    newdata[:,massvars] = np.nan_to_num(np.log10((newdata[:,massvars]-30.)/newdata[:,0:1]+30./3000.))
    return newdata[:,1:]


bg_plus_signal=bg_plus_signal[:,selected_vars_plus]
myscaler = preprocessing.StandardScaler().fit(myprepreprocessor(bg_plus_signal,massvars=[1,2]))
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

# This is leftover from earlier days when we experimented with removing jet mass from the auxilliary variables to make them dimensionless.
no_mass = False
numvars=len(selected_vars)
if no_mass:
    numvars = numvars-2


    
# I have found that large batch sizes ~5000 work well for our problem. I have not done a systematic test or scan. But my intuition is that because signal events are so rare, even in the signal region, you want batch sizes large enough so that most batches will contain at least a few true signal events.
batch_size = myargs['-batchsize']

model_utils = {}
model_utils[bin_i] = model_ensemble(bg_plus_signal_binned, bin_i = bin_i, kfolds=myargs['-kfold'], preprocess = preprocess, eff_for_thresh = myargs['-checkeff'])

gc.collect()

if (not myargs['-loadonly']):
# if True:
    #Loop over kfolds
    startnum = 0
    if myargs['-trainonly']:
        startnum = myargs['-startk']
        
    for k in range(startnum,myargs['-kfold']):
        if (myargs['-endk'] > 0) and (k == myargs['-endk']):
            print("Ending at k = ", k)
            sys.stdout.flush()
            quit()
        print('Starting kfold', k, 'of', myargs['-kfold'])
        
        #Loop over validation sets
        for l in range(myargs['-kfold']):
            if (myargs['-endl'] > 0) and (l == myargs['-endl']):
                print("Ending at l = ", l)
                sys.stdout.flush()
                quit()
            if l == k:
                continue

            if myargs['-trainonly'] and (l < myargs['-startl']):
                continue

            print('Starting lfold', l, 'of', myargs['-kfold'])
            
            data_train, data_valid, labels_train, labels_valid, weights_train, weights_valid = model_utils[bin_i].get_trainval_data(k,l)

            for i in range(ntries):
                print(" k =", k, "l =", l)
                #Naming convention for model files.
                checkpoint_name = output_prefix + "_" + str(bin_i) + "_[" + str(k) + "," + str(l) + "]_" + str(i)
                start = time.time()
                
                #The flag myargs['-loadonly'] allows to skip training step and just load pre-saved models.
                if (not myargs['-makeensemble']) & (not ((k == myargs['-startk']) & (l < myargs['-startl']))) & (k >= myargs['-startk']):
                    print("Now training model ", i + 1, " of ", ntries)
                    
                    K.clear_session()
                    #Following hyperparams seem to work well. Not done systematic optimization. Maybe something else works much better.
                    myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
                    
                    #Custom callback to record tpr at fixed fpr (set by eff_rate), where tpr and fpr refer to signal and sideband regions rather than truth-labels.
                    my_check_eff = check_eff(verbose=0, filename = checkpoint_name + '_best.h5', patience = myargs['-patience'],
                                             preprocessed_training_data=(preprocess(data_train),labels_train),
                                             min_epoch=10,
                                             plot_period=2,eff_rate=myargs['-checkeff'],
                                             validation_data = (preprocess(data_valid), labels_valid, weights_valid))
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
                    #However, bias initialization seems very important. Keras relu by default initializes to 0 bias, and especially in the first layer will not move from that initialization during training. This is very suboptimal.
                    model = Sequential(name = str(bin_i) + "_" + str(k) + "_" + str(l) + "_" + str(i))
                    model.add(Dense(128, input_dim=numvars,use_bias=True,
                                    #activation='relu',
                                    bias_initializer = keras.initializers.TruncatedNormal(mean=0., stddev=0.04)))
                    model.add(keras.layers.LeakyReLU(alpha=0.1))
                    model.add(Dropout(0.1))
                    model.add(Dense(128, use_bias=True, activation='elu',
                                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)))
                    model.add(Dropout(0.1))
                    model.add(Dense(128, use_bias=True, activation='elu',
                                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)))
                    model.add(Dropout(0.1))
                    model.add(Dense(128, use_bias=True, activation='elu',
                                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)))
                    model.add(Dense(1, activation='sigmoid'))
                    
                    model.compile(optimizer=myoptimizer,
                                  loss='binary_crossentropy')
                    
                    model_hist = model.fit(preprocess(data_train), labels_train, epochs=2000, batch_size=batch_size,
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
            
                if (not myargs['-loadonly']) & (not ((k == myargs['-startk']) & (l < myargs['-startl']))) & (k >= myargs['-startk']):
                    model_utils[bin_i].print_scatter_onemodel_signalregion(k,l,i,axes_list=axes_list,axes_labels=axes_labels,
                                                                           rates = np.array([1. - myargs['-checkeff']]),
                                                                           colors=['silver','firebrick'])
                    figfile=checkpoint_name + '_scatterk_' + str(k) + '.png'
                    print('Saving fig', figfile)
                    plt.savefig(figfile)
                    
                for i in range(5):
                    gc.collect()
                end = time.time()
                
                times.append(end-start)
                print("Elapsed Time = ", times[-1])
                sys.stdout.flush()
                
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
                                                               rates = np.array([1. - myargs['-checkeff']]),
                                                               colors=['silver','firebrick'])
        figfile=output_prefix + '_scatterk_' + str(bin_i) + '_' + str(k) + '.png'
        print('Saving fig', figfile)
        plt.savefig(figfile)
        del ensemble_model
        K.clear_session()
        sys.stdout.flush()
    if myargs['-trainonly']:
        quit()
    model_utils[bin_i].load_all_ensemble_models()
else:
    # Only load already-trained model
    ensemble_model_names = [output_prefix + "_" + str(bin_i) + "_ensemble_k" + str(k) + ".h5" for k in range(myargs['-kfold'])]               
    model_utils[bin_i].load_all_ensemble_models(model_names = ensemble_model_names)

# Make scatter plots to show selected events.
plt.close('all')
model_utils[bin_i].print_scatter_avg_allk_signalregion(axes_list=axes_list,axes_labels=axes_labels,
                                                       rates = np.array([1. - myargs['-checkeff']]),
                                                       colors=['silver','firebrick'], fontsize = fontsize)

plt.subplot(1, len(axes_list), 1)
plt.xlim([0,1200])
plt.ylim([0,1200])
plt.subplots_adjust(wspace=0.2)

figfile=output_prefix + '_scatter_' + str(bin_i) + '_allk.png'
print('Saving fig', figfile)
plt.savefig(figfile, bbox_inches='tight')

sys.stdout.flush()


######
###### Apply cuts on data, save mJJ data distributions, calculate p-values
######

for key in model_utils.keys():
    print("Now working on bin: ", key)

    kset_data, kset_prediction = model_utils[key].avg_model_predict_kset()
    plt.close('all')
    AddPredictionsToScatter_nestedcrossval(kset_data,kset_prediction,axes_list = axes_list)
    plt.title(key)
    plt.savefig(output_prefix + '_avg_model_' + str(key) + '.png')
    
    chosen_effs = [1.0,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001]

    ymin=1E8
    ymax=0
    plt.close('all')

    file = open(output_prefix + '_' + str(bin_i) +  '_bincounts.dat','w')
    bincutcountslist = []
    
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
        bincutcountslist.append(bincutcounts)
        
    for i, eff in enumerate(chosen_effs):
        print("Getting p-value for eff:", eff)

        pplotname = output_prefix + '_pplot_' + str(bin_i) + '_' + str(eff) + '.png'
        get_p_value(bincutcountslist[i],mybinboundaries,mask=[bin_i-1,bin_i,bin_i+1],verbose=1,
                    plotfile=pplotname)

    file.close()

sys.stdout.flush()