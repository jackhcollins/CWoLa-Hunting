import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
        
    return dir_path




def getopts(argv):
    ''' Get command line arguments '''
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

print("\n \n \n ########################## \n")
myargs = getopts(sys.argv)
print("Command line arguments: ", myargs)
if '-ktrain' not in myargs:
    myargs['-ktrain'] = 0
else:
    myargs['-ktrain'] = int(myargs['-ktrain'])
print('-ktrain = ', myargs['-ktrain'])
    
if '-o' not in myargs:
    myargs['-o'] = './train_output'
output_folder = create_dir(myargs['-o'])
print('-o = ', myargs['-o'])
print("\n ########################## \n")
sys.stdout.flush()

mydata = np.load("/data0/users/bpnachman/outfile100M.npz")
mysignal = np.load("/home/bpnachman/data_setup_sig_Wprime_WZqqqq_M3000_m400_m400_EXOT3_4.10.19.npz")
mysignal2 = np.load("/home/bpnachman/data_setup_sig_Wprime_WZqqqq_M3000_m200_m400_EXOT3_4.10.19.npz")

central_bin = 16    # 3 TeV signal = 16
output_prefix = output_folder + '/CWola_output'
mjjmin = 1100
mjjmax = 8300
newbinboundaries = np.logspace(np.log10(mjjmin),np.log10(mjjmax),36)
print("\n Bin boundaries: ")
print(newbinboundaries)
signalsideband_boundaries = [newbinboundaries[central_bin-3],
                             newbinboundaries[central_bin-1],
                             newbinboundaries[central_bin+2],
                             newbinboundaries[central_bin+4]]
print("\n Sideband and signal region boundaries:")
print(signalsideband_boundaries)


dijet_data = mydata
msqr = dijet_data['M1']*dijet_data['M1']
ptsqr = dijet_data['Pt1']*dijet_data['Pt1']
coshetasqr = np.cosh(dijet_data['Eta1'])*np.cosh(dijet_data['Eta1'])
y1 = np.log((np.sqrt(msqr + ptsqr * coshetasqr)+dijet_data['Pt1']*np.sinh(dijet_data['Eta1']))/np.sqrt(msqr+ptsqr))

msqr = dijet_data['M2']*dijet_data['M2']
ptsqr = dijet_data['Pt2']*dijet_data['Pt2']
coshetasqr = np.cosh(dijet_data['Eta2'])*np.cosh(dijet_data['Eta2'])
y2 = np.log((np.sqrt(msqr + ptsqr * coshetasqr)+dijet_data['Pt2']*np.sinh(dijet_data['Eta2']))/np.sqrt(msqr+ptsqr))

del msqr
del ptsqr
del coshetasqr

selected_bg = np.all(np.stack([dijet_data['Mjj'] > mjjmin,
                               dijet_data['Mjj'] < mjjmax,
                               np.abs(y1-y2) < 1.2,
                               (dijet_data['M1']-45)*3000/dijet_data['Mjj'] + 45 > 60,
                               (dijet_data['M2']-45)*3000/dijet_data['Mjj'] + 45 > 60
                              ]),
                      axis=0)


dijet_data = mysignal2
msqr = dijet_data['M1']*dijet_data['M1']
ptsqr = dijet_data['Pt1']*dijet_data['Pt1']
coshetasqr = np.cosh(dijet_data['Eta1'])*np.cosh(dijet_data['Eta1'])
y1 = np.log((np.sqrt(msqr + ptsqr * coshetasqr)+dijet_data['Pt1']*np.sinh(dijet_data['Eta1']))/np.sqrt(msqr+ptsqr))

msqr = dijet_data['M2']*dijet_data['M2']
ptsqr = dijet_data['Pt2']*dijet_data['Pt2']
coshetasqr = np.cosh(dijet_data['Eta2'])*np.cosh(dijet_data['Eta2'])
y2 = np.log((np.sqrt(msqr + ptsqr * coshetasqr)+dijet_data['Pt2']*np.sinh(dijet_data['Eta2']))/np.sqrt(msqr+ptsqr))

del msqr
del ptsqr
del coshetasqr


selected_sig_2 = np.all(np.stack([mysignal2['Mjj']*1000 > mjjmin,
                                  mysignal2['Mjj']*1000 < mjjmax,
                                  np.abs(y1-y2) < 1.2,
                                  (mysignal2['M1']-45)*3/mysignal2['Mjj'] + 45 > 60,
                                  (mysignal2['M2']-45)*3/mysignal2['Mjj'] + 45 > 60
                                 ]),
                         axis=0)


# Rescale jet masses
bg_traindata_mjj = mydata['Mjj'][selected_bg]
bg_mjmaxrescaled = (np.maximum(mydata['M1'],mydata['M2'])[selected_bg]-45)*3000/bg_traindata_mjj+45
bg_mjminrescaled = (np.minimum(mydata['M1'],mydata['M2'])[selected_bg]-45)*3000/bg_traindata_mjj+45

bg_traindata = np.stack([bg_mjmaxrescaled,bg_mjminrescaled]).T
bg_mJJbins = np.digitize(bg_traindata_mjj,newbinboundaries)
bg_sigsidebins = np.digitize(bg_traindata_mjj,signalsideband_boundaries)

sig_traindata_mjj = mysignal2['Mjj'][selected_sig_2]*1000
sig_mjmaxrescaled = (np.maximum(mysignal2['M1'],mysignal2['M2'])[selected_sig_2]-45)*3000/sig_traindata_mjj+45
sig_mjminrescaled = (np.minimum(mysignal2['M1'],mysignal2['M2'])[selected_sig_2]-45)*3000/sig_traindata_mjj+45

sig_traindata = np.stack([sig_mjmaxrescaled,sig_mjminrescaled]).T
sig_mJJbins = np.digitize(sig_traindata_mjj,newbinboundaries)
sig_sigsidebins = np.digitize(sig_traindata_mjj,signalsideband_boundaries)


numsig = 600
print("\n \n \n ########################## \n")
print("Using", numsig, "signal events")
print("\n S/sqrt(B) in low sideband, signal region, and high sideband:")
print([np.sum((sig_sigsidebins == i)[:numsig])/np.sqrt(np.sum(bg_sigsidebins == i)) for i in range(1,4)])
print("\n S/B in low sideband, signal region, and high sideband:")
print([1. * np.sum((sig_sigsidebins == i)[:numsig])/np.sum(bg_sigsidebins == i) for i in range(1,4)])
print("\n S in low sideband, signal region, and high sideband:")
print([np.sum((sig_sigsidebins == i)[:numsig]) for i in range(1,4)])
print("\n B in low sideband, signal region, and high sideband:")
print([np.sum(bg_sigsidebins == i) for i in range(1,4)])
print("\n ########################## \n")
sys.stdout.flush()


# Combine signal and sideband events, then shuffle (with fixed seed for consistency between runs)
all_traindata_mjj = np.append(bg_traindata_mjj,sig_traindata_mjj[:numsig])
all_traindata = np.append(bg_traindata,sig_traindata[:numsig],axis=0)
all_mJJbins = np.append(bg_mJJbins,sig_mJJbins[:numsig])
all_sigsidebins = np.append(bg_sigsidebins,sig_sigsidebins[:numsig])

np.random.seed(seed=0)
perms = np.random.permutation(len(all_traindata))
all_traindata_mjj = all_traindata_mjj[perms]
all_traindata = all_traindata[perms]
all_mJJbins = all_mJJbins[perms]
all_sigsidebins = all_sigsidebins[perms]

# Select only low mass jets. This is after rescaling.
clip = np.logical_and(all_traindata[:,0] < 600,all_traindata[:,1] < 600)

all_traindata_mjj = all_traindata_mjj[clip]
all_traindata = all_traindata[clip]
all_sigsidebins = all_sigsidebins[clip]
all_mJJbins = all_mJJbins[clip]


# Log and standardize training data for NN
from sklearn import preprocessing
def prepreprocess(data):
    return np.nan_to_num(np.log10(data+40))

myscaler = preprocessing.StandardScaler().fit(prepreprocess(all_traindata[np.logical_and(all_sigsidebins > 0,all_sigsidebins < 4)]))
def preprocess(data):
    return np.clip(myscaler.transform(prepreprocess(data)),-3,3)


import gc
import math
import time

import keras as keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from cwola_utils_ace import AddPredictionsToScatter_nestedcrossval
from cwola_utils_ace import model_ensemble
from cwola_utils_ace import check_eff
from cwola_utils_ace import print_scatter_checkpoint


kfolds = 5
checkeff = 0.003
patience = 1000
ntries = 5
bin_i=16
axes_list=[[0,1]]
axes_labels=[[r'$m_1$',r'$m_2$']]
plotrange = np.array([[[0,600],[0,600]]])
numvars=2
batch_size=2000

rates = np.array([0.5,0.95,0.99,0.998])
colors=['silver','grey','khaki','goldenrod','firebrick']


# Needed for model_ensemble utility function
bg_plus_signal_binned = [all_traindata[all_mJJbins == i] for i in range(1,len(newbinboundaries))]


times = list()

# A model_ensemble (defined in cwola_utils) manages the data and many models associated with the cross-validation training procedure.
model_utils = model_ensemble(bg_plus_signal_binned,
                             bin_i = bin_i,
                             kfolds=kfolds,
                             eff_for_thresh = checkeff,
                             preprocess = preprocess)

# Normally loop of ksets. In this version, only do one kset
for k in range(myargs['-ktrain'],myargs['-ktrain']+1):
    print('Starting kfold', k, 'of', kfolds-1)

    # Loop over validation sets
    for l in range(kfolds):
        if l == k:
            continue
        print('Starting lfold', l, 'of', kfolds-1)

        data_train, data_valid, labels_train, labels_valid, weights_train, weights_valid = model_utils.get_trainval_data(k,l)

        for i in range(ntries):
            print(" k =", k, "l =", l)
            #Naming convention for model files.
            checkpoint_name = output_prefix + "_" + str(bin_i) + "_[" + str(k) + "," + str(l) + "]_" + str(i)
            start = time.time()


            print("Now training model ", i + 1, " of ", ntries)

            K.clear_session()
            #Following hyperparams seem to work well. Not done systematic optimization. Maybe something else works much better.
            myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

            #Custom callback to record tpr at fixed fpr (set by eff_rate), where tpr and fpr refer to signal and sideband regions rather than truth-labels.
            my_check_eff = check_eff(verbose=0,
                                     filename = checkpoint_name + '_best.h5',
                                     patience = patience,
                                     min_epoch=50, batch_size=batch_size,
                                     plot_period=200,eff_rate=checkeff,
                                     plotmode="save",avg_length=20,
                                     preprocessed_training_data = [preprocess(data_train), labels_train],
                                     show_plot_end = False,
                                     max_epochs=2500)
            
            mycallbacks=[my_check_eff]

            #Following seems to work well for benchmarks. Not systematically optimized. I basically just played around until something worked.
            #However, bias initialization seems very important. Keras relu by default initializes to 0 bias, and especially in the first layer will not move from that initialization during training. This is very suboptimal.
            model = Sequential()
            model.add(Dense(256, input_dim=numvars,use_bias=True,
                            bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.5)))
            model.add(keras.layers.LeakyReLU(alpha=0.01))
            model.add(Dropout(0.1))
            model.add(Dense(128, use_bias=True, activation='elu',
                            bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
            model.add(Dropout(0.1))
            model.add(Dense(64, use_bias=True, activation='elu',
                            bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
            model.add(Dropout(0.1))
            model.add(Dense(32, use_bias=True, activation='elu',
                            bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
            model.add(Dense(8, use_bias=True, activation='elu',
                            bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer=myoptimizer,
                          loss='binary_crossentropy')

            model_hist = model.fit(preprocess(data_train), labels_train, epochs=2500, batch_size=batch_size,
                                   validation_data=(preprocess(data_valid), labels_valid, weights_valid),
                                   callbacks=mycallbacks,verbose=0,
                                   sample_weight = weights_train)

            del model
            K.clear_session()           #Otherwise TensorFlow eats up all GPU memory with previous models.


            model = keras.models.load_model(checkpoint_name + "_best.h5")
            model_utils.add_model(model, None, k, l,checkpoint_name + "_best.h5")
            plt.close('all')
            


            print("Selected Model:")
            model_utils.print_scatter_onemodel_signalplussidebandregion(k,l,i,
                                                                        axes_list=axes_list,
                                                                        axes_labels=axes_labels,
                                                                        rates=rates,
                                                                        colors=colors,
                                                                        plotmode = "save")
            
            plt.savefig(checkpoint_name + '_scatter.png')

            
            K.clear_session()  

            for i in range(5):
                gc.collect()
            end = time.time()

            times.append(end-start)
            print("Elapsed Time = ", times[-1])

    print("Bin = ", bin_i)
    print("aucs valid: ", model_utils.aucs_valid)
    print("Effs valid: ", model_utils.effs_valid)
    print("aucs train: ", model_utils.aucs_train)
    print("Effs train: ", model_utils.effs_train)
    print("\n")

    #Make an ensemble model using the average of the best models trained using the (k-1) training-validation splits. Save this as a single model.
    ensemble_model = model_utils.makeandsave_ensemble_model(k,output_prefix + "_" + str(bin_i) + "_ensemble_k" + str(k) + ".h5")
    plt.close('all')
    
    print("Ensemble model for k =", k)
    model_utils.print_scatter_avg_onek_signalplussidebandregion(k,
                                                                axes_list=axes_list,
                                                                axes_labels=axes_labels,
                                                                rates=rates,
                                                                colors=colors,
                                                                plotmode = "save")
    plt.savefig(output_prefix + "_" + str(bin_i) + "_" + str(k) + '_avg_scatter.png')
    
    del ensemble_model
    K.clear_session()



