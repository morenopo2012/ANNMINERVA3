#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os
import gzip
import matplotlib.pyplot as plt
import numpy as np
import logging
from pprint import pformat
from matplotlib.colors import LogNorm
from decimal import Decimal
from mnvtf.hdf5_readers import SimpleCategorialHDF5Reader as HDF5Reader
#from sklearn.preprocessing import normalize

def usage():
    sys.exit('''
Usage:
    python mat_plot.py <(probabilities).py> <(truth-labels).py>
Code to print confusion matrix given the actuals vector ant the probability
vector
Example usage:
    python mat_plot.py <(probabilities).py> <(truth-labels).py> \
        optional <(invariant mass W).py>
or:
    python mat_plot.py <(conf_mat).py>
'''
    )


#----------Function that reads a .gz and .root files to know how many entries have. ----------------------------------
#---------This function works to save the values of the prediction and probabilities in the data_dict{} dictionary----
#---------------------------------------------------------------------------------------------------------------------
def file2npy(filename, index_dict):
    #data_dict is used to save the classification number and the probabilities vectors
    data_dict = {}
    LOGGER.info('\nLoading {} '.format(filename)) #Oscar
    string = filename.split('.') #This line separate the filename into words separated by "."

    #ftype is going to save the extension of the file to read, in the usual case 'gz'
    ftype = string[len(string)-1]

    #LOGGER.info('index_dict = {}'.format(index_dict)) #Shows the arrays prediction and probabilities
    LOGGER.info('The ftype = string[len(string)-1] = {}'.format(ftype))

#Open the file .gz and checks how many entries are and how it separate the line (l)
    if ftype == 'gz':
        text_file = gzip.open( filename, 'r' )
        for i, l in enumerate( text_file ):
            pass
        entries = i + 1

    elif ftype == 'root':
        from ROOT import TFile, TTree
        mc_file = TFile( filename, 'read' )
        mc_tree = mc_file.Get('evt_pred')
        entries = mc_tree.GetEntries()
    else:
        sys.exit('File extension not supported')

    for key in index_dict:
        data_dict[key] = np.zeros([entries,len(index_dict[key])])
        #LOGGER.info("for key: {} the data_dict is: {}".format(key,data_dict)) #print zero arrays for prediction and probabilities arrays
        LOGGER.info("with entries: {}".format(entries))

#This part take each line of the prediction file and separate variables by commas
    if ftype == 'gz':
        text_file = gzip.open( filename, 'rt' )
        for i, line in enumerate(text_file): #Take the i line and line is for strip it
            line = line.strip() # Take the values of all the line
            #LOGGER.info('line = {}'.format(line))
            currentline = line.split(',') #Divide the line into variables by comma
            #LOGGER.info('currentline is = {}'.format(currentline))

            for key in index_dict:
                for entry in range(len(index_dict[key])):
                    data_dict[key][i][entry] = \
                        currentline[index_dict[key][entry]] #Looks for the 4th column in that line to assing that number

        text_file.close()

    elif ftype == 'root':
        entries = mc_tree.GetEntries()
        for i, event in enumerate(mc_tree):
            if i % 50000 == 0:
                LOGGER.info('Entry {}/{}'.format(i,entries))
                sys.stdout.flush()

            W[i]           = mc_tree.mc_w / 1000
            Q2[i]          = mc_tree.mc_Q2 / 1000000
            n_tracks[i]    = mc_tree.n_tracks
            actuals[i]     = mc_tree.true_mult
            predictions[i] = mc_tree.pred_mult

        mc_file.Close()

        LOGGER.info("ROOT file loaded")

    #Returns a vector data_dict: {'predictions': array([2., 2., 2., ..., 2., 2., 2.])}
    return data_dict #, W, Q2, n_tracks actuals,
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


def plot_array(array, hist_title, file_title, categories,
               ann = False, logz = False, dec = False):
    '''
    Read the confusion matrix <array> and plot it. Needed input is the
    title drawn in the plot, and the name of the .pdf output file.
    Also change <ann> to True if you want annotations of bin values on
    top of them. <logz> is to draw log scales, and dec is truncate decimals
    '''
    #LOGGER.info("array.max(): {} ".format(array.max()))
    #LOGGER.info("Saving {} plot...".format(hist_title))
    #LOGGER.info("array is: {} ".format(array)) #This is for know the array
    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(20,20))

    if(categories == 213):
        x_label_list = ['47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64',
                        '65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84',
                        '85','86','87','88','89','90','91','92','93','94','95','97','99','101','103','105','107','109','111','113']
        y_label_list = ['47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64',
                        '65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84',
                        '85','86','87','88','89','90','91','92','93','94','95','97','99','101','103','105','107','109','111','113']
        #y_label_list = ['104','103','102','101','100','99','98','97','96','95','94','93','92','91','90','89','88','87','86','85']
        ax.set_xticks([1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,
                        41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,
                        81.0,83.0,85.0,87.0,89.0,91.0,93.0,95.0,97.0,99.0,101.0,103.0,105.0,107.0,109.0,111.0,113.0,115.0])
        ax.set_yticks([1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,
                        41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,
                        81.0,83.0,85.0,87.0,89.0,91.0,93.0,95.0,97.0,99.0,101.0,103.0,105.0,107.0,109.0,111.0,113.0,115.0])


    if logz:
        norm = LogNorm()
    else:
        norm = None

    if array.max() > 1:
        #img1 = ax.imshow(array, cmap='YlOrBr', norm=norm,extent=[175,209,175,209])  #before extent=[176,195,176,195]
        if(categories == 213):
            img1= ax.imshow(array, cmap='cool', norm=norm, extent=[0,116,116,0])
        ax.set_xticklabels(x_label_list, fontsize = 3)
        ax.set_yticklabels(y_label_list, fontsize = 4)
    else:
      #img1 = ax.imshow(array, cmap='YlOrBr', vmin=0, vmax=0.8,extent=[175,209,175,209])
        if logz == False :
            img1= ax.imshow(array, cmap='cool',vmin=0, vmax=1.0, extent=[0,116,116,0])
            ax.set_xticklabels(x_label_list, fontsize = 3)
            ax.set_yticklabels(y_label_list, fontsize = 4)

        elif logz == True :
            img1= ax.imshow(array, cmap='cool',norm=LogNorm(), extent=[0,116,116,0]) #LogNorm(vmin=0.0001, vmax=1)
            ax.set_xticklabels(x_label_list, fontsize = 3)
            ax.set_yticklabels(y_label_list, fontsize = 4)


    colours = img1.cmap(img1.norm(np.unique(array)))

    #To put number in the boxes
#    for i in range(len(array)):
#        for j in range(len(array)):
#            text = ax.text(j, i, Decimal(array[i,j]),ha="center", va="center", color="white",fontsize=2)

    #We want to show all ticks...
    axis_max = len(array)
    #LOGGER.info("The axis_max is: {}".format(axis_max))


    if axis_max < 15:
        ax.set_xticks(np.arange(axis_max))
        ax.set_yticks(np.arange(axis_max))

        #...and label them with the respective list entries
        ax.set_xticklabels(np.arange(axis_max))
        ax.set_yticklabels(np.arange(axis_max))

        #Loop over data dimensions and create text annotations.
        threshold = img1.norm(array.max())/2.
        textcolors = ["black", "white"]

    if axis_max < 9:
        for j, row in enumerate(array):  #Columns and rows are same length
            for i, column in enumerate(array):
                if dec:
                    x = round(array[i,j], 3)
                else:
                    x = Decimal(array[i,j])
                try:
                    color = textcolors[img1.norm(array[i,j]) > .35]
                except TypeError:          #We have problems with log norm
                    color = "black"
                text = ax.text(j, i, x,
                       ha="center", va="center", color=color)

    fig.colorbar(img1, ax=ax)              #Color bar is the size of the plot
    plt.ylabel("truth (label)")
    plt.xlabel("prediction (target)")
    plt.title(hist_title)
    plt.tight_layout()
    plt.savefig(file_title)
    plt.close()



def column_normalize(array):
    for j, column in enumerate(array):
        norm = array[:,j].sum(0)
        if norm != 0:
            array[:,j] = np.round(array[:,j]/norm,5)

    return array



def row_normalize(array):
    for j, row in enumerate(array):
        norm = array[j].sum(0)
        if norm != 0:
            array[j] = np.round(array[j]/norm,5)

    return array



# Recall, precision and F1 score
def prcsn_rcll(array):
    perf_mat = []
    keyes = ['Mult', 'r', 'p', 'F1']
    perf_mat.append(keyes)
    acc = 0
    f1mean = 0
    f1hmean = 0
    f1valid = True

    for j, column in enumerate(array):
        rcll_norm  = array[:,j].sum(0)
        if rcll_norm != 0:
            rcll = array[j,j]/rcll_norm
        else:
            rcll = 0

        prcsn_norm = array[j].sum(0)

        if prcsn_norm != 0:
            prcsn = array[j,j]/prcsn_norm
        else:
            prcsn = 0

        if prcsn+rcll != 0:
            F1 = 2*prcsn*rcll/(prcsn+rcll)
        else:
            F1 = 0

        values = [j, round(rcll, 4), round(prcsn, 4), round(F1, 4)]
        perf_mat.append(values)

        if F1 != 0 and f1valid:
            f1hmean += 1/F1
        else:
            f1hmean = 0
            f1valid = False

        f1mean += F1
        acc += array[j,j]

    if f1valid:
        f1hmean = len(array)/f1hmean

    f1mean /= len(array)
    acc = float(acc) / float(array.sum())

    acc = 100*round(acc,3)
    f1mean = 100*round(float(f1mean),3)
    f1hmean = 100*round(f1hmean,3)

    LOGGER.info('"Recall", "precision" and, "F1 score" per label:')
    LOGGER.info(pformat(perf_mat))
    LOGGER.info( "Global accuracy: {}%".format(acc)  )
    LOGGER.info( "F1 mean: {}%".format(f1mean) )
    LOGGER.info( "F1 harmonic mean: {}%\n".format(f1hmean) )



def conf_mat(actuals=None, predictions=None,categories=194, suffix='', suffix2='', cm=None):
    dec=True
    ann=True

    predval= np.amax(predictions)
    predvad=np.amin(predictions)

    LOGGER.info('prediction value is: {}'.format(predictions))
    LOGGER.info('prediction value max is: {}'.format(predval))
    LOGGER.info('prediction value min is: {}'.format(predvad))
    LOGGER.info('size of predictions: {}'.format(len(predictions)))

    predval2= np.amax(actuals)
    predvad2=np.amin(actuals)

    LOGGER.info('actuals value is: {}'.format(actuals))
    LOGGER.info('actuals value max is: {}'.format(predval2))
    LOGGER.info('actuals value min is: {}'.format(predvad2))
    LOGGER.info('size of actuals: {}'.format(len(actuals)))
    LOGGER.info('cm: {}'.format(cm))

    if '0' in suffix:
        prefix = '1.0 < W < 1.4 [GeV] ' + suffix2 + '[GeV\u00b2]'
    elif '1' in suffix:
        prefix = '1.4 < W < 2 [GeV] ' + suffix2 + '[GeV\u00b2]'
    elif '2' in suffix:
        prefix = 'W > 2 [GeV] ' + suffix2 + '[GeV\u00b2]'
    else:
        prefix = ' ' + suffix2 + '[GeV\u00b2]'

    if ' Q\u00b2 > 1 ' in suffix2:
        q2title = 'Q1'
    else:
        q2title = 'Q0'

    if cm is not None:
        f = cm
    elif actuals is not None and predictions is not None:
        mat_size = np.amax(np.maximum(actuals, predictions)) + 1
        LOGGER.info('mat_size = np.amax(np.maximum(actuals, predictions)) + 1 = {}'.format(mat_size))
        conf_mat = np.zeros((mat_size,mat_size))
        #LOGGER.info('conf_mat = np.zeros((mat_size,mat_size)) = {}'.format(conf_mat)) #Print the array with all zeros
        LOGGER.info('conf_mat.shape = {}'.format(conf_mat.shape))
        #f = conf_mat
        #To bigger size
        if(categories > 100):
            f = conf_mat[100:categories,100:categories]
        for a, p in zip(actuals, predictions):
            conf_mat[a,p] += 1 #Acumulador, para obtener las frecuencias.    
    else:
        sys.exit("There is nothing to plot")


    plot_array(
        f.copy(),
        prefix +"Confusion Matrix",
        "conf_mat"+suffix+q2title+".pdf",
        categories,
        ann=ann

    )
    LOGGER.info('f.shape after = {}'.format(f.shape))
    plot_array(
        f.copy(),
        prefix +"LogConfusion Matrix",
        "log_conf_mat"+ suffix +q2title+".pdf",
        categories,
        ann=ann, logz=True
    )
    g = column_normalize(f.copy())
    plot_array(
        g,
        prefix +"Column Normalized Confusion Matrix",
        "conf_mat_col_norm"+ suffix +q2title+".pdf",
        categories,
        ann=ann,
        dec=dec
    )
    plot_array(
        g,
        prefix +"Log Column Norm. Conf. Matrix",
        "log_conf_mat_col_norm"+ suffix +q2title+".pdf",
        categories,
        ann=ann, logz=True,
        dec=dec
    )
    h = row_normalize(f.copy())
    # printing row norm matrix is usefull for bilinear loss
    if suffix is '':
        LOGGER.debug("\nRow normalized comfusion matrix:\n{}".format(h))
    plot_array(
        h,
        prefix +"Row Normalized Confusion Matrix",
        "conf_mat_row_norm"+ suffix +q2title+".pdf",
        categories,
        ann=ann,
        dec=dec
    )
    plot_array(
        h,
        prefix +"Log Row Norm. Conf. Matrix",
        "log_conf_mat_row_norm"+ suffix +q2title+".pdf",
        categories,
        ann=ann,logz=True,
        dec=dec
    )
    fmat = 2*np.where( g*h != 0, np.divide(g*h, g+h), 0)
    plot_array(
        fmat,
        prefix +"F1 Confusion Matrix",
        "conf_mat_f1"+ suffix +q2title+".pdf",
        categories,
        ann=ann,
        dec=dec
    )

    prcsn_rcll(f.copy())



def mult_plot(actuals, predictions, n_tracks=None, suffix='', suffix2=''):
    if suffix == '0':
        prefix = '0.9 < W < 1.4 [GeV] ' + suffix2
    elif suffix == '1':
        prefix = '1.4 < W < 2 [GeV] '+suffix2
    elif suffix == '2':
        prefix = 'W > 2 [GeV] '+suffix2
    else:
        prefix = suffix+suffix2+'  [GeV\u00b2]'

    if ' Q\u00b2 > 1 ' in suffix2:
        q2title = 'Q1'
    else:
        q2title = 'Q0'

    LOGGER.info("Saving multiplicities {}plot...".format(prefix))
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(nrows=2,sharex=True)

    ax1.get_position()
    ax1.set_position([0.125,0.37, 0.775, 0.51]) #This is to put the axes-position in the box
    ax2.set_position([0.125, 0.11, 0.775, 0.22])

    ax1.grid(True)
    ax2.grid(True)
#    plt.minorticks_on()
#    plt.grid(b=True, which='minor', linestyle=':')
#    fig.tight_layout()

    ns1, bins1, patches1 = ax1.hist(
        actuals,
        bins=range(50,213,1),
        align='left',
        histtype=u'step',
        edgecolor='orange',
        linewidth=0.9,
        label='Truth'
    )
    ns2, bins2, patches2 = ax1.hist(
        predictions,
        bins=range(50,213,1),
        align='left',
        histtype=u'step',
        edgecolor='blue',
        linewidth=0.9,
        label='ML pred'
    )
    chi2binpre = np.divide((ns2 - ns1)**2, ns1*ns1.sum())
    ratio = np.divide(ns2, ns1)
    if n_tracks is not None:
        ns3, bins3, patches2 = ax1.hist(
            n_tracks,
            bins=range(50,213,1),
            align='left',
            histtype=u'step',
            edgecolor='green',
            linewidth=0.9,
            label='Tracked'
        )
        chi2bintra = np.divide((ns3 - ns1)**2, ns-1*ns1.sum())
    ax1.legend()
    ax1.ticklabel_format(
        axis='y',
        style='sci',
        scilimits=(0,2)
    )
    ax1.set_axisbelow(True)

    x = np.linspace(50,212,100) #Before -1,212,100
    y = np.ones_like(x)

    ax2.plot(x, y, '-k', linewidth=.795) #where='mid',
    ax2.plot(
        range(50,212,1),
        ratio,
        '.',#drawstyle='steps-mid',
        color='blue') #where='mid',
    if n_tracks is not None:
        ax2.plot(
            range(50,212,1),
            chi2bintra,
            'o',#drawstyle='steps-mid',
            color='green') #where='mid',
    plt.ylim(0, 2)
#    plt.yscale('log')

    plt.xlim((50,213)) #-.8,5.8
    ax1.yaxis.set_label_coords(-0.075,0.5)
    ax2.yaxis.set_label_coords(-0.075,0.5)
    ax1.set_title(prefix + 'Plane Code Events')
    ax1.set_ylabel('Events')
    ax2.set_ylabel('Ratio [ML/Truth]')
    ax2.set_xlabel('Plane Code')
    plt.savefig('mult'+ suffix +q2title+'.pdf')
    plt.close()

    chi2 = round(np.divide((ns2 - ns1)**2 , ns1*ns1.sum()).sum(), 4)
    LOGGER.debug("------------------------------------------------------")
    LOGGER.debug('ML Prediction $\\chi^2 \\approx {}$'.format(chi2))
    if n_tracks is not None:
        chi2 = round(np.divide((ns3 - ns1)**2 , ns1*ns1.sum()).sum(), 4)
        LOGGER.debug('Tracks $\\chi^2 \\approx {}$\n'.format(chi2))




def mult_stacked_plot(actuals, classifications, suffix='', suffix2='', norm=None):
    if '0' in suffix:
        prefix = '0.9 < W < 1.4 [GeV] ' + suffix2
    elif '1' in suffix:
        prefix = '1.4 < W < 2 [GeV] ' + suffix2
    elif '2' in suffix:
        prefix = 'W > 2 [GeV] ' + suffix2
    else:
        prefix = suffix + suffix2
    LOGGER.info("Saving stacked multiplicities {} plot...".format(prefix))

    if ' Q\u00b2 > 1 ' in suffix2:
        q2title = 'Q1'
    else:
        q2title = 'Q0'

    plt.figure()
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(175,195,1)) #ECAL
    ax.set_xticklabels(['175','176','177','178','179','180','181','182','183',
                        '184','185','186','187','188','189','190','191','192',
                        '193','194'],fontsize =6)
    #HCAL
#    ax.set_xticklabels(['195','196','197','198','199','200','201','202','203',
#                        '204','205','206','207','208','209','210','211',
#                        '212','213','214'],fontsize =6)
    #ECAL
    classifications_true = (
        classifications[actuals==175], classifications[actuals==176],
        classifications[actuals==177], classifications[actuals==178],
        classifications[actuals==179], classifications[actuals==180],
        classifications[actuals==181], classifications[actuals==182],
        classifications[actuals==183], classifications[actuals==184],
        classifications[actuals==185], classifications[actuals==186],
        classifications[actuals==187], classifications[actuals==188],
        classifications[actuals==189], classifications[actuals==190],
        classifications[actuals==191], classifications[actuals==192],
        classifications[actuals==193], classifications[actuals==194]
        )

    #HCAL
    #classifications_true = (
    #    classifications[actuals==195], classifications[actuals==196],
    #    classifications[actuals==197], classifications[actuals==198],
    #    classifications[actuals==199], classifications[actuals==200],
    #    classifications[actuals==201], classifications[actuals==202],
    #    classifications[actuals==203], classifications[actuals==204],
    #    classifications[actuals==205], classifications[actuals==206],
    #    classifications[actuals==207], classifications[actuals==208],
    #    classifications[actuals==209], classifications[actuals==210],
    #    classifications[actuals==211], classifications[actuals==212],
    #    classifications[actuals==213], classifications[actuals==214]
    #    )

    hist = []
    weights = []

    for i, array in enumerate(classifications_true):
        xi_counts, bins = np.histogram(array, bins=range(175,200,1))
        if norm is None:
            xi_weights = 100 * xi_counts / len(classifications)
        else:
            xi_weights = 100 * xi_counts / norm

        hist.append(bins[:-1])
        weights.append(xi_weights)
#    print(hist)
#    print(weights)
#Truth
    labels = ('175','176','177',
              '178','179','180','181',
              '182','183','184','185','186',
              '187','188','189','190','191',
              '192','193','194')

    plt.hist(
        hist,
        bins=bins,
        weights=weights,
        align='left',
        stacked=True,
        #histtype=u'step',
        #edgecolor='blue',
        linewidth=1.2,
        label=labels)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.title(prefix + 'Planecode classification')
    plt.ylabel('Events')
    plt.xlabel('Plane Code ')
    plt.savefig('multperbin'+ suffix + q2title +'.pdf')
    plt.close('all')

def wdistplot(Wdist, bins, label, title, ylog = False, xlog = False):
    plt.figure()
    fig, ax = plt.subplots()
    plt.grid(True)
    if ylog:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    plt.title('W distribution')
    plt.xlabel('W [GeV]')
    plt.ylabel('Events')
    plt.hist(Wdist, bins=bins, histtype=u'step', linewidth=1.2, label=label)
    ax.set_axisbelow(True)
    plt.legend()
    if not ylog or not xlog:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    plt.savefig(title)
    plt.close()

#--------------------------------------------------------------------------------------------------------------------------
#--------------------------Here starts the second function called from main------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
def matplt(actuals, data_dict, W, Q2, n_tracks=None,DIS=None, low_prob=0.0, probs=None, categories=194):

    #Used to convert to int type (eg. from 184. to 184)
    actuals = actuals.astype(np.int)
    predictions = data_dict['predictions'].astype(np.int)
    predictions = np.squeeze(predictions)
    probabilities = data_dict['probabilities']

#----------------------------------cuts----------------------------------------
    if DIS == "DIS":
        DoLimQ2 = 1.
        Q2str=' Q\u00b2 > 1 '
    else:
        DoLimQ2 = 0.
        Q2str=' Q\u00b2 > 0 '
    #W limits
    UpLimW      = None # 2*np.log(15.0)
    DoLimW      = 0.0
    #low_prob    = 0.
    init_size = W.shape[0]
    (LOGGER.info('DIS Sample\n') if DoLimQ2 == 1.0
        else LOGGER.info('Non DIS Sample\n'))

    LOGGER.info("Q2 entries: {}".format(Q2.shape[0]))
    LOGGER.info("Predictions: {}".format(predictions.shape[0]))

    #First cut for  Q2 (DIS or Not DIS)
    actuals     = actuals[Q2>DoLimQ2]
    predictions = predictions[Q2>DoLimQ2]
    probabilities = probabilities[Q2>DoLimQ2] #NewOne
    W           = W[Q2>DoLimQ2]

    if probs is not None:
        predictions = predictions[ probs>low_prob ]
        actuals     = actuals[ probs>low_prob ]
        Q2          = Q2[ probs>low_prob ]
        W           = W[ probs>low_prob ]

    if n_tracks is not None:
        n_tracks    = n_tracks[Q2>DoLimQ2]

    #LOGGER.info('actuals now is = {}'.format(actuals))
    #LOGGER.info('actuals size is = {}'.format(actuals.size))
    #LOGGER.info('predictions now is = {}'.format(predictions))
    #LOGGER.info('predictions size is = {}'.format(predictions.size))
    #LOGGER.info('Q2 now is = {}'.format(Q2))
    #LOGGER.info('Q2 size is = {}'.format(Q2.size))
    #LOGGER.info('W now is = {}'.format(W))
    #LOGGER.info('W size is = {}'.format(W.size))

    if UpLimW is not None:
        probabilities = probabilities[W<UpLimW]
        actuals     = actuals[W<UpLimW]
        predictions = predictions[W<UpLimW]
        if n_tracks is not None:
            n_tracks    = n_tracks[W<UpLimW]
        W           = W[W<UpLimW]

    if DoLimW is not None:
        probabilities = probabilities[W>DoLimW]
        actuals     = actuals[W>DoLimW]
        predictions = predictions[W>DoLimW]
        if n_tracks is not None:
            n_tracks    = n_tracks[W>DoLimW]
        W           = W[W>DoLimW]

    #Vector which take the size of the nevents and in each row select the column prediction value
    #if the first value in the prediction is "182" the value asigned will be whose which are in that
    #column number. should be a ~1 number if the prediction is working well
    prob_lim = probabilities[np.arange(predictions.size), predictions]

#-------File temporaly to compare----------------------------------------
    with open('maxprob.txt', 'w') as f:
        for item in prob_lim:
            f.write("%s\n" % item)

    with open('actuals.txt', 'w') as f:
        for item2 in actuals:
            f.write("%s\n" % item2)

    with open('prediction.txt', 'w') as f:
        for item3 in predictions:
            f.write("%s\n" % item3)
#-----------------------------------------------------------------------
    LOGGER.info('prob_lim= {}'.format(prob_lim))
    LOGGER.info('prob_lim size= {}'.format(prob_lim.size))
    LOGGER.info('actuals= {}'.format(actuals))
    LOGGER.info('actuals size= {}'.format(actuals.size))

    actuals     = actuals[prob_lim>low_prob]
    predictions = predictions[prob_lim>low_prob]
    probabilities = probabilities[prob_lim>low_prob]
    if n_tracks is not None:
        n_tracks    = n_tracks[prob_lim>low_prob]
    W           = W[prob_lim>low_prob]

    UpLimW = W.max() # 2*np.log(15.0)
    DoLimW = W.min()

    LOGGER.info('W max for this sample = {} , W min for this sample = {}'.format(UpLimW,DoLimW))

    norm = predictions.size
    LOGGER.info("There are {} for W events".format(W.shape[0]))

    #To calculate the variance
    exp_v = np.copy(probabilities)
    exp_v = np.mean(exp_v, axis=0)
    var = np.copy(probabilities)
    var[np.arange(predictions.size), predictions] -= 1
    var = np.sqrt(np.mean(var**2, axis=0))

    #DSCAL
    if(categories == 213):
        label = ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
                '21','22','23','24','25','26','27','28','29','20','31','32','33','34','35','36','37','38','39','40',
                    '41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60',
                        '61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80',
                            '81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100',
                                '101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120',
                                    '121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140',
                                        '141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160',
                                            '161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180',
                                                '181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200',
                                                    '201','202','203','204','205','206','207','208','209','210','211','212','213')

#--- 0.9 < W < 1.4 GeV ------To take differents W regions--------------------------------
    actuals0     = actuals[ W<1.4 ]
    predictions0 = predictions[ W<1.4 ]
    probabilities0 = probabilities[ W<1.4 ]
    if n_tracks is not None:
        n_tracks0    = n_tracks[ W<1.4 ]

    # 1.4 < W < 2 GeV
    actuals1     = actuals[W<2]
    actuals1     = actuals1[W[W<2]>1.4]
    predictions1 = predictions[W<2]
    predictions1 = predictions1[W[W<2]>1.4]
    probabilities1 = probabilities[W<2]
    probabilities1 = probabilities1[W[W<2]>1.4]
    if n_tracks is not None:
        n_tracks1    = n_tracks[W<2]
        n_tracks1    = n_tracks1[W[W<2]>1.4]

    # W > 2 GeV
    actuals2     = actuals[W>2]
    predictions2 = predictions[W>2]
    probabilities2 = probabilities[W>2]
    if n_tracks is not None:
        n_tracks2    = n_tracks[W>2]
#---------------------------------------------------------------------------------------------
    # W multiplicities distribution
    W0A = W[actuals == 175]
    W1A = W[actuals == 176]
    W2A = W[actuals == 177]
    W3A = W[actuals == 178]
    W4A = W[actuals == 179]
    W5A = W[actuals == 180]
    W0P = W[predictions == 175]
    W1P = W[predictions == 176]
    W2P = W[predictions == 177]
    W3P = W[predictions == 178]
    W4P = W[predictions == 179]
    W5P = W[predictions == 180]
    if n_tracks is not None:
        W0T = W[n_tracks == 175]
        W1T = W[n_tracks == 176]
        W2T = W[n_tracks == 177]
        W3T = W[n_tracks == 178]
        W4T = W[n_tracks == 179]
        W5T = W[n_tracks == 180]

    # W dist mean per multiplicity
    WmeanA = np.array([(W0A).mean(), (W1A).mean(), (W2A).mean(), (W3A).mean(), (W4A).mean(), (W5A).mean()])
    WmeanP = np.array([(W0P).mean(), (W1P).mean(), (W2P).mean(), (W3P).mean(), (W4P).mean(), (W5P).mean()])
    if n_tracks is not None:
        WmeanT = np.array([(W0T).mean(), (W1T).mean(),(W2T).mean(), (W3T).mean(),(W4T).mean(), (W5T).mean()])
    WmeanA = 2*np.log(WmeanA)
    WmeanP = 2*np.log(WmeanP)

    if n_tracks is not None:
        WmeanT = 2*np.log(WmeanT)
        LOGGER.info('\nTrack based values'.format(Wmeant))

    mults = range(175,181,1)
    fitA = np.polyfit(mults, WmeanA, 1)
    fit_fnA = np.poly1d(fitA)
    fitP = np.polyfit(mults, WmeanP, 1)
    fit_fnP = np.poly1d(fitP)
    LOGGER.debug('Actuals fit: {}'.format(fit_fnA))
    LOGGER.debug('ML predtions fit: {}'.format(fit_fnP))

    bins = np.logspace(np.log10(DoLimW-0.1*DoLimW),np.log10(UpLimW), 75)
#    bins = np.linspace(DoLimW,UpLimW, 75)
    WP = (W, W0P, W1P, W2P, W3P, W4P, W5P)
    labelP = ('All', 'ML pred 175', 'ML pred 176','ML pred 177', 'ML pred 178', 'ML pred 179', 'ML pred 180')
    #WA = (W, W0A, W1A, W2A, W3A, W4A, W5A)
    labelA = ('All', 'Truth 175', 'Truth 176', 'Truth 177','Truth 178', 'Truth 179', 'Truth 180')

    if n_tracks is not None:
        WT = (W, W0T, W1T, W2T, W3T, W4T, W5T)
        labelT = ('All', 'Tracked 0', 'Tracked 1', 'Tracked 2','Tracked 3', 'Tracked 4', 'Tracked 5')
#---------------------------------------------------------------------------------------------------------------------
    final_size = W.shape[0]
    lost_size = float(init_size - final_size)

    LOGGER.info(
        "Ratio lost events {}\n".format((lost_size)/float(init_size)))

#----------------------------------plots---------------------------------------

    LOGGER.info('probabilities shape: {}'.format(probabilities.shape))
    LOGGER.info('predictions shape: {}'.format(predictions.shape))
    #LOGGER.info('probabilities[classifications==i]: {} '.format(probabilities[predictions==0].shape))

    #To know from which column to which one to take in the probabilities plot_array
    #In the case for small images choose min=0, max=19
    #In the case of whole images choose min=175, max=209
    if(categories > 100):  #DSCAL
        min=165
        max=170 #Changeeeeee
    if(categories < 100):   #ECAL: 20, ECALplus 4 tracker modules: 28, DSCAL: 34, ECAL 4 Tracker and 2 HCAL: 32
        min=1
        max=categories


    probabilities_class = [1 - probabilities[predictions==i][:,i]
        for i in range(min,max)]
    probabilities_nonclass = [probabilities[predictions!=i][:,i]
        for i in range(min,max)]

    probs_minus = probs_sum(probabilities_class, label=label,title='probs_minus',suffix2=Q2str)
    probs_plus = probs_sum(probabilities_nonclass, label=label,title='probs_plus',suffix2=Q2str)
    probs_minus = np.array(probs_minus)
    probs_plus = np.array(probs_plus)

    probabilities_class0 = [1 - probabilities0[predictions0==i][:,i]
        for i in range(min,max)]
    probabilities_nonclass0 = [probabilities0[predictions0!=i][:,i]
        for i in range(min,max)]

    probs_minus0 = probs_sum(probabilities_class0, label=label,title='probs_minus0',suffix2=Q2str)
    probs_plus0 = probs_sum(probabilities_nonclass0, label=label,title='probs_plus0',suffix2=Q2str)
    probs_minus0 = np.array(probs_minus0)
    probs_plus0 = np.array(probs_plus0)

    probabilities_class1 = [1 - probabilities1[predictions1==i][:,i]
        for i in range(min,max)]
    probabilities_nonclass1 = [probabilities1[predictions1!=i][:,i]
        for i in range(min,max)]

    probs_minus1 = probs_sum(probabilities_class1, label=label,title='probs_minus1',suffix2=Q2str)
    probs_plus1 = probs_sum(probabilities_nonclass1, label=label,title='probs_plus1',suffix2=Q2str)
    probs_minus1 = np.array(probs_minus1)
    probs_plus1 = np.array(probs_plus1)

    probabilities_class2 = [1 - probabilities2[predictions2==i][:,i]
        for i in range(min,max)]
    probabilities_nonclass2 = [probabilities2[predictions2!=i][:,i]
        for i in range(min,max)]
    probs_minus2 = probs_sum(probabilities_class2, label=label,title='probs_minus2',suffix2=Q2str)
    probs_plus2 = probs_sum(probabilities_nonclass2, label=label,title='probs_plus2',suffix2=Q2str)
    probs_minus2 = np.array(probs_minus2)
    probs_plus2 = np.array(probs_plus2)


    mult_stacked_plot(actuals, predictions, suffix2=Q2str)
    mult_stacked_plot(actuals0, predictions0, suffix='0', suffix2=Q2str, norm=norm)
    mult_stacked_plot(actuals1, predictions1, suffix='1', suffix2=Q2str, norm=norm)
    mult_stacked_plot(actuals2, predictions2, suffix='2', suffix2=Q2str, norm=norm)


#------------Confusion matrices-----------------------
    conf_mat(actuals, predictions,categories, suffix2=Q2str)
    conf_mat(actuals0, predictions0,categories, '0', suffix2=Q2str)
    conf_mat(actuals1, predictions1,categories, '1', suffix2=Q2str)
    conf_mat(actuals2, predictions2,categories, '2', suffix2=Q2str)

#    conf_mat(actuals, n_tracks, 'tr')
#    conf_mat(actuals0, n_tracks0, 'tr0')
#    conf_mat(actuals1, n_tracks1, 'tr1')
#    conf_mat(actuals2, n_tracks2, 'tr2')

    if n_tracks is not None:
        mult_plot(actuals, predictions, n_tracks, suffix2=Q2str)
        mult_plot(actuals0, predictions0, n_tracks0, '0', suffix2=Q2str)
        mult_plot(actuals1, predictions1, n_tracks1, '1', suffix2=Q2str)
        mult_plot(actuals2, predictions2, n_tracks2, '2', suffix2=Q2str)
    else:
        mult_plot(actuals, predictions, suffix2=Q2str )
        mult_plot(actuals0, predictions0, suffix='0', suffix2=Q2str)
        mult_plot(actuals1, predictions1, suffix='1', suffix2=Q2str)
        mult_plot(actuals2, predictions2, suffix='2', suffix2=Q2str)

    #wdistplot(WP, bins, labelP, title="WPredDist.pdf", ylog = True, xlog=True)
    #wdistplot(WA, bins, labelA, title="WTruthDist.pdf", ylog = True, xlog=True)
    #if n_tracks is not None:
    #    wdistplot(WT, bins, labelT, title="WTrackedDist.pdf", ylog = True, xlog=True)
#
    plt.figure()
    plt.grid(True)
    plt.title(r'Multiplicity as function of $\langle W\rangle^2$')
    plt.xlabel(r'$\langle W \rangle^2$ [GeV]')
    plt.ylabel('Multiplicity')
#    plt.plot( fit_fnA(mults), mults, 'r--', label='Truth Fit ' )
#    plt.plot( fit_fnP(mults), mults, 'b--', label='ML Pred Fit' )
    #plt.plot( WmeanA, mults, '^', color='red', label='Truth' )
    #plt.plot( WmeanP, mults, 'o', color='blue', label='ML pred' )
    #if n_tracks is not None:
    #    plt.plot( WmeanT, mults, 's', color='green', label='Tracked' )
    #plt.legend()
#    plt.xscale('log')
    #plt.savefig('MultMeanW.pdf')
    plt.close()


def probs_sum(all_probs, mkplot=True, label=None, title='probs',suffix2=''):
    if mkplot:
        plt.figure()
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.set_axisbelow(True)
        plt.hist(
            all_probs,
            bins=np.linspace(0, 1, 51),
            histtype=u'step',
            label=label,
            density=True,
            linewidth=0.9
            )
        ax.ticklabel_format(
            axis='y',
            style='sci',
            scilimits=(0, 0) #0, 3
        )


        if ' Q\u00b2 > 1 ' in suffix2:
            q2title = 'Q1'
        else:
            q2title = 'Q0'

        #plt.yscale('log')
        plt.legend(fontsize='small')
        plt.title('Probability distribution '+suffix2 )
        plt.xlabel('Probability' )
        plt.ylabel('% Events')
        plt.savefig(title +q2title + '.pdf')
        plt.close('all')

    sum_probs = []
    mean_probs = []

    for i, probs in enumerate(all_probs):
        sum_probs.append(np.sum(probs))

    return sum_probs



if __name__ == '__main__':
    try:
        # Get prediction path as a parameter check how many entries are as sys.arg
        #Usually to run is python matplot.py $path_prediction_file.
        if len(sys.argv) == 2:
            filename = sys.argv[1]
            DIS = ''
        elif len(sys.argv) == 3:
            filename = sys.argv[1]
            DIS = sys.argv[2]
        else:
            usage()

        #Select the number of categories to train: 194 209. image opts: 'big' ot 'small'
        categories = 213
        image = 'big'
        # Take the path of the current directory.
        cwd = os.getcwd()
        print('We are in {}'.format(cwd)) #Added by Oscar

        # Take the filename and split into:
        basedir =  filename.split('/')[-2]
        playlist = filename.split('_')[-1]
        playlist = playlist.split('.')[0]
        stepmodel =  filename.split('_')[-2]

        plotdir = os.path.join(cwd, "plots")
        basedir = os.path.join(plotdir, basedir)
        pldir = os.path.join(basedir, playlist)
        logdir = os.path.join(pldir, stepmodel)
        logdir += DIS

        print('the log dis is: {}'.format(logdir)) #Added by Oscar
        #LOGGER.info('the log dis is: {}'.format(logdir))

# This part is going creating the final directory to store the plots
        for directory in [plotdir, basedir, pldir, logdir]:
            if not os.path.exists(directory):
                print('{} will be created.'.format(directory))
                os.mkdir(directory)
            else:
                print('{} will be used.'.format(directory))

#Creates the logfile or reemplace de last logfilename
        logfilename = os.path.join(
            logdir, filename.split('/')[-1].split('.')[0] + DIS + '.log')
        if os.path.exists(logfilename):
            os.remove(logfilename)
            #print('{} removed.'.format(logfilename))
            #print('{} will be created.'.format(logfilename))

#Creates the format to use the logfile
        formatter = logging.Formatter('%(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(logfilename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        LOGGER = logging.getLogger(__name__)
        LOGGER.setLevel(logging.DEBUG)

        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(stream_handler)

        LOGGER.info('We are in {}'.format(cwd)) #Oscar
        #LOGGER.info('Basedir: {}'.format(basedir))
        #LOGGER.info('playlist: {}'.format(playlist))
        #LOGGER.info('stepmodel: {}'.format(stepmodel))
        #LOGGER.info('plotdir: {}'.format(plotdir))
        #LOGGER.info('basedir: {}'.format(basedir))
        #LOGGER.info('pldir: {}'.format(pldir))
        #LOGGER.info('logdir: {}'.format(logdir))
        #LOGGER.info('the log dis is: {}'.format(logdir))

        os.chdir(logdir)
        LOGGER.info("Playlist: {}".format(playlist))
        #hdf5 = '/lfstev/e-938/omorenop/hdf5/me1Amc_86-626499-3/hadmultkineimgs_127x94_{}.hdf5'.format(
        #  playlist)
        #hdf5 = '/data/omorenop/minerva/hdf5/201911/NukeTrain/hadmultkineimgs_127x94_{}.hdf5'.format(playlist)
        #hdf5 = '/data/omorenop/minerva/hdf5/201911/Nuke119-Evaluate-040/hadmultkineimgs_127x94_me1Amc.hdf5'
        #hdf5 = '/data/omorenop/minerva/hdf5/201911/Nuke119-Test-0402/hadmultkineimgs_127x94_me1Amc.hdf5'
        #hdf5 = '/lfstev/e-938/omorenop/hdf5/NukeECAL-Evaluate-lattice/hadmultkineimgs_127x94_me1Amc.hdf5'
        #hdf5 = '/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test/hadmultkineimgs_127x94_me1AmcDSCALTest.hdf5'
        #hdf5 = '/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-RawCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
        #hdf5 = '/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-VtxPlaneCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
        #hdf5 = '/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-zLowHighApril/hadmultkineimgs_127x94_me1Amc.hdf5'
        hdf5 = '/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-WholeDetectorMix/hadmultkineimgs_127x94_me1Amc.hdf5'

        reader = HDF5Reader(hdf5)
        nevents = reader.openf()

        LOGGER.info('The nevents are: {}'.format(nevents))

        #index_dict is a dictonary which is used to save and load the column numbers where
        # the predictions and probabilities (classifications) are in the prediction file. Usually
        #that info are in the 4th column and so on.

        # For DSCAL
        if(categories == 213):
            index_dict = {
                'predictions': [4],
                'probabilities': [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                            21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                                    61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
                                        81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
                                            101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,
                                                121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,
                                                    141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
                                                        161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,
                                                            181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,
                                                                201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217]

                                                                }

        data_dict = file2npy(filename, index_dict) #, W, Q2, n_tracks actuals,

        #LOGGER.info('data_dict: {}'.format(data_dict)) #Print prediction and probabilities arrays e.g. 'predictions': array([[173.], [143.],... 'probabilities': array([[1.2384001e-12, 4.7781833e-17, 1.7108721e-16, ..., 4.2754018e-23
        nevents =  data_dict['predictions'].shape[0]

        LOGGER.info('nevents = {}'.format(nevents)) #To be sure about the number of entries

        if(image == 'big'):
            actuals = reader.get_key(0, nevents, key='vtx_data/planecodes')
        if(image == 'small'):
            actuals = reader.get_key(0, nevents, key='vtx_data/planecodesDScal')

        LOGGER.info('Actuals: {}'.format(actuals))
        #sys.exit()

        W = reader.get_key(0, nevents, key='gen_data/W')/1000
        Q2 = reader.get_key(0, nevents, key='gen_data/Q2')/1000000

#        n_tracks[ n_tracks>5 ] = 5
        n_tracks = None
        low_prob=0.

        matplt(actuals, data_dict, W, Q2, n_tracks, DIS=DIS, low_prob=low_prob, categories = categories)

        #mc_plots(actuals,data_dict, DIS=DIS, low_prob=low_prob)
        #mult_plot([data_classifications, mc_classifications], suffix='_data',label=["data","mc"])

        os.chdir(cwd)

    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
