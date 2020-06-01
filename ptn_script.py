#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import image, plotting
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#mat is shape n_subjectsXn_edges y is shape n_edges
def correlate_edges(mat,y):
    edge_corr = np.zeros((mat.shape[1],2))    

    for i in range(mat.shape[1]):
        #correlation, p value
        edge_corr[i,0], edge_corr[i,1] = pearsonr(netmats_train[:,i],b_train)
        #if edge is all zeros, replace the nan value with 0 and set the pvalue to 1 so we ignore it
        if(np.isnan(edge_corr[i,0])): 
            edge_corr[i,0] = 0
            edge_corr[i,1] = 1
    return edge_corr

#correlations is shape n_edgesX2
def filter_edges(correlations, thresh=0.01):
    edges = pd.DataFrame(correlations, columns = ['corr','p'])
    sig_edges = edges[edges['p']<thresh]
    pos_edges = sig_edges[sig_edges['corr']>0].index.tolist()
    neg_edges = sig_edges[sig_edges['corr']<0].index.tolist()
    sig_edges = sig_edges.index.tolist()
    return sig_edges, pos_edges, neg_edges

#mat is shape n_subjectsXn_edges edges is a list of chosen edges
def get_scores(mat, edges):
    mask = np.zeros(net.shape[1])
    mask[edges]=1
    return np.matmul(mat,mask)

if __name__ == "__main__":
    # connectivity matrices, 200x200 matrices in vector 810 subjects (shape 810x40000)
    netmats = np.loadtxt('netmats2_clean.txt')
    neuroticism = np.loadtxt('neuroticism.csv',skiprows=1,delimiter = ',')

    netmats_train, netmats_test, neuro_train, neuro_test = train_test_split(netmats,neuroticism)

    edge_correlations = correlate_edges(netmats_train,neuro_train)
    significant_edges, positive_edges, negative_edges = filter_edges(edge_correlations)

    positive_scores_train = get_scores(netmats_train,positive_edges).reshape(-1,1)
    positive_scores_test = get_scores(netmats_test,positive_edges).reshape(-1,1)
    negative_scores_train = get_scores(netmats_train,negative_edges).reshape(-1,1)
    negative_scores_test = get_scores(netmats_test,negative_edges).reshape(-1,1)

    #positive correlation scores
    positive_model = LinearRegression().fit(positive_scores_train,neuro_train.renshape(-1,1))
    positive_pred = positive_model.predict(positive_scores_test)

    #negative correlation scores
    negative_model = LinearRegression().fit(negative_scores_train,neuro_train.renshape(-1,1))
    negative_pred = positive_model.predict(negative_scores_test)

    #use both scores in multiple regression
    both_scores_train = np.concatenate((positive_scores_train,negative_scores_train),axis=1)
    both_scores_test = np.concatenate((positive_scores_test,negative_scores_test),axis=1)
    both_model = LinearRegression().fit(both_scores_train,neuro_train.reshape(-1,1))
    both_pred = multi_model.predict(both_scores_test)

    MSE = {'positive':mean_squared_error(neuro_test,positive_pred),
           'negative':mean_squared_error(neuro_test,negative_pred),
           'both':mean_squared_error(neuro_test,both_pred)}

    #WRITE DATA OUT