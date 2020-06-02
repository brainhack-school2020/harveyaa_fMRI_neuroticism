#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#mat is shape n_subjectsXn_edges y is shape n_edges
def correlate_edges(mat,y):
    edge_corr = np.zeros((mat.shape[1],2))    

    for i in range(mat.shape[1]):
        #correlation, p value
        edge_corr[i,0], edge_corr[i,1] = pearsonr(mat[:,i],y)
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
    mask = np.zeros(mat.shape[1])
    mask[edges]=1
    return np.matmul(mat,mask)

def leave_one_out_CPM(mat,y,thresh=0.01):
    n_subjects = mat.shape[0]
    n_edges = mat.shape[1]

    #combined, positive, negative, multiple
    prediction = np.zeros((n_subjects,4))
    #for each edge number of subjects for which it was significantly correlated
    edge_count = np.zeros((n_edges,3))
    
    loo = LeaveOneOut()
    i = 0
    for train_index,test_index in loo.split(mat):
        mat_train, mat_test = mat[train_index], mat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #For each of 200x200 edges across subjects find the correlation with neuroticism score
        corr = correlate_edges(mat_train,y_train)
        #Filter edges to significantly correlated (p value below threshold), then into positive and negative correlation
        sig_edges, pos_edges, neg_edges = filter_edges(corr,thresh)
        
        #Create binary masks edges, for each subject count up the values of those edges in their connectivity matrices
        #equivalent to the dot product between the mask and their connectivity matrix (both flattened)
        #use all significant edges in summary score
        combined_scores_train = get_scores(mat_train,sig_edges).reshape(-1,1)
        combined_scores_test = get_scores(mat_test,sig_edges).reshape(-1,1)
        #use only positive significant edges
        positive_scores_train = get_scores(mat_train,pos_edges).reshape(-1,1)
        positive_scores_test = get_scores(mat_test,pos_edges).reshape(-1,1)
        #use only negative significant edges
        negative_scores_train = get_scores(mat_train,neg_edges).reshape(-1,1)
        negative_scores_test = get_scores(mat_test,neg_edges).reshape(-1,1)
        #combine pos and neg scores in multiple regression
        multiple_reg_scores_train = np.concatenate((positive_scores_train,negative_scores_train),axis=1)
        multiple_reg_scores_test = np.concatenate((positive_scores_test,negative_scores_test),axis=1)

        combined_model = LinearRegression().fit(combined_scores_train,y_train.reshape(-1,1))
        positive_model = LinearRegression().fit(positive_scores_train,y_train.reshape(-1,1))
        negative_model = LinearRegression().fit(negative_scores_train,y_train.reshape(-1,1))        
        multiple_reg_model = LinearRegression().fit(multiple_reg_scores_train,y_train.reshape(-1,1))
        
        #count how many times each edge is significant for each type of edges
        mask = np.zeros((n_edges,3))
        mask[sig_edges,0]=1
        mask[pos_edges,1]=1
        mask[neg_edges,2]=1
        edge_count = edge_count + mask

        prediction[i,0]=combined_model.predict(combined_scores_test)
        prediction[i,1]=positive_model.predict(positive_scores_test)
        prediction[i,2]=negative_model.predict(negative_scores_test)
        prediction[i,3]=multiple_reg_model.predict(multiple_reg_scores_test)
        
        if (i%10 ==0): print('CPM fold: ',i)
        i = i+1
    return prediction,edge_count
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--netmats_file",help="Path to netmats file",dest='netmats_file')
    
    parser.add_argument("--traits_file",help="Path to personality traits file",dest='traits_file')
 
    args = parser.parse_args()
    print('loading data...')
    # connectivity matrices, 200x200 matrices in vector 810 subjects (shape 810x40000)
    netmats = np.loadtxt(args.netmats_file)
    traits = pd.read_csv(args.traits_file, index_col=0)
    neuroticism = traits['NEOFAC_N'].to_numpy()

    
    pred, edge_count = leave_one_out_CPM(netmats,neuroticism)

    MSE = [['combined',mean_squared_error(neuroticism,pred[:,0])],
            ['positive',mean_squared_error(neuroticism,pred[:,1])],
           ['negative',mean_squared_error(neuroticism,pred[:,2])],
           ['multiple_reg',mean_squared_error(neuroticism,pred[:,3])]]

    
    #TODO: WRITE DATA OUT
    print(MSE)
    pd.DataFrame(columns = ['type','MSE']).to_csv('neuroticism_MSE.csv')
    np.savetxt('neuroticism_edge_count.csv',edge_count)
    np.savetxt('neuroticism_predictions.csv',pred)