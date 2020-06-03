#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import warnings

def correlate_edges(mat,y):
    """
    Correlation between edges in connectivity matrices and feature being predicted.
    Inputs
    -------
    mat: ndarray (n_subjects,n_edges)
        flattened connectivity matrix for each subject
    y: ndarray (n_subjects,)
        feature being predicted for each subject
    Returns
    -------
    edge_corr: ndarray (n_edges,2)
        correlations and p-value for each edge
        edge_corr[i,0] correlation
        edge_corr[i,1] p-value
    """
    edge_corr = np.zeros((mat.shape[1],2))    

    for i in range(mat.shape[1]):
        #correlation, p value
        edge_corr[i,0], edge_corr[i,1] = pearsonr(mat[:,i],y)
        #if edge is all zeros, replace the nan value with 0 and set the pvalue to 1 so we ignore it
        if(np.isnan(edge_corr[i,0])): 
            edge_corr[i,0] = 0
            edge_corr[i,1] = 1
    return edge_corr

def filter_edges(correlations, thresh=0.01):
    """
    Filter edges based on p-value, then by positive or negative correlation.
    Inputs
    -------
    correlations: ndarray (n_edges,2)
        correlations and p-value for each edge
        edge_corr[i,0] correlation
        edge_corr[i,1] p-value
    thresh: float
        p-value cut off for significance
    Returns
    -------
    sig_edges: list
        all significantly correlated edges
    pos_edges: list
        significantly positively correlated edges
    neg_edges: list
        significantly negatively correlated edges
    """
    edges = pd.DataFrame(correlations, columns = ['corr','p'])
    sig_edges = edges[edges['p']<thresh]
    pos_edges = sig_edges[sig_edges['corr']>0].index.tolist()
    neg_edges = sig_edges[sig_edges['corr']<0].index.tolist()
    sig_edges = sig_edges.index.tolist()
    return sig_edges, pos_edges, neg_edges
    
#mat is shape n_subjectsXn_edges edges is a list of chosen edges
def get_scores(mat, edges):
    """
    Produces summary scores for CPM
    Inputs
    -------
    mat: ndarray (n_subjects,n_edges)
        flattened connectivity matrix for each subject
    edges: list
        selected edges to use in score calculation
    Returns
    -------
    float
        summary score for CPM
    """
    mask = np.zeros(mat.shape[1])
    mask[edges]=1
    return np.matmul(mat,mask)

def leave_one_out_CPM(mat,y,thresh=0.01):
    """
    Do CPM prediction using leave one out strategy.
    Inputs
    -------
    mat: ndarray (n_subjects,n_edges)
        flattened connectivity matrix for each subject
    y: ndarray (n_subjects,)
        feature being predicted for each subject
    Returns
    -------
    prediction: ndarray (n_subjects,4)
        prediction for each subject based on different strategies
        prediction[i,0] use all significantly correlated edges in combined score
        prediction[i,1] use only significant positvely correlated edges for score
        prediction[i,2] use only significant negatively correlated edges for score
        prediction[i,3] use both sig. pos. and neg. scores in multiple regression
    edge_count: ndarray (n_subjects,3)
        number of times (max n_subjects) each edge was significantly correlated
        edge_count[i,0] all significant edges
        edge_count[i,1] positive edges
        edge_count[i,2] negative edges
    MSE: list
        MSE for each strategy of prediction ordered as above
    """
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
    
    MSE = [['combined',mean_squared_error(y,prediction[:,0])],
            ['positive',mean_squared_error(y,prediction[:,1])],
           ['negative',mean_squared_error(y,prediction[:,2])],
           ['multiple_reg',mean_squared_error(y,prediction[:,3])]]
    
    return prediction,edge_count,MSE

def leave_one_out_LR(mat,y,thresh=0.01):
    """
    Fit a multiple regression model on selected edges as features using leave one out strategy.
    Inputs
    -------
    mat: ndarray (n_subjects,n_edges)
        flattened connectivity matrix for each subject
    y: ndarray (n_subjects,)
        feature being predicted for each subject
    Returns
    -------
    prediction: ndarray (n_subjects,4)
        prediction for each subject based on different strategies
        prediction[i,0] use all significantly correlated edges
        prediction[i,1] use only significant positvely correlated edges
        prediction[i,2] use only significant negatively correlated edges
    edge_count: ndarray (n_subjects,3)
        number of times (max n_subjects) each edge was significantly correlated
        edge_count[i,0] all significant edges
        edge_count[i,1] positive edges
        edge_count[i,2] negative edges
    MSE: list
        MSE for each strategy of prediction ordered as above
    """
    n_subjects = mat.shape[0]
    n_edges = mat.shape[1]

    #significant, positive, negative
    prediction = np.zeros((n_subjects,3))
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
        significant_edges_train = mat_train[:,sig_edges]
        significant_edges_test = mat_test[:,sig_edges]
        #use only positive significant edges
        positive_edges_train = mat_train[:,pos_edges]
        positive_edges_test = mat_test[:,pos_edges]
        #use only negative significant edges
        negative_edges_train = mat_train[:,neg_edges]
        negative_edges_test = mat_test[:,neg_edges]

        significant_model = LinearRegression().fit(significant_edges_train,y_train.reshape(-1,1))
        positive_model = LinearRegression().fit(positive_edges_train,y_train.reshape(-1,1))
        negative_model = LinearRegression().fit(negative_edges_train,y_train.reshape(-1,1))        
        
        #count how many times each edge is significant for each type of edges
        mask = np.zeros((n_edges,3))
        mask[sig_edges,0]=1
        mask[pos_edges,1]=1
        mask[neg_edges,2]=1
        edge_count = edge_count + mask

        prediction[i,0]=significant_model.predict(significant_edges_test)
        prediction[i,1]=positive_model.predict(positive_edges_test)
        prediction[i,2]=negative_model.predict(negative_edges_test)
        
        if (i%10 ==0): print('Fold: ',i)
        i = i+1
    
    MSE = [['significant',mean_squared_error(y,prediction[:,0])],
            ['positive',mean_squared_error(y,prediction[:,1])],
           ['negative',mean_squared_error(y,prediction[:,2])]]
    
    return prediction,edge_count,MSE

def leave_one_out_SVR(mat,y,thresh=0.01):
    """
    Fit an SVR model on selected edges as features using leave one out strategy.
    Inputs
    -------
    mat: ndarray (n_subjects,n_edges)
        flattened connectivity matrix for each subject
    y: ndarray (n_subjects,)
        feature being predicted for each subject
    Returns
    -------
    prediction: ndarray (n_subjects,4)
        prediction for each subject based on different strategies
        prediction[i,0] use all significantly correlated edges
        prediction[i,1] use only significant positvely correlated edges
        prediction[i,2] use only significant negatively correlated edges
    edge_count: ndarray (n_subjects,3)
        number of times (max n_subjects) each edge was significantly correlated
        edge_count[i,0] all significant edges
        edge_count[i,1] positive edges
        edge_count[i,2] negative edges
    MSE: list
        MSE for each strategy of prediction ordered as above
    """
    n_subjects = mat.shape[0]
    n_edges = mat.shape[1]

    #significant, positive, negative
    prediction = np.zeros((n_subjects,3))
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
        significant_edges_train = mat_train[:,sig_edges]
        significant_edges_test = mat_test[:,sig_edges]
        #use only positive significant edges
        positive_edges_train = mat_train[:,pos_edges]
        positive_edges_test = mat_test[:,pos_edges]
        #use only negative significant edges
        negative_edges_train = mat_train[:,neg_edges]
        negative_edges_test = mat_test[:,neg_edges]

        significant_model = SVR().fit(significant_edges_train,y_train)
        positive_model = SVR().fit(positive_edges_train,y_train)
        negative_model = SVR().fit(negative_edges_train,y_train)        
        
        #count how many times each edge is significant for each type of edges
        mask = np.zeros((n_edges,3))
        mask[sig_edges,0]=1
        mask[pos_edges,1]=1
        mask[neg_edges,2]=1
        edge_count = edge_count + mask

        prediction[i,0]=significant_model.predict(significant_edges_test)
        prediction[i,1]=positive_model.predict(positive_edges_test)
        prediction[i,2]=negative_model.predict(negative_edges_test)
        
        if (i%10 ==0): print('Fold: ',i)
        i = i+1
    
    MSE = [['significant',mean_squared_error(y,prediction[:,0])],
            ['positive',mean_squared_error(y,prediction[:,1])],
           ['negative',mean_squared_error(y,prediction[:,2])]]
    
    return prediction,edge_count,MSE
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--netmats_file",help="Path to netmats file",dest='netmats_file')   
    parser.add_argument("--traits_file",help="Path to personality traits file",dest='traits_file')
    parser.add_argument("--trait",help="Personality trait to predict, options:'N','E','A','O','C'.",dest='trait')
    parser.add_argument("--method",help="Prediction method, options: 'LR','CPM','SVR'.",dest='method') 
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")
    
    print('loading full data...')
    # connectivity matrices, 200x200 matrices in vector 810 subjects (shape 810x40000)
    netmats = np.loadtxt(args.netmats_file)
    traits = pd.read_csv(args.traits_file, index_col=0)
    col_name = 'NEOFAC_{}'.format(args.trait)
    trait = traits[col_name].to_numpy()

    print('starting {}...'.format(args.method))
    if (args.method == 'CPM'):
        pred, edge_count, MSE = leave_one_out_CPM(netmats,trait)
    if (args.method == 'LR'):
        pred, edge_count, MSE = leave_one_out_LR(netmats,trait)
    if (args.method == 'SVR'):
        pred, edge_count, MSE = leave_one_out_SVR(netmats,trait)
    
    print('writing data...')
    print('MSE ',MSE)
    pd.DataFrame(MSE, columns = ['type','MSE']).to_csv('{}_{}_MSE.csv'.format(col_name,args.method))
    #np.savetxt('{}_{}_edge_count.csv'.format(col_name,args.method),edge_count)
    np.savetxt('{}_{}_predictions.csv'.format(col_name,args.method),pred)