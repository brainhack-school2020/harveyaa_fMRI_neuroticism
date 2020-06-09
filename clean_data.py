import numpy as np
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--netmats_file",help="Path to netmats file",dest='netmats_file')   
    parser.add_argument("--traits_file",help="Path to personality traits file",dest='traits_file')
    parser.add_argument("--subjectIDs_file",help="Path to subjectIDs file",dest='subjectIDs_file')
    args = parser.parse_args()
 
    netmats = np.loadtxt(args.netmats_file)

    # behaviour from full HCP (~1200 subjects)
    behaviour = pd.read_csv(args.traits_file,index_col=0)

    # all subjects from recon2 in PTN release (n = 812)
    subjectIDs = np.loadtxt(args.subjectIDs_file)
    
    behaviour = behaviour[['NEOFAC_N','NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_E']].loc[subjectIDs]
    #subs_null = behaviour[behaviour.isnull()].index.tolist()
    subs_null = [109830, 614439]
    # drop subjects with missing values
    behaviour = behaviour.drop(subs_null, axis=0)
    id_to_array = dict(zip(subjectIDs,range(len(subjectIDs))))
    netmats = np.delete(netmats,[id_to_array[sub] for sub in subs_null], axis=0)
    
    #save the cleaned up data
    np.savetxt('netmats2_clean.txt',netmats)
    behaviour.to_csv('all_traits.csv')

