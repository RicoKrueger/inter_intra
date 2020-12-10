import numpy as np
import pandas as pd
import pickle

np.random.seed(4711)

R = 10

for r in np.arange(R):

    ###
    #Load data in long format
    ###
    
    df_long = pd.read_table('kw_dataset08.txt', 
                            header=None, index_col=False,
                            names=[
                                'indID', 'indobsID', 'altID', 'chosen',
                                'price', 'overseas', 'airline', 'length_stay',
                                'meal', 'tours', 'peak', 'four_star',
                                'length_trip', 'culture', 'distance', 'pool',
                                'help', 'individual', 'beach', 'brand'
                                ])
    
    attributes = [
        'price', 'overseas', 'airline', 'length_stay',
        'meal', 'tours', 'peak', 'four_star',
        'length_trip', 'culture', 'distance', 'pool',
        'help', 'individual', 'beach', 'brand']
    for a in attributes:
        df_long[a] = np.where(df_long[a] == -1, 0, df_long[a])
    df_long['beach_pool'] = np.where(
        np.logical_or(df_long['beach'], df_long['pool']), 1, -1)
    
    n_ind = df_long['indID'].nunique()
    obs_per_ind = int(df_long['indobsID'].max())
    n_obs = n_ind * obs_per_ind
    n_alt = int(df_long['altID'].max())
    df_long['obsID'] = np.repeat(np.arange(n_obs) + 1, n_alt)
    
    attributes_rnd = ['overseas', 'length_stay', 'four_star', 'beach_pool']
    attributes_fix = ['price', 'meal', 'distance', 'tours', 'individual']
    
    
    ###
    #Shuffle data
    ###
    
    df_long['indID'] = np.repeat(np.random.choice(np.arange(n_ind) + 1, size=n_ind, 
                                                  replace=False), 
                                 obs_per_ind * n_alt)
    
    df_long['indobsID'] = np.repeat(
        np.concatenate([np.random.choice(np.arange(obs_per_ind) + 1, 
                                         size=obs_per_ind, replace=False)
                        for i in np.arange(n_ind)]), n_alt)
    
    df_long = df_long[df_long['indobsID'] <= 21].copy()
    obs_per_ind = int(df_long['indobsID'].max())
    n_obs = n_ind * obs_per_ind
    
    df_long.sort_values(['indID', 'indobsID'], inplace=True)
    df_long['obsID'] = np.repeat(np.arange(n_obs) + 1, n_alt)
    
    df_long = df_long.astype('int')
    
    ###
    #Select training and test data
    ###
    
    n_valB = 50
    n_valW = 200
    
    #Between
    df_long['test_b'] = np.concatenate(
        [np.repeat(np.random.choice(np.append(np.zeros((obs_per_ind - 1,)), 1), 
                                    size=obs_per_ind, replace=False), n_alt) 
         if i > (n_ind - n_valB) else np.zeros((obs_per_ind * n_alt,))
         for i in df_long['indID'].unique()])
    
    #Within
    df_long['test_w'] = np.logical_and(df_long['indID'] <= n_valW,
                                       (df_long['obsID'] % obs_per_ind) == 0) * 1
    
    df_train = df_long[(df_long['indID'] <= (n_ind - n_valB)) 
                       & ((df_long['obsID'] % obs_per_ind) > 0)].copy()
    df_train['obsID'] = np.repeat(np.arange(df_train['obsID'].nunique()) + 1, n_alt)
    
    df_test_b = df_long[df_long['test_b'] == 1].copy()
    df_test_b['obsID'] = np.repeat(np.arange(df_test_b['obsID'].nunique()) + 1, n_alt)
    
    df_test_w = df_long[df_long['test_w'] == 1].copy()
    df_test_w['obsID'] = np.repeat(np.arange(df_test_w['obsID'].nunique()) + 1, n_alt)
    
    ###
    #Store data
    ###
    
    data = {'indID': df_train['indID'].values, 'obsID': df_train['obsID'].values, 
            'altID': df_train['altID'].values, 'chosen': df_train['chosen'].values,
            'xRnd': df_train[attributes_rnd].values, 'nRnd': len(attributes_rnd),
            'xFix': df_train[attributes_fix].values, 'nFix': len(attributes_fix),
            'indID_valB': df_test_b['indID'].values, 'obsID_valB': df_test_b['obsID'].values, 
            'altID_valB': df_test_b['altID'].values, 'chosen_valB': df_test_b['chosen'].values,
            'xRnd_valB': df_test_b[attributes_rnd].values, 'xFix_valB': df_test_b[attributes_fix].values,
            'indID_valW': df_test_w['indID'].values, 'obsID_valW': df_test_w['obsID'].values, 
            'altID_valW': df_test_w['altID'].values, 'chosen_valW': df_test_w['chosen'].values,
            'xRnd_valW': df_test_w[attributes_rnd].values, 'xFix_valW': df_test_w[attributes_fix].values
            }
    
    filename = 'data_route_r{}'.format(r)
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()