import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix
import random

scaler = StandardScaler()

# ---- math function helper ----
def norm_scale(counts_vector):
    norm = np.linalg.norm(np.array(counts_vector))
    normalized_vector = counts_vector / norm if norm != 0 else counts_vector
    return normalized_vector
def percent_scale(counts_vector):
    norm = sum(np.array(counts_vector))
    normalized_vector = counts_vector / norm
    return normalized_vector
def cal_entropy(counts_vector):
    entropy = -np.sum(counts_vector[counts_vector > 0] * np.log2(counts_vector[counts_vector > 0]))
    return entropy

# ---- data loading and processing ----
# Load expected product exposure distribution (vector e)
def load_fairness_distribution_by_train_session(file_path, feat_dict, item_feature_remap_df, train_session_df, standard = ['DP','EO']):
    if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                fairness_loaded = json.load(json_file)
            fairness_dict = {str(k): {int(cat): np.array(v) for cat, v in value.items()} for k, value in fairness_loaded.items()}
            print(f'Loaded Fairness distribution from {file_path}')

    else:
        cat_ids = feat_dict.keys()
        train_items = list(train_session_df['item_id'].unique())

        fairness_dict = {}
        for ss in standard:
            fairness_dict[ss] = {}
            for cat in cat_ids:
                fairness_dict[ss][cat] = np.zeros(feat_dict[cat])

        for cat in cat_ids:
            session_item_features = pd.merge(train_session_df, 
                                                item_feature_remap_df[item_feature_remap_df['feature_category_id']==cat], 
                                                on='item_id', 
                                                how='left')
            session_item_features['feature_category_id'].fillna(cat,inplace=True)
            session_item_features['remap_value_id'].fillna(feat_dict[cat]-1,inplace=True)
            eo_df = session_item_features['remap_value_id'].value_counts().sort_index()
            
            item_features = session_item_features[['item_id','feature_category_id','remap_value_id']].drop_duplicates()
            dp_df = item_features['remap_value_id'].value_counts().sort_index()

            for i in range(feat_dict[cat]):
                try:
                    fairness_dict['EO'][cat][i] = eo_df[i]
                except:
                    print('EO',cat,i,len(eo_df))
                    pass
                try:
                    fairness_dict['DP'][cat][i] = dp_df[i]
                except:
                    print('DP',cat,i,len(dp_df))
                    pass

        fairness_converted = {str(k): {int(cat): v.tolist() for cat, v in value.items()} for k, value in fairness_dict.items()}
        with open(file_path, 'w') as json_file:
            json.dump(fairness_converted, json_file)
        print(f'Saved Fairness distribution in {file_path}')

    return fairness_dict

# Load user preference distribution (vector p)
def load_preference_for_test_session(file_path, feat_dict, item_feature_remap_df, test_session_df):
    if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                preference_loaded = json.load(json_file)

            preference_dict = {int(k): {int(cat): np.array(v) for cat, v in value.items()} for k, value in preference_loaded.items()}

            print(f'Loaded Variety-seeking distribution from {file_path}')
    else:
        cat_ids = feat_dict.keys()
        
        session_set = test_session_df['session_id'].unique()

        preference_dict = {}
        for ss in session_set:
            preference_dict[int(ss)] = {}
            for cat in cat_ids:
                preference_dict[int(ss)][int(cat)] = np.zeros(feat_dict[cat])

        for cat in cat_ids:
            session_item_features = pd.merge(test_session_df, 
                                                item_feature_remap_df[item_feature_remap_df['feature_category_id']==cat], 
                                                on='item_id', 
                                                how='left')
            feature_distribution = session_item_features.groupby(['session_id', 'feature_category_id','remap_value_id'],dropna=False).size().reset_index(name='count')
            feature_distribution['feature_category_id'].fillna(cat,inplace=True)
            feature_distribution['remap_value_id'].fillna(feat_dict[cat]-1,inplace=True)
            for _, row in feature_distribution.iterrows():
                session_id = int(row['session_id'])
                cat_id = int(row['feature_category_id'])
                remap_value_id = int(row['remap_value_id'])
                count = row['count']
                preference_dict[session_id][cat_id][remap_value_id] = count 

        preference_converted = {int(k): {int(cat): v.tolist() for cat, v in value.items()} for k, value in preference_dict.items()}
        with open(file_path, 'w') as json_file:
            json.dump(preference_converted, json_file)
        print(f'Saved Variety-seeking distribution in {file_path}')

    return preference_dict

# Main data loader
def get_data(cat_ids, feature_value_df, item_feature_remap_df, train_session_df, test_session_df, data_folder='data/dressipy/rerank_input/'):
    session_set = list(test_session_df['session_id'].unique())
    item_set = list(item_feature_remap_df['item_id'].unique())

    print(f"NUM_SESSIONS: {len(session_set)}",f"NUM_ITEMS: {len(item_set)}")

    feat_df = feature_value_df[feature_value_df['feature_category_id'].isin(cat_ids)][['feature_category_id','value_num']]
    feat_df['value_num'] += 1
    feat_dict = feat_df.set_index('feature_category_id')['value_num'].to_dict()

    cat_name_prefix = '_'.join([str(feat) for feat in feat_dict.keys()])
    fairness_path = f'{data_folder}expected_fairness_dist_{cat_name_prefix}.json'
    variety_path = f'{data_folder}variety_seeking_dist_{cat_name_prefix}.json'

    fairness_dict = load_fairness_distribution_by_train_session(fairness_path, feat_dict, item_feature_remap_df, train_session_df)
    preference_dict = load_preference_for_test_session(variety_path, feat_dict, item_feature_remap_df, test_session_df)

    tau_values = get_tau(session_set,feat_dict,preference_dict)

    return session_set,item_set,feat_dict,fairness_dict,preference_dict,tau_values

# ------- Objective function helper -------
def get_tau(session_set,feat_dict,preference_dict):
    tau_values = {}
    for u in session_set:
        tau_values[u] = {}
        for feat in feat_dict.keys():
            tau = cal_entropy(percent_scale(preference_dict[u][feat]))
            if (tau-0) < 1e-6:
                tau = 1e-6
            tau_values[u][feat] = tau
    return tau_values

def get_opt_input_tau(feat_dict,fairness_dict,preference_dict,session_set,tau_values,lam=0.5):
    g = {}
    g_dp = {}
    g_eo = {}
    for feat in feat_dict.keys():
        flag = np.prod(fairness_dict['DP'][feat])
        if flag > 0:
            flag = 1
        e_dp = norm_scale(fairness_dict['DP'][feat])
        g_dp[feat] = {}
        e_eo = norm_scale(fairness_dict['EO'][feat])
        g_eo[feat] = {}
        for u in session_set:
            f = norm_scale(preference_dict[u][feat])
            tau = tau_values[u][feat]/np.log2(feat_dict[feat]+flag*(-1))
            g_dp[feat][u] = np.array(100 * lam*f+100 * (1-lam)*tau*e_dp)
            g_eo[feat][u] = np.array(100 * lam*f+100 * (1-lam)*tau*e_eo)
    g['DP'] = g_dp
    g['EO'] = g_eo
    return g

def get_opt_input_lam(feat_dict,fairness_dict,preference_dict,session_set,lam=0.5):
    g = {}
    g_dp = {}
    g_eo = {}
    for feat in feat_dict.keys():
        e_dp = norm_scale(fairness_dict['DP'][feat])
        g_dp[feat] = {}
        e_eo = norm_scale(fairness_dict['EO'][feat])
        g_eo[feat] = {}
        for u in session_set:
            f = norm_scale(preference_dict[u][feat])
            cal_lam = 1-lam
            g_dp[feat][u] = np.array((1-cal_lam)*f+cal_lam*e_dp)
            g_eo[feat][u] = np.array((1-cal_lam)*f+cal_lam*e_eo)
    g['DP'] = g_dp
    g['EO'] = g_eo
    return g

def get_opt_input_tau_mean(feat_dict,fairness_dict,preference_dict,session_set,tau_values,lam=0.5):
    g = {}
    g_dp = {}
    g_eo = {}
    for feat in feat_dict.keys():
        flag = np.prod(fairness_dict['DP'][feat])
        if flag > 0:
            flag = 1
        e_dp = norm_scale(fairness_dict['DP'][feat])
        g_dp[feat] = {}
        e_eo = norm_scale(fairness_dict['EO'][feat])
        g_eo[feat] = {}
        tau_list = []
        for u in session_set:
            f = norm_scale(preference_dict[u][feat])
            tau = tau_values[u][feat]/np.log2(feat_dict[feat]+flag*(-1))
            tau_list.append(tau)

        tau_mean = np.mean(tau_list)
        for u in session_set:
            f = norm_scale(preference_dict[u][feat])
            g_dp[feat][u] = np.array(100 * lam*f+100 * (1-lam)*tau_mean*e_dp)
            g_eo[feat][u] = np.array(100 * lam*f+100 * (1-lam)*tau_mean*e_eo)
    g['DP'] = g_dp
    g['EO'] = g_eo
    return g

def get_opt_input_random(feat_dict,fairness_dict,preference_dict,session_set,tau_values,lam=0.5):
    tau_values_transposed = {}
    for feat in feat_dict.keys():
        tau_values_transposed[feat] = []
        for u in session_set:
            tau_values_transposed[feat].append(tau_values[u][feat])
    g = {}
    g_dp = {}
    g_eo = {}
    for feat in feat_dict.keys():
        flag = np.prod(fairness_dict['DP'][feat])
        if flag > 0:
            flag = 1
        e_dp = norm_scale(fairness_dict['DP'][feat])
        g_dp[feat] = {}
        e_eo = norm_scale(fairness_dict['EO'][feat])
        g_eo[feat] = {}
        for u in session_set:
            f = norm_scale(preference_dict[u][feat])
            random.seed = int(u)
            tau = random.choice(tau_values_transposed[feat])/np.log2(feat_dict[feat]+flag*(-1))
            g_dp[feat][u] = np.array(100 * lam*f+100 * (1-lam)*tau*e_dp)
            g_eo[feat][u] = np.array(100 * lam*f+100 * (1-lam)*tau*e_eo)
    g['DP'] = g_dp
    g['EO'] = g_eo
    return g

# Load matrix used in optimization solution
def save_A_to_file(A, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(A, f)
    print(f"A has been saved to {file_path}.")

def load_A_from_file(file_path):
    with open(file_path, 'rb') as f:
        A = pickle.load(f)
    print(f"A has been loaded from {file_path}.")
    return A

def get_matrix_A(baserec_df, session_set):
    baserec_df = baserec_df.set_index('session_id')
    A = (baserec_df.loc[list(session_set), 'top_rec_score']
         .apply(lambda x: scaler.fit_transform(np.array([float(i) for i in x.split('|')]).reshape(-1, 1)))
         .to_dict())
    if len(A) != len(session_set):
        raise ValueError('Length Error')
    # print("get A")
    return A

def get_matrix_A_by_cache(baserec_df, session_set, model, candsK, data_folder):
    file_path = data_folder+f'A_{model}_{candsK}.pkl'
    if os.path.exists(file_path):
        return load_A_from_file(file_path)
    else:
        print(f"File {file_path} not found. Generating A...")
        A = get_matrix_A(baserec_df, session_set)
        save_A_to_file(A, file_path)
        # print("get A")
        return A
    
def get_C_by_one(u,feat_dict,baserec_df,item_feature_remap_df,candsK):
    cat_ids = feat_dict.keys()
    C_dict = {cat: {} for cat in cat_ids}
    rec_list = baserec_df.loc[u,'top_rec']
    rec_df = pd.DataFrame({'item_id': [int(i) for i in rec_list.split('|')]})

    for cat in cat_ids:
        C_dict[cat] = lil_matrix((feat_dict[cat], candsK), dtype=int)

        rec_item_features = pd.merge(
            rec_df, 
            item_feature_remap_df[item_feature_remap_df['feature_category_id'] == cat], 
            on='item_id', how='left'
        )

        rec_item_features['feature_category_id'].fillna(cat, inplace=True)
        rec_item_features['remap_value_id'].fillna(feat_dict[cat] - 1, inplace=True)

        for idx, value in enumerate(rec_item_features['remap_value_id']):
            try:
                C_dict[cat][int(value), idx] = 1
            except:
                raise ValueError(f'{cat},{value},{idx}')
        del rec_item_features

    del rec_list, rec_df
    return C_dict
