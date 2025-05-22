import timeit
import pandas as pd
import numpy as np
from data import get_data,get_matrix_A_by_cache,get_C_by_one
from data import get_opt_input_tau,get_opt_input_tau_mean,get_opt_input_random,get_opt_input_lam
from model import iterative_algorithm
from evaluation_function import evaluation_main
import argparse
import concurrent.futures
import os
import datetime
from tqdm import tqdm
import json
today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
epsilon = 1e-03

# ---- Initial recommendation loader ----
# Load the recommendation results from a TXT file
def read_recommendation_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            session_id = int(parts[0])  # 第一部分是 session_id
            items = '|'.join(parts[1:])  # 其余部分是推荐列表或分数，用 | 分隔
            data.append((session_id, items))
    return pd.DataFrame(data, columns=['session_id', 'top_rec'])
# Load the recommendation scores from a TXT file
def read_score_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            session_id = int(parts[0])  # 第一部分是 session_id
            scores = '|'.join(parts[1:])  # 其余部分是推荐分数，用 | 分隔
            data.append((session_id, scores))
    return pd.DataFrame(data, columns=['session_id', 'top_rec_score'])

# ---- Math function helper ----
def norm_scale(counts_vector):
    norm = np.linalg.norm(np.array(counts_vector))
    normalized_vector = counts_vector / norm if norm != 0 else counts_vector
    return normalized_vector
def cal_obj(g, C, y, u):
    obj = 0
    for feat in g.keys():
        r = norm_scale(C[feat] @ y)
        g_f = np.array(g[feat][u])
        obj += np.dot(g_f,r)
    return obj / 2   

# Parallel re-ranking function for individual users
# (based on multi-objective optimization)
def process_user_subset(user_subset, feat_dict, 
                        baserec_df, item_feature_remap_df, 
                        epsilon, A_input, g_input, args,
                        pbar, tbar):
    re_rank_result_subset = {}
    param_record_subset = {}
    re_rank_dict_subset = {}
    time_dict_subset = {}
    obj_record_subset = {'baseline': {}, 'rerank': {}}
    
    for u in user_subset:
        # 1. get C_input (i.e., F in the article)
        C_input = get_C_by_one(u, feat_dict, baserec_df, item_feature_remap_df, args.baserecK)
        
        # 2. Run the iterative algorithm to obtain optimization solution
        start = timeit.default_timer()
        u_opt, u_param = iterative_algorithm(epsilon, C_input, A_input, g_input, u, feat_dict, args.k, args.q, args.baserecK)
        stop = timeit.default_timer()
        time_dict_subset[u] = stop - start
        re_rank_result_subset[u] = u_opt
        param_record_subset[u] = u_param
        
        # 3. Re-rank indices
        opt_index = np.argsort(-np.array(re_rank_result_subset[u]))[:args.k].tolist()
        reranked_opt_index = sorted(opt_index, key=lambda idx: A_input[u][idx], reverse=True)
        
        # 4. Compute the objective function
        y_b = np.zeros(args.baserecK)
        y_b[range(args.k)] = 1
        y_r = np.zeros(args.baserecK)
        y_r[reranked_opt_index] = 1
        obj_record_subset['baseline'][u] = cal_obj(g_input, C_input, y_b, u)
        obj_record_subset['rerank'][u] = cal_obj(g_input, C_input, y_r, u)
        
        # 5. Retrieve the original recommendation list and perform re-ranking
        user_rec = baserec_df[baserec_df['session_id'] == u]['top_rec'].iloc[0]
        ori_rank_list = [int(i) for i in user_rec.split('|')]
        
        # 6. Check the objective function result
        if obj_record_subset['baseline'][u] <= obj_record_subset['rerank'][u]:
            re_rank_list = [ori_rank_list[i] for i in reranked_opt_index]
            re_rank_dict_subset[u] = '|'.join(str(i) for i in re_rank_list)
        else:
            re_rank_list = ori_rank_list[:args.k]
            re_rank_dict_subset[u] = '|'.join(str(i) for i in re_rank_list)
        
        pbar.update(1)
        tbar.update(1)
    pbar.close()
    return re_rank_result_subset, param_record_subset, re_rank_dict_subset, obj_record_subset, time_dict_subset

def main(args):
    print('-------start solving--------')
    A_input = A.copy()
    g_input = g[args.standard].copy()

    re_rank_result = {}
    param_record = {}
    re_rank_dict = {}
    time_record = {}
    obj_record = {'baseline': {}, 'rerank': {}}

    # Split session_set into user subsets for each thread, with 2000 users per subset
    user_chunks = [session_set[i:i + 2000] for i in range(0, len(session_set), 2000)]
    total_pbar = tqdm(total=len(session_set), desc="Total", position=0, ncols=60, leave=False)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        start = timeit.default_timer()
        for i, user_subset in enumerate(user_chunks):
            thread_pbar = tqdm(total=len(user_subset), desc=f"Thread {i+1}", position=i+1, ncols=60, leave=False)
            futures.append(executor.submit(process_user_subset, user_subset, feat_dict, 
                                           baserec_df, item_feature_remap_df, 
                                           epsilon, A_input, g_input, args, 
                                           thread_pbar, total_pbar))
        
        # Process results as they are completed
        for future in concurrent.futures.as_completed(futures):
            re_rank_result_subset, param_record_subset, re_rank_dict_subset, obj_record_subset, time_dict_subset = future.result()
            re_rank_result.update(re_rank_result_subset)
            param_record.update(param_record_subset)
            time_record.update(time_dict_subset)
            re_rank_dict.update(re_rank_dict_subset)
            obj_record['baseline'].update(obj_record_subset['baseline'])
            obj_record['rerank'].update(obj_record_subset['rerank'])
        stop = timeit.default_timer()
        tqdm.write('Process Time: %.2f secs' % (stop - start))

    # Report metrics over objective function
    obj_record_df = pd.DataFrame(obj_record)
    obj_rd = obj_record_df.describe()
    ir = (obj_rd.loc['mean','rerank'] - obj_rd.loc['mean','baseline']) / obj_rd.loc['mean','baseline']
    tqdm.write(f'Average Increase Ration of OBJ: {ir * 100:.2f}%')
    
    # Report metrics over recommendation accuracy and multi-sided fairness
    re_rank_df = pd.DataFrame(columns=['session_id','top_rec'])
    re_rank_df['session_id'] = re_rank_dict.keys()
    re_rank_df['top_rec'] = re_rank_dict.values()
    
    # re_rank_df.to_csv(result_folder+f'{args.model}_{args.standard}_top{args.k}_q{args.q}_lam{args.l}_{today}.csv',index=False)
    evaluation_main(baserec_df,re_rank_df,feat_dict,item_feature_remap_df,test_df,fairness_dict,preference_dict,
                    result_folder,args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--model', type=str, default='MGS',choices=['FPMC','SASRec','MGS'],help='Initial recommendation model')
    parser.add_argument('--standard', type=str, default='DP',choices=['DP','EO'],help='Standard name')
    parser.add_argument('--k', type=int, default=10, help='Value of k')
    parser.add_argument('--q', type=float, default=0.95, help='Value of q')
    parser.add_argument('--lam', type=float, default='0.25', help='Value of lam')
    parser.add_argument('--tau', type=str, default='tau',choices=['tau','lam','tau_rand','tau_mean'], help='Whether to add tau as weight on PFairness')
    parser.add_argument('--baserecK', type=int, default=500, help='Length of initial recommendation list')
    parser.add_argument('--dataset', type=str, default='dressipy',choices=['dressipy','tmall'], help='Dataset')
    parser.add_argument('--cat_ids', type=str, default='-1',choices=['-1','68,-1'], help='Attribute ids (-1:vpopularity)')

    args = parser.parse_args()

    # Load best parameters
    with open('config.json', 'r', encoding='utf-8') as file:
        args_dict_existed = json.load(file)
        args_dict_existed = args_dict_existed[f'{args.dataset}_{args.standard}']
    for key, value in args_dict_existed.items():
        if hasattr(args, key):
            setattr(args, key, value)

    cat_ids_input = [int(i) for i in args.cat_ids.split(',')]

    print(f"Dataset: {args.dataset}",f"Cats: {cat_ids_input}",f"Model: {args.model}")
    print("---------------")
    result_folder = f'result/'
    data_folder = f'data/{args.dataset}/'
    rerank_data_folder = f'data/{args.dataset}/rerank_input/'
    baserec_result_folder = f'data/{args.dataset}/baserec_result/'
    if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    train_session_df = pd.read_csv(f"{data_folder}train_session.csv")
    train_purchase_df = pd.read_csv(f"{data_folder}train_purchase.csv")
    train_df = pd.concat([train_session_df,train_purchase_df],axis=0).sort_values(by=['session_id','date'])
    test_session_df = pd.read_csv(f"{data_folder}test_session.csv")
    test_df = pd.read_csv(data_folder+'test_purchase.csv')
    test_df.sort_values('session_id',inplace=True)
    item_feature_remap_df = pd.read_csv(f"{data_folder}item_feature_remap_with_pop.csv")
    feature_value_df = pd.read_csv(f"{data_folder}feature_value_with_pop.csv")

    session_set,item_set,feat_dict,fairness_dict,preference_dict,tau_values = get_data(cat_ids_input, feature_value_df, item_feature_remap_df, train_df, test_session_df, rerank_data_folder)
    print("Basic data loaded..")

    print(f"Model: {args.model}")
    baserec_df = read_recommendation_file(baserec_result_folder + args.model + f'_baserec_{args.baserecK}.txt')
    baserec_df_score = read_score_file(baserec_result_folder + args.model + f'_baserec_score_{args.baserecK}.txt')
    baserec_df = baserec_df[baserec_df['session_id'].isin(session_set)].reset_index(drop=True)
    baserec_df = pd.merge(baserec_df, baserec_df_score, how='inner', on='session_id')
    baserec_df.index = baserec_df['session_id']
    # print(baserec_df)

    A = get_matrix_A_by_cache(baserec_df, session_set, args.model, args.baserecK, rerank_data_folder)
    print("---------------")

    print(f"Standard: {args.standard}",f"k: {args.k}",f"q: {args.q}",f"lam: {args.lam}",f"tau: {args.tau}")
    if args.tau.lower() == 'tau':
        g = get_opt_input_tau(feat_dict,fairness_dict,preference_dict,session_set,tau_values,args.lam)
    elif args.tau.lower() == 'lam':
        g = get_opt_input_lam(feat_dict,fairness_dict,preference_dict,session_set,args.lam)
    elif args.tau.lower() == 'tau_rand':
        g = get_opt_input_random(feat_dict,fairness_dict,preference_dict,session_set,tau_values,args.lam)
    elif args.tau.lower() == 'tau_mean':
        g = get_opt_input_tau_mean(feat_dict,fairness_dict,preference_dict,session_set,tau_values,args.lam)
    else:
        raise ValueError(f"Unknown tau option: {args.tau}")

    main(args)
        