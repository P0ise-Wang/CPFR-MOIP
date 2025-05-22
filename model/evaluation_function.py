import numpy as np
import pandas as pd
from tqdm import tqdm
METRICS = ['R','NDCG','UFMS','PFMS']

# ---- math function helper ----
def norm_scale(counts_vector):
    norm = np.linalg.norm(np.array(counts_vector))
    normalized_vector = counts_vector / norm if norm != 0 else counts_vector
    return normalized_vector
def percent_scale(counts_vector):
    norm = sum(np.array(counts_vector))
    normalized_vector = counts_vector / norm
    return normalized_vector
def cal_cosine_similarity(p1,p2):
    p1 = norm_scale(p1)
    p2 = norm_scale(p2)
    return np.array(p1).T@np.array(p2)
def cal_cosine_similarity_2(p1,p2):
    p1 = percent_scale(p1)
    p2 = percent_scale(p2)
    return np.array(p1).T@np.array(p2)

# ---- data loading ----
# Generate recommendation list representation (vector r)
def generate_overall_dist(rerank_df, feat_dict, item_feature_remap_df, k):
    cat_ids = feat_dict.keys()
    R_dict = {}
    for _,row in rerank_df.iterrows():
        ss = row['session_id']
        R_dict[ss] = {}
        rec_list = [int(i) for i in row['top_rec'].split('|')][:k]
        rec_df = pd.DataFrame({'item_id':rec_list})
        for cat in cat_ids:
            R_dict[ss][cat] = np.zeros(feat_dict[cat])
            rec_item_features = pd.merge(rec_df,item_feature_remap_df[item_feature_remap_df['feature_category_id']==cat], 
                                                on='item_id', 
                                                how='left')
            rec_distribution = rec_item_features['remap_value_id'].value_counts().sort_index()
            for i in range(feat_dict[cat]):
                try:
                    R_dict[ss][cat][i] = rec_distribution[i]
                except:
                    pass    
    return R_dict

#-----------------------------------------------------
# calculate DCG
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
# calculate NDCG
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max
# Recommendation accuracy evaluation
def evaluate_recommendations(recommendations, test_data, k):
    accu_dict = {}
    accu_dict[f'P@{k}'] = {}
    accu_dict[f'R@{k}'] = {}
    accu_dict[f'HR@{k}'] = {}
    accu_dict[f'NDCG@{k}'] = {}
    # precision, recall, hr_num, ndcg = [],[],[],[]
    tests = {}
    test_users = test_data['session_id'].unique()
    for user in test_users:
        tests[user] = set(test_data[test_data['session_id'] == user]['item_id'].values)
    for user, true_items in tests.items():
        pred_items = recommendations[user][:k]
        num_hits = len(set(pred_items) & set(true_items))
        accu_dict[f'P@{k}'][user]=num_hits / k
        accu_dict[f'R@{k}'][user]=num_hits / len(true_items)
        accu_dict[f'HR@{k}'][user]= 1 if num_hits > 0 else 0
        accu_dict[f'NDCG@{k}'][user]=ndcg_at_k([1 if item in true_items else 0 for item in pred_items], k)
    # precision /= len(test_users)
    # recall /= len(test_users)
    # hr /= len(test_users)
    # ndcg /= len(test_users)
    return pd.DataFrame.from_dict(accu_dict,orient='index').T

def model_eval_acc(df,test_df,k):
    result = pd.Series(df.top_rec.values, index=df.session_id).to_dict()
    result_dict = {key: [int(x) for x in value.split('|') if x] for key, value in result.items()}
    eval_df = evaluate_recommendations(result_dict,test_df,k)
    eval_df = pd.DataFrame(eval_df)
    eval_df['f1'] = 2*eval_df[f'P@{k}']*eval_df[f'R@{k}']/(eval_df[f'P@{k}']+eval_df[f'R@{k}'])
    eval_df['f1'].fillna(0)
    eval_df_f = eval_df[[f'R@{k}',f'NDCG@{k}']].copy()
    return eval_df,eval_df_f

#--------------------------------------
# Recommendation fairness evaluation
def cal_ams(feat_dict,R,e=None,d=None):
    ams_dict={}
    for u in R.keys():
        d_u = d[u]
        r_u = R[u]
        ams_dict[u] = {}
        ams_dict[u]['UFMS'] = 0
        ams_dict[u]['PFMS'] = 0
        for feat in feat_dict.keys():
            ams_dict[u]['UFMS'] += cal_cosine_similarity(r_u[feat],d_u[feat])
            ams_dict[u]['PFMS'] += cal_cosine_similarity(r_u[feat],e[feat])
        ams_dict[u]['UFMS'] /= len(feat_dict.keys())
        ams_dict[u]['PFMS'] /= len(feat_dict.keys())
    return ams_dict

def cal_ams_feature(feat_dict,R,e,d):
    ams_dict={}
    for u in R.keys():
        d_u = d[u]
        r_u = R[u]
        ams_dict[u] = {}
        for feat in feat_dict.keys():
            ams_dict[u][f'UFMS_{feat}'] = 0
            ams_dict[u][f'PFMS_{feat}'] = 0
        for feat in feat_dict.keys():
            ams_dict[u][f'UFMS_{feat}'] = cal_cosine_similarity(r_u[feat],d_u[feat])
            ams_dict[u][f'PFMS_{feat}'] = cal_cosine_similarity(r_u[feat],e[feat])
    return ams_dict

# Metrics calculation main function
def model_eval(df,feat_dict,item_feature_remap_df,test_df,e,d,k=10,features_detail = False):
    result_df = df[df['session_id'].isin(test_df['session_id'])]
    _,ori_df = model_eval_acc(df,test_df,k)
    R = generate_overall_dist(result_df,feat_dict,item_feature_remap_df,k)
    if features_detail:
        ams_dict = cal_ams_feature(feat_dict,R,e,d)
        ams_dict2 = cal_ams(feat_dict, R, e, d)
        ams_df = pd.DataFrame.from_dict(ams_dict, orient='index')
        ams_df2 = pd.DataFrame.from_dict(ams_dict2, orient='index')
        eval_df = pd.concat([ori_df, ams_df], axis=1)
        eval_df2 = pd.concat([ori_df, ams_df2], axis=1)
        eval_result_df = pd.DataFrame(eval_df2.mean())
        return eval_result_df.round(3), eval_df
    else:
        ams_dict = cal_ams(feat_dict,R,e,d)
        ams_df = pd.DataFrame.from_dict(ams_dict,orient='index')
        eval_df = pd.concat([ori_df,ams_df],axis=1)
        eval_result_df = pd.DataFrame(eval_df.mean())
        if (d is not None) and (e is not None):
            pass
        else:
            print('Baseline Evaluation.')
        return eval_result_df,eval_df

# Metrics difference calculation (Rerank v.s. initial)
def metric_cal_helper(base_result,rerank_result,topk,metric_list):
    diff_dict = {}
    for m in metric_list:
        if m in ['NDCG','R']:
            base = base_result.loc[f'{m}@{topk}'].values[0]
            rerank = rerank_result.loc[f'{m}@{topk}'].values[0]
        else:
            base = base_result.loc[f'{m}'].values[0]
            rerank = rerank_result.loc[f'{m}'].values[0]
        diff = (rerank-base)/base
        diff_dict[m] = round(diff,4)
    return diff_dict

# Evaluation main function
def evaluation_main(baserec_df,rerank_df,feat_dict,item_feature_remap_df,test_df,fairness_dict,preference_dict,
                    result_folder,args):
    tqdm.write(f'------Start Eval: Baseline-------------')
    base_result,base_eval_df = model_eval(baserec_df,feat_dict,item_feature_remap_df,test_df,fairness_dict[args.standard],preference_dict,args.k)
    base_result.columns = [f'Base']
    tqdm.write(f'------Finish Eval: Baseline-------------')
    result_df = base_result.copy()

    recall_diff_list = [0]
    ndcg_diff_list = [0]
    uf_diff_list = [0]
    pf_diff_list = [0]

    ndcg_vs_uf = [0]
    ndcg_vs_pf = [0]
    tqdm.write(f'------Start Eval: Reranking-------------')
    rerank_result,rerank_eval_df = model_eval(rerank_df,feat_dict,item_feature_remap_df,test_df,fairness_dict[args.standard],preference_dict,args.k)
    rerank_result.columns = [f'Rerank']
    result_df = pd.concat([result_df,rerank_result],axis=1)

    diff_dict = metric_cal_helper(base_result,rerank_result,args.k,METRICS)
    
    recall_diff_list.append(diff_dict['R'])
    ndcg_diff_list.append(diff_dict['NDCG'])
    uf_diff_list.append(diff_dict['UFMS'])
    pf_diff_list.append(diff_dict['PFMS'])
    
    ndcg_vs_uf.append(diff_dict['UFMS']/diff_dict['NDCG'])
    ndcg_vs_pf.append(diff_dict['PFMS']/diff_dict['NDCG'])

    result_df.loc[f'ΔR@{args.k}'] = recall_diff_list
    result_df.loc[f'ΔNDCG@{args.k}'] = ndcg_diff_list
    result_df.loc['ΔUFMS'] = uf_diff_list
    result_df.loc['ΔPFMS'] = pf_diff_list

    result_df.loc[f'ΔUFMS/ΔNDCG@{args.k}'] = ndcg_vs_uf
    result_df.loc[f'ΔPFMS/ΔNDCG@{args.k}'] = ndcg_vs_pf

    tqdm.write(f'------Finish Eval: {args.model}_{args.standard}_k{args.k}_l{args.lam}_q{args.q}-------------')
    tqdm.write(result_df.to_string())
        
    result_df.to_csv(f'{result_folder}{args.dataset}@{args.cat_ids}_{args.model}@{args.baserecK}_{args.tau}_{args.standard}.csv',index=False) 