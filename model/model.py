import numpy as np
import coptpy as cp
from coptpy import COPT,quicksum,EnvrConfig
from tqdm import tqdm
MAX_ITER = 20
# SAVE_MODEL = 0
envconfig = EnvrConfig()
envconfig.set("nobanner", "1")
np.random.seed(2025)

def iterative_algorithm(epsilon, C, A, g, u, feat_dict,k=10,q=0.9,m=500):
    """
    iterative_algorithm

    Parameters:
    - epsilon: Convergence threshold.
    - C: Feature matrices of candidate items for each attribute.
    - A: Relevance score matrices for each candidate item.
    - g: Weighted sum of user's diversity preference and product's expected exposure distribution across attributes.
    - u: Target user ID.
    - feat_dict: Attribute id and corresponding number of value.
    - k: Length of the re-ranked recommendation list.
    - q: Accuracy threshold.
    - m: Candidate item number.
    - max_iter: Maximum number of iterations, 20 here.

    Returns:
    - ys: Final solution after convergence.
    """
    param_record = {}
    param_record['delta']=[]
    l = 1  # iteration calculator
    H = len(feat_dict)  # attribute number

    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            # randomize γl,βl
            if attempts == 0:
                y_s = np.concatenate([np.ones(k), np.zeros(m - k)])
                gamma_l = np.zeros(H)
                beta_l = np.zeros(H)
                for h in range(H):
                    feat = list(feat_dict.keys())[h]
                    C_h = C[feat]
                    g_h = g[feat][u]
                    Chy_l = C_h@y_s
                    beta_l[h] = np.dot(g_h.T, Chy_l) / np.linalg.norm(Chy_l)
                    gamma_l[h] = 1 / np.linalg.norm(Chy_l)
                    if gamma_l[h] ==0 :
                        gamma_l[h]+=1e-06
                    if beta_l[h]==0:
                        beta_l[h]+=1e-06
            else:
                gamma_l = np.random.rand(H)
                beta_l = np.random.rand(H)

            y_s,model_s = solve_socp(feat_dict, u, C, g, A, gamma_l, beta_l, k,q,m)
            if model_s.status == COPT.OPTIMAL:
                #tqdm.write(gamma_l,beta_l)
                break

            attempts += 1

        except cp.CoptError as e:
            attempts += 1
            tqdm.write(f"CoptError detected: {e}")
            tqdm.write("Parameters reinitialized. Retrying.")

    if attempts == max_attempts:
        tqdm.write("Max attempts reached.")
        raise KeyboardInterrupt

    while l < MAX_ITER:
        # calculating error vector δl
        delta_l = np.ones(2 * H)
        if model_s.status == COPT.OPTIMAL:
            for h in range(H):
                feat = list(feat_dict.keys())[h]
                C_h = C[feat]
                g_h = g[feat][u]
                Chy_l = C_h@y_s
                delta_l[h] = beta_l[h] * np.linalg.norm(Chy_l) - g_h.T@Chy_l
                delta_l[H + h] = gamma_l[h] * np.linalg.norm(Chy_l) - 1
                # Parameter updating
                beta_l[h] = np.dot(g_h.T, Chy_l) / np.linalg.norm(Chy_l)
                gamma_l[h] = 1 / np.linalg.norm(Chy_l)
                if gamma_l[h] ==0 :
                        gamma_l[h]+=1e-06
                if beta_l[h]==0:
                    beta_l[h]+=1e-06
        else:
            gamma_l = np.random.rand(H)
            beta_l = np.random.rand(H)
            delta_l = np.ones(2 * H)
        # Convergence checking
        param_record['delta'].append(np.linalg.norm(delta_l))
        if np.linalg.norm(delta_l) < epsilon:
            break


        l += 1
        y_s,model_s = solve_socp(feat_dict, u, C, g, A, gamma_l, beta_l, k,q,m)

    if np.linalg.norm(delta_l) >= epsilon:
        y_s = np.concatenate([np.ones(k), np.zeros(m - k)])
    for p in param_record.keys():
        param_record[p] = '|'.join(map(str,(param_record[p])))
    return y_s,param_record

def solve_socp(feat_dict, u, C, g, A, gamma, beta, k,q, m):
    """
    Solves the Second-Order Cone Programming (SOCP) problem based on the given parameters.

    Parameters:
    - feat_dict: Attribute id and corresponding number of value.
    - u: Target user ID.
    - C: Feature matrices of candidate items for each attribute.
    - g: Weighted sum of user's diversity preference and product's expected exposure distribution across attributes.
    - A: Relevance score matrices for each candidate item.
    - gamma, beta: Fixed Lagrange multipliers.
    - k, q, m.

    Returns:
    - y_opt: Optimal solution vector.
    """
    H = len(feat_dict)
    # Model initialization
    env = cp.Envr(envconfig)
    model = env.createModel("SOCP")
    model.setParam(COPT.Param.Logging,0)
    model.setParam(COPT.Param.LogToConsole,0)

    # Decision variables
    y = model.addMVar(m,lb=0, ub=1, nameprefix="y")
    
    # Define auxiliary variables S and V
    S = model.addMVar(H,lb=0, nameprefix="S")
    V = model.addMVar(H,lb=0, nameprefix="V")

    v = 1/(gamma*beta)
    Sv = v*S

    # Add constraints：
    # 1. Recommendation accuracy
    a = A[u] # --baseline score
    accu_eq = np.sum(a[:k])
    model.addConstr(a.T@y >= accu_eq*q, name="accu")
    # 2. Re-rank list length
    model.addConstr(np.ones(m)@y == k, name="sum_y")
    # 3. Second-Order Cone Constraint
    model.addConstrs((V[i]-Sv[i] == 0 for i in range(H)), f"cal_v")
    for h in range(H):
        feat = list(feat_dict.keys())[h]
        fl = feat_dict[feat]
        C_h = C[feat].toarray()
        tmp_h = C_h@y
        V_h = V[h]
        r_h = model.addMVar(fl,lb=0, nameprefix=f"r_{feat}")
        model.addConstrs((tmp_h[i]-r_h[i] == 0 for i in range(fl)), f"cal_r_{feat}")
        model.addCone(V_h.tolist()+r_h.tolist(),COPT.CONE_QUAD)

    # Set objective function
    obj = cp.MLinExpr.zeros((1,1))
    obj = obj+quicksum(beta)
    for h in range(H):
        feat = list(feat_dict.keys())[h]
        C_h = C[feat].toarray()
        g_h = g[feat][u]
        tmp_h = C_h@y
        tmp_expr = g_h.T@tmp_h
        gamma_h = gamma[h]
        obj += gamma_h*tmp_expr
        obj -= S[h]
    model.setObjective(obj, COPT.MAXIMIZE)

    # Model sove
    # tqdm.write("---model sovling---")
    model.solve()


    # Get optimal solution
    y_opt = []
    try:
        for i in range(m):
            y_i = model.getVarByName(f"y({i})")
            y_opt.append(y_i.getInfo("Value"))
    except:
        y_opt = None
    
    return np.array(y_opt),model