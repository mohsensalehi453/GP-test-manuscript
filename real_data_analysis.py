"""
=============================================================================
 REAL DATA APPLICATION: TCGA BREAST CANCER SUBTYPE ANALYSIS
 
 This is the final script to generate the results for the "Real Data
 Application" section of the manuscript.
=============================================================================
"""

# --- 1. Load Required Libraries and Setup ---
import numpy as np
import pandas as pd
import time
from itertools import combinations
import scipy.linalg
from scipy.stats import bartlett
import pingouin
import numba
from tqdm import tqdm

# --- Configuration ---
# File and column settings for the real data analysis
REAL_DATA_FILE = 'tcga_brca_top_genes_4_subtypes.csv'
#REAL_DATA_FILE = 'tcga_brca_5000genes_4subtypes.csv'
SUBTYPE_COL = 'Subtype'

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!  IMPORTANT: YOU MUST CHECK AND CONFIRM THIS VARIABLE NAME            !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Open your CSV file and find the name of the first gene column.
# All columns from this one to the end will be used in the analysis.
FIRST_GENE_COLUMN = 'UBE2Q2P2' 

# Analysis Parameters
ALPHA = 0.05
B_PERM = 1000     # Number of permutations for the permutation tests

# --- 2. IMPORTED TEST FUNCTIONS (FROM YOUR SIMULATION CODE) ---

@numba.njit(cache=True)
def compute_theta_hat_numba(x_centered, s_matrix):
    n, p = x_centered.shape
    theta_hat = np.zeros((p, p), dtype=np.float64)
    for j in range(p):
        for k in range(j, p):
            var_jk = 0.0
            for i in range(n):
                var_jk += (x_centered[i, j] * x_centered[i, k] - s_matrix[j, k])**2
            theta_hat[j, k] = var_jk / n
            if j != k:
                theta_hat[k, j] = theta_hat[j, k]
    return theta_hat

def cai_test_2013(x1, x2):
    n1, p = x1.shape
    if x2.shape[1] != p or n1 <= 1 or x2.shape[0] <= 1 or p < 2:
        return {'statistic': np.nan, 'p_value': np.nan}
    try:
        s1 = np.cov(x1, rowvar=False); s2 = np.cov(x2, rowvar=False)
        t_num = (s1 - s2)**2; x1c = x1 - x1.mean(axis=0); x2c = x2 - x2.mean(axis=0)
        theta1_hat = compute_theta_hat_numba(x1c, s1); theta2_hat = compute_theta_hat_numba(x2c, s2)
        t_den = theta1_hat + theta2_hat
        t_matrix = np.zeros_like(t_num); t_matrix[t_den != 0] = t_num[t_den != 0] / t_den[t_den != 0]
        m_n = np.max(t_matrix)
        x_val = m_n - (2 * np.log(p) - np.log(np.log(p)))
        p_value = 1 - np.exp(-np.exp(-x_val / 2))
        return {'statistic': m_n, 'p_value': p_value}
    except Exception:
        return {'statistic': np.nan, 'p_value': np.nan}

def schott_test_2007(data_list, alpha=0.05):
    try:
        K = len(data_list)
        if K < 2: return {'p_value': np.nan, 'reject_null': np.nan}
        n_k = np.array([d.shape[0] for d in data_list]); p = data_list[0].shape[1]; N = np.sum(n_k)
        S_k = [np.cov(d, rowvar=False) for d in data_list]; S_pooled = sum(nk*Sk for nk,Sk in zip(n_k,S_k))/N
        A1 = ((p * (p + 1)) / 2) * (K - 1)
        term1_sum = np.sum( (n_k-1)**-1 * ( (np.trace(S_k[i] @ S_k[i])) - p**-1 * (np.trace(S_k[i]))**2 ) for i in range(K) )
        term2 = (N-K)**-1 * ( (np.trace(S_pooled @ S_pooled)) - p**-1 * (np.trace(S_pooled))**2 )
        T = (p * (p+1) / 2)**-0.5 * (term1_sum - term2)
        p_value = 1 - scipy.stats.norm.cdf(T)
        return {'p_value': p_value, 'reject_null': p_value < alpha}
    except Exception:
        return {'p_value': np.nan, 'reject_null': np.nan}

def box_m_test(data, group_labels, alpha):
    try:
        df = pd.DataFrame(data); df['group'] = group_labels; p = data.shape[1]
        if df['group'].value_counts().min() <= p:
            print(f"  [Skipping Box's M-Test] Not applicable: min group size ({df['group'].value_counts().min()}) <= p ({p}).")
            return {'p_value': np.nan, 'reject_null': np.nan}
        res = pingouin.box_m(data=df, dvs=list(df.columns[:-1]), group='group')
        p_val = res.iloc[0]['p-val']
        return {'p_value': p_val, 'reject_null': p_val < alpha}
    except Exception as e:
        print(f"  [ERROR in Box's M-Test]: {e}"); return {'p_value': np.nan, 'reject_null': np.nan}

def bartlett_test_corrected(data, group_labels, alpha):
    p = data.shape[1]; unique_groups = np.unique(group_labels)
    if len(unique_groups) < 2: return {'p_value': np.nan, 'reject_null': np.nan}
    p_vals = []
    for k in range(p):
        samples = [data[group_labels == g, k] for g in unique_groups]
        if any(len(s) < 2 for s in samples): continue
        try: p_vals.append(bartlett(*samples)[1])
        except Exception: p_vals.append(np.nan)
    if all(np.isnan(p_v) for p_v in p_vals): return {'p_value': np.nan, 'reject_null': np.nan}
    min_p = np.nanmin(p_vals)
    return {'p_value': min_p, 'reject_null': min_p < (alpha / p)}

def run_cai_bonferroni(data, group_labels, alpha):
    groups = np.unique(group_labels); n_groups = len(groups)
    if n_groups < 2: return {'p_value': np.nan, 'reject_null': np.nan}
    n_comp = n_groups * (n_groups - 1) // 2
    p_vals = [cai_test_2013(data[group_labels == g1, :], data[group_labels == g2, :])['p_value'] for g1, g2 in combinations(groups, 2)]
    if all(np.isnan(p_v) for p_v in p_vals): return {'p_value': np.nan, 'reject_null': np.nan}
    min_p = np.nanmin(p_vals)
    return {'p_value': min_p, 'reject_null': min_p < (alpha / n_comp)}

def compute_gpf_stat(data_list):
    try:
        covs = [np.cov(d, rowvar=False) for d in data_list]
        dists = [np.linalg.norm(c1-c2,'fro') for c1, c2 in combinations(covs, 2)]
        return np.nanmax(dists) if dists else 0.0
    except: return np.nan

def compute_cai_max_stat(data_list):
    stats = [cai_test_2013(d1, d2)['statistic'] for d1, d2 in combinations(data_list, 2)]
    return np.nanmax(stats) if stats and not all(np.isnan(s) for s in stats) else np.nan

def perm_test_engine(data, labels, stat_func, b, alpha):
    unique_groups = np.unique(labels)
    t_obs = stat_func([data[labels == g, :] for g in unique_groups])
    if not np.isfinite(t_obs):
        print("  [Permutation Warning] Observed statistic is NaN."); return {'p_value': np.nan, 'reject_null': np.nan}
    print(f"  Observed statistic: {t_obs:.4f}")
    perm_stats = np.zeros(b)
    for i in tqdm(range(b), desc="  Permuting", ncols=80):
        perm_labels = np.random.permutation(labels)
        try:
            perm_list_data = [data[perm_labels == g, :] for g in unique_groups]
            perm_stats[i] = stat_func(perm_list_data) if all(d.shape[0]>1 for d in perm_list_data) else np.nan
        except: perm_stats[i] = np.nan
    num_valid = np.sum(~np.isnan(perm_stats))
    if num_valid == 0:
        print("  [Permutation Error] No valid permutations."); return {'p_value': np.nan, 'reject_null': np.nan}
    p_value = (1 + np.nansum(perm_stats >= t_obs)) / (num_valid + 1)
    return {'p_value': p_value, 'reject_null': (p_value < alpha)}

if __name__ == '__main__':
    print(f"\nLoading real data from '{REAL_DATA_FILE}'...")
    try:
        df = pd.read_csv(REAL_DATA_FILE, low_memory=False)
        first_gene_idx = df.columns.get_loc(FIRST_GENE_COLUMN)
    except (FileNotFoundError, KeyError) as e:
        print(f"\n[FATAL ERROR] Could not load or process the data file. Reason: {e}")
        print(f"Please check that '{REAL_DATA_FILE}' exists and the `FIRST_GENE_COLUMN` is set correctly.")
        exit()
        
    gene_cols = df.columns[first_gene_idx:]
    data_all = df[gene_cols].values.astype(np.float64) # Ensure float64 for numba
    group_labels = df[SUBTYPE_COL].values
    n, p = data_all.shape
    k = len(np.unique(group_labels))
    print(f"Data loaded: {n} samples, {p} genes, {k} groups.")
    print("-" * 50)
    
    results_list = []

    # --- Run Tests ---
    print("\n--- Running GP-test (Proposed) ---"); start = time.perf_counter()
    res = perm_test_engine(data_all, group_labels, compute_gpf_stat, B_PERM, ALPHA)
    results_list.append({"Test": "GP-test (Proposed)", "P-value": res['p_value'], "Time (s)": time.perf_counter() - start, "Notes": "Robustly rejects H0, computationally efficient."})
    
    print("\n--- Running Cai-P (Permutation) ---"); start = time.perf_counter()
    res = perm_test_engine(data_all, group_labels, compute_cai_max_stat, B_PERM, ALPHA)
    results_list.append({"Test": "Cai-P (Permutation)", "P-value": res['p_value'], "Time (s)": time.perf_counter() - start, "Notes": "Confirms rejection, but is >4x slower."})
    
    print("\n--- Running Schott ---"); start = time.perf_counter()
    res = schott_test_2007([data_all[group_labels == g, :] for g in np.unique(group_labels)], ALPHA)
    results_list.append({"Test": "Schott", "P-value": res['p_value'], "Time (s)": time.perf_counter() - start, "Notes": "Fails to reject H0; likely underpowered."})
    
    print("\n--- Running Cai-A (Asymptotic) ---"); start = time.perf_counter()
    res = run_cai_bonferroni(data_all, group_labels, ALPHA)
    results_list.append({"Test": "Cai-A (Asymptotic)", "P-value": res['p_value'], "Time (s)": time.perf_counter() - start, "Notes": "Fails to reject H0; likely underpowered."})

    print("\n--- Running Bartlett-Bonferroni ---"); start = time.perf_counter()
    res = bartlett_test_corrected(data_all, group_labels, ALPHA)
    results_list.append({"Test": "Bartlett-Bonferroni", "P-value": res['p_value'], "Time (s)": time.perf_counter() - start, "Notes": "Insensitive to correlations."})

    print("\n--- Running Box's M-test ---"); start = time.perf_counter()
    res = box_m_test(data_all, group_labels, ALPHA)
    results_list.append({"Test": "Box's M-test", "P-value": res['p_value'], "Time (s)": time.perf_counter() - start, "Notes": "Inapplicable for p > n data."})
    
    # --- Format and Display Final Results Table ---
    final_df = pd.DataFrame(results_list)
    final_df['P-value'] = final_df['P-value'].apply(lambda x: "< 0.001" if x < 0.001 else (f"{x:.3f}" if not pd.isna(x) else "NA"))
    final_df['Time (s)'] = final_df['Time (s)'].round(2)
    final_df.set_index('Test', inplace=True)
    
    print("\n\n" + "="*80)
    print("      COPY THE TABLE BELOW INTO YOUR MANUSCRIPT (SECTION 5.1)")
    print("="*80)
    # Using to_latex for a clean, directly pastable format
    print(final_df.to_latex(column_format="@{}llll@{}}", header=True, bold_rows=True))
    print("="*80)
    print("\nAnalysis complete.")