'''
-------------------------------------------------------------------------------
Wonjin Lee
12-13-2020
-------------------------------------------------------------------------------
This Python program replicates the following paper:
Jason Abrevaya. 2006. "Estimating the effect of smoking on birth outcomes using
a matched panel data approach." Journal of Applied Econometrics.
-------------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import linearmodels as plm
from linearmodels.panel import compare

import os
path_dir = '/Users/wonjin/GitHub/birthweight-smoking'
os.chdir(path_dir)

#------------------------------------------------------------------------------
# 1. Table 2 (Matched panel #3)
#------------------------------------------------------------------------------
# Load the data set
cols = [
    'momid3','idx','stateres','dmage', 'dmeduc', 'mplbir', 'nlbnl', 
    'gestat' , 'dbirwt', 'cigar', 'smoke', 'male', 'year', 'married', 'hsgrad', 
    'somecoll', 'collgrad', 'agesq', 'black', 'adeqcode2', 'adeqcode3', 
    'novisit', 'pretri2', 'pretri3', 'proxy_exists', 'proxy_or_proxyhat'
    ]
df = pd.read_csv('birpanel.txt', sep='\s+', header=None, names=cols)

# Dummy for lowbirthweight (LBW) infants
df['LBW'] = 0
df.loc[df['dbirwt']<2500, 'LBW'] = 1
#df['LBW'].describe()

# Table II. Summary statistics
data_index_temp = [
    'dbirwt', 'LBW', 'gestat', 'smoke', 'dmage', 'dmeduc', 'black', 
    'adeqcode2', 'adeqcode3', 'novisit', 'pretri2', 'pretri3'
    ]
data_index = [
    'Birthweight (g)', 'LBW indicator', 'Gestation (wks)', 
    'Smoking indicator', 'Age', 'Education (yrs)', 'Black indicator', 
    'Kessner index = 2', 'Kessner index = 3', 'No prenatal visit', 
    'First prenatal visit in 2nd trimester', 
    'First prenatal visit in 3rd trimester', 
    '# cigarettes/day (for smokers)'
    ]
data_mean = list(df.describe()[data_index_temp].loc['mean'])
data_std = list(df.describe()[data_index_temp].loc['std'])
cond_cigar = (df["cigar"] >0) & (df['cigar'] <99)
data_mean.append(df.loc[cond_cigar, 'cigar'].describe().loc['mean'])
data_std.append(df.loc[cond_cigar, 'cigar'].describe().loc['std'])
data_table = pd.DataFrame(
    {'mean': [ round(elem, 3) for elem in data_mean ], 
    'std': [ round(elem, 3) for elem in data_std ]}, 
    index = data_index
    )
print(data_table)
print(data_table.to_latex())
print(data_table, file=open('sum_stats.txt','w'))

print('Share of smoking moms = ', round(df['smoke'].describe().loc['mean'], 2))
print('Share of smoking moms = ',
    round(df.groupby('momid3')['smoke'].mean().describe().loc['mean'], 2))

#------------------------------------------------------------------------------
# 2. Table 3, 4, and 6: Pooled OLS
#------------------------------------------------------------------------------
# A function for OLS regression tables
def reg_ols_table(reg_res, dep_name):
    table_temp = pd.DataFrame({'b': round(reg_res.params['smoke'], 2),
                        'se': round(reg_res.bse['smoke'], 2),
                        't': round(reg_res.tvalues['smoke'], 2),
                        'pval': round(reg_res.pvalues['smoke'], 2)}, 
                        index=[f"smoke on {dep_name}"])
    return table_temp

# Regressand = dbirwt
ols_dbirwt = smf.ols(
            formula='dbirwt ~ C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3', 
            data=df)
res_ols_dbirwt = ols_dbirwt.fit(cov_type='HC1')
table_ols1 = reg_ols_table(res_ols_dbirwt, 'dbirwt')

# Regressand = dbirwt (w/ SE clustered by momid3)
res_ols_dbirwt_cluster = ols_dbirwt.fit(
    cov_type='cluster', cov_kwds={'groups': df['momid3']}
    )
table_ols1_cluster = reg_ols_table(res_ols_dbirwt_cluster, 'dbirwt')

# Regressand = dbirwt (w/ additional regressor, cigar)
res_ols_dbirwt_cig = smf.ols(
            formula='dbirwt ~ C(year) + C(stateres) + C(nlbnl)'
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3 + cigar', 
            data=df,  subset=(df['cigar'] <99)
            ).fit(cov_type='HC1')
table_ols2 = reg_ols_table(res_ols_dbirwt_cig, 'dbirwt')

# Regressand = gestat
res_ols_gestat = smf.ols(
            formula='gestat ~ C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3', 
            data=df
            ).fit(cov_type='HC1')
table_ols3 = reg_ols_table(res_ols_gestat, 'gestat')

# Collect the coefficient of smoke for each regression into one table
table_all = pd.concat([table_ols1,table_ols1_cluster,table_ols2,table_ols3])
print(f'Coeff. of smoke: \n{table_all}\n')


# Show the three regression results in one table
# (1) Naive table
table_compare = summary_col(
        [res_ols_dbirwt, res_ols_dbirwt_cig, res_ols_gestat], 
        stars=True, float_format='%0.2f',
        model_names=['dbirwt\n(1)','dbirwt\n(2)','gestat\n(1)']
    )
print(table_compare)

# (2) Better way
all_regressors = sorted(
    list(
        set(res_ols_dbirwt.params.index) | set(res_ols_dbirwt_cig.params.index) 
        | set(res_ols_gestat.params.index)
        )
    )
# Drop the fixed effect coefficients.
all_regressors_no_fe = [
    var_name for var_name in all_regressors if not var_name.startswith('C(')
    ]
table_compare_no_fe = summary_col(
        [res_ols_dbirwt, res_ols_dbirwt_cig, res_ols_gestat], 
        stars=True, float_format='%0.2f',
        model_names=['dbirwt\n(1)','dbirwt\n(2)','gestat\n(1)'],
        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
            'R2':lambda x: "{:.2f}".format(x.rsquared)},
        regressor_order=all_regressors_no_fe,
        drop_omitted=True
        )
print(table_compare_no_fe)
print(table_compare_no_fe.as_latex())
print(table_compare_no_fe, file=open('OLS.txt','w'))


#------------------------------------------------------------------------------
# FE model
#------------------------------------------------------------------------------
'''
The author did not, but it is better to cluster SE by mom's id. 
    - Use .fit(cov_type='clustered', cluster_entity=True)
However, clustering does not make much difference.
'''
df_panel = df.set_index(['momid3', 'idx'], drop=False)

# A function for OLS regression tables
def reg_fe_table(reg_res, dep_name):
    table_temp = pd.DataFrame({'b': round(reg_res.params['smoke'], 2),
                        'se': round(reg_res.std_errors['smoke'], 2),
                        't': round(reg_res.tstats['smoke'], 2),
                        'pval': round(reg_res.pvalues['smoke'], 2)}, 
                        index=[f"smoke on {dep_name}"])
    return table_temp

# Regressand = dbirwt
res_fe_dbirwt = plm.PanelOLS.from_formula(
            formula='dbirwt ~ EntityEffects +  C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3',
            data=df_panel, drop_absorbed=True
            ).fit()
table_fe1 = reg_fe_table(res_fe_dbirwt, 'dbirwt')

# Regressand = dbirwt (w/ additional regressor, cigar)
res_fe_dbirwt_cig = plm.PanelOLS.from_formula(
            formula='dbirwt ~ EntityEffects +  C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3 + cigar',
            data=df_panel.loc[df_panel['cigar']<99, :], drop_absorbed=True
            ).fit()
table_fe2 = reg_fe_table(res_fe_dbirwt_cig, 'dbirwt')

# Regressand = gestat
res_fe_gestat = plm.PanelOLS.from_formula(
            formula='gestat ~ EntityEffects +  C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3',
            data=df_panel, drop_absorbed=True
            ).fit()
table_fe3 = reg_fe_table(res_fe_gestat, 'gestat')


# Collect the coefficient of smoke for each regression into one table
table_all = pd.concat([table_fe1,table_fe2,table_fe3])
print(f'Coeff. of smoke: \n{table_all}\n')

# Show the three regression results in one table
table_compare = compare(
    {'dbirwt\n(1)': res_fe_dbirwt, 'dbirwt\n(2)': res_fe_dbirwt_cig, 
    'gestat\n(1)':  res_fe_gestat}
    , stars = True
    )
print(table_compare)
print(table_compare.summary.as_latex())
print(table_compare.summary, file=open('FE.txt','w'))

#------------------------------------------------------------------------------
# RE model
#------------------------------------------------------------------------------
# Regressand = dbirwt
res_re_dbirwt = plm.RandomEffects.from_formula(
            formula='dbirwt ~ C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3',
            data=df_panel
            ).fit()
table_re1 = reg_fe_table(res_re_dbirwt, 'dbirwt')

# Regressand = gestat
res_re_gestat = plm.RandomEffects.from_formula(
            formula='gestat ~ C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3',
            data=df_panel
            ).fit()
table_re2 = reg_fe_table(res_re_gestat, 'gestat')

table_all = pd.concat([table_re1,table_re2])
print(f'Coeff. of smoke: \n{table_all}\n')


#------------------------------------------------------------------------------
# Hausman test of FE vs. RE
#------------------------------------------------------------------------------
'''
See: https://en.wikipedia.org/wiki/Durbin–Wu–Hausman_test
'''
b_fe = res_fe_dbirwt.params
b_fe_cov = res_fe_dbirwt.cov
b_re = res_re_dbirwt.params
b_re_cov = res_re_dbirwt.cov

# Find common coefficients among b_fe and b_re
com_coef = set(b_fe.index).intersection(b_re.index)

# Compute differences between FE model and RE model.
b_diff = np.array(b_fe[com_coef] - b_re[com_coef])
DoF = len(b_diff)
b_diff.reshape((DoF, 1))
b_cov_diff = np.array(
    b_fe_cov.loc[com_coef, com_coef] -b_re_cov.loc[com_coef, com_coef]
    )
b_cov_diff.reshape((DoF, DoF))

# The Hausman statistic:
Hausman_stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(Hausman_stat, DoF)

print(f'Hausman statistic = {round(Hausman_stat, 2)}')
print(f'pval: {pval}')

#------------------------------------------------------------------------------
# Estimation w/ proxy variable
#------------------------------------------------------------------------------
# Regressand = dbirwt
ols_dbirwt = smf.ols(
            formula='dbirwt ~ C(year) + C(stateres) + C(nlbnl)' 
            '+ smoke + male + dmage + agesq + hsgrad + somecoll + collgrad'
            '+ married + black + adeqcode2 + adeqcode3 + novisit' 
            '+ pretri2 + pretri3', 
            data=df.loc[df['proxy_or_proxyhat'] == 1, :]
            ).fit(cov_type='HC1')
table_ols = pd.DataFrame({'b': round(ols_dbirwt.params['smoke'], 4),
                        'se': round(ols_dbirwt.bse['smoke'], 4),
                        't': round(ols_dbirwt.tvalues['smoke'], 4),
                        'pval': round(ols_dbirwt.pvalues['smoke'], 4)}, 
                        index=['smoke'])
print(f'Coeff. of smoke: \n{table_ols}\n')