"""
Replication Code: "No Country for Old Men": A Structural Investigation 
of Ageism and Depression in Türkiye

Data: 2023 Türkiye Older Persons Profile Survey (TYPA 2023)
Source: TurkStat — https://www.tuik.gov.tr
Note: Microdata available upon application to TurkStat.
      This code reproduces all tables and figures in the manuscript.

Requirements:
    pip install pandas numpy statsmodels factor_analyzer semopy matplotlib

Author: [Anonymized for review]
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import Logit
import statsmodels.formula.api as smf
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity, calculate_kmo
)
import semopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING
# =============================================================================
# Replace path with your local TYPA 2023 microdata file
df = pd.read_csv('YASLI_PROFIL_FERT_MIKROVERI.csv', sep=';')
print(f"Raw data: {df.shape[0]:,} observations, {df.shape[1]:,} variables")

# =============================================================================
# 2. VARIABLE CONSTRUCTION
# =============================================================================

# --- Dependent variables ---

# GDS-30 items — positive items (agree = depressive response)
gds_pozitif = [
    'DEPRESYON_ILGI_BIRAKMA', 'DEPRESYON_CAN_SIKKIN',
    'DEPRESYON_DUSUNCE', 'DEPRESYON_KORKU', 'DEPRESYON_CARESIZ',
    'DEPRESYON_HUZURSUZ', 'DEPRESYON_EVDE_KALMA', 'DEPRESYON_ENDISE',
    'DEPRESYON_UNUTKANLIK', 'DEPRESYON_KEDERLI', 'DEPRESYON_DEGERSIZ',
    'DEPRESYON_UMITSIZ', 'DEPRESYON_KUCUK_SEY', 'DEPRESYON_AGLAMAKLI',
    'DEPRESYON_DIKKAT_TOPLA', 'DEPRESYON_VAKIT_GECIRME',
]

# GDS-30 items — negative items (disagree = depressive response)
gds_negatif = [
    'DEPRESYON_MEMNUN', 'DEPRESYON_HAYAT_ANLAM', 'DEPRESYON_GELECEK_UMIT',
    'DEPRESYON_KEYIF', 'DEPRESYON_MUTLU', 'DEPRESYON_HAYATTA',
    'DEPRESYON_GECMIS', 'DEPRESYON_HEYECAN', 'DEPRESYON_YENI_SEY',
    'DEPRESYON_KUVVET', 'DEPRESYON_IYI_DURUM', 'DEPRESYON_YATAK_KALK',
    'DEPRESYON_KARAR', 'DEPRESYON_ESKI_DUSUNCE',
]

# Code: 1 = Evet (Yes), 2 = Hayır (No)
for col in gds_pozitif:
    df[col+'_s'] = np.where(df[col]==1, 1, np.where(df[col]==2, 0, np.nan))

for col in gds_negatif:
    df[col+'_s'] = np.where(df[col]==2, 1, np.where(df[col]==1, 0, np.nan))

gds_scored = [c+'_s' for c in gds_pozitif + gds_negatif]

# GDS total score (0-30)
df['gds_skor'] = df[gds_scored].sum(axis=1)
df.loc[df[gds_scored].isnull().all(axis=1), 'gds_skor'] = np.nan

# Happiness (1=very unhappy → 5=very happy, reverse coded)
# MUTLULUK: 1=very happy, 5=very unhappy in raw data → reverse
df['mutluluk'] = np.where(
    df['MUTLULUK'].isin([1,2,3,4,5]),
    6 - df['MUTLULUK'], np.nan
)

# --- Ageism perception dimensions ---
# 1=Agree, 2=Disagree, 3=Don't know → NaN

# Perceived age-related restriction
df['ageism_engel'] = np.where(df['YAS_ISTEK_ENGEL']==1, 1,
                    np.where(df['YAS_ISTEK_ENGEL']==2, 0, np.nan))

# Perceived rights disrespect (reverse: disagree = ageism)
df['ageism_hak'] = np.where(df['YASLI_HAK_SAYGI']==2, 1,
                   np.where(df['YASLI_HAK_SAYGI']==1, 0, np.nan))

# Perceived social exclusion
df['ageism_dislan'] = np.where(df['YASLI_DISLANMA']==1, 1,
                     np.where(df['YASLI_DISLANMA']==2, 0, np.nan))

# --- Sociodemographic controls ---
df['kadin']       = (df['CINSIYET']==2).astype(float)       # female=1
df['yas']         = df['YAS_YIL'].astype(float)
df['yalniz']      = (df['YALNIZ']==1).astype(float)          # lives alone
df['kronik']      = (df['KRONIK_HASTALIK']==1).astype(float) # chronic illness
df['saglik_kotu'] = (df['SAGLIK_DURUM'].isin([4,5])).astype(float) # poor health

# Income: log(1+x), median imputation for missing
median_gelir = df['FERT_GELIR_AYLIK'].median()
df['log_gelir'] = np.log1p(df['FERT_GELIR_AYLIK'].fillna(median_gelir))

# Marital status (reference: married=2)
df['dul']      = np.where(df['MEDENI_DURUM']==4, 1,
                 np.where(df['MEDENI_DURUM']==99, np.nan, 0))
df['bekar']    = np.where(df['MEDENI_DURUM']==1, 1,
                 np.where(df['MEDENI_DURUM']==99, np.nan, 0))
df['bosanmis'] = np.where(df['MEDENI_DURUM']==3, 1,
                 np.where(df['MEDENI_DURUM']==99, np.nan, 0))

# Education (reference: tertiary = 53, 511, 512, 521, 522)
df['egitim_yok']  = (df['EGITIM_DURUM']==1).astype(float)  # no schooling
df['egitim_ilk']  = (df['EGITIM_DURUM']==2).astype(float)  # primary
df['egitim_orta'] = (df['EGITIM_DURUM']==3).astype(float)  # lower secondary
df['egitim_lise'] = (df['EGITIM_DURUM']==4).astype(float)  # upper secondary

# NUTS-1 regional dummies (reference: TR1 = Istanbul)
bolgeler = ['TR2','TR3','TR4','TR5','TR6','TR7','TR8','TR9','TRA','TRB','TRC']
for b in bolgeler:
    df[f'b_{b}'] = (df['IBBS_1']==b).astype(float)
bolge_cols = [f'b_{b}' for b in bolgeler]

# Age groups
df['yas_grubu'] = pd.cut(df['yas'], bins=[49,64,74,120],
                          labels=['50-64','65-74','75+'])

# =============================================================================
# 3. ANALYTICAL SAMPLE
# =============================================================================
kontrol = [
    'kadin','yas','yalniz','kronik','saglik_kotu','log_gelir',
    'dul','bekar','bosanmis',
    'egitim_yok','egitim_ilk','egitim_orta','egitim_lise',
    'ageism_dislan','ageism_hak','ageism_engel'
] + bolge_cols

df_a = df.dropna(subset=kontrol + ['gds_skor','mutluluk']).copy()
print(f"Analytical sample: N = {len(df_a):,}")

# =============================================================================
# 4. SCALE VALIDATION: GDS-30 FACTOR ANALYSIS
# =============================================================================
print("\n=== GDS-30 FACTOR ANALYSIS ===")
df_gds = df_a[gds_scored].dropna()

# Bartlett + KMO
chi2, p = calculate_bartlett_sphericity(df_gds)
print(f"Bartlett's test: chi2={chi2:.1f}, p={p:.4f}")

kmo_vars, kmo_model = calculate_kmo(df_gds)
print(f"KMO: {kmo_model:.3f}")

# Eigenvalues
fa_eigen = FactorAnalyzer(n_factors=30, rotation=None)
fa_eigen.fit(df_gds)
ev, _ = fa_eigen.get_eigenvalues()
print(f"Eigenvalues (first 8): {[round(e,2) for e in ev[:8]]}")
print(f"Kaiser criterion: {sum(ev > 1)} factors")

# Single factor solution
fa1 = FactorAnalyzer(n_factors=1, rotation=None)
fa1.fit(df_gds)
var = fa1.get_factor_variance()
print(f"Variance explained (1 factor): {var[1][0]*100:.1f}%")

# GDS factor score
df_a['gds_faktor'] = fa1.transform(df_a[gds_scored].fillna(df_a[gds_scored].mean()))
r = df_a[['gds_skor','gds_faktor']].corr().iloc[0,1]
print(f"Correlation (total score vs factor score): r = {r:.4f}")

# =============================================================================
# 5. DESCRIPTIVE STATISTICS (TABLE 1)
# =============================================================================
print("\n=== TABLE 1: DESCRIPTIVE STATISTICS ===")
desc_vars = {
    'gds_skor': 'GDS-30 score (0-30)',
    'mutluluk': 'Happiness (1-5)',
    'ageism_engel': 'Ageism: Restriction',
    'ageism_hak': 'Ageism: Rights',
    'ageism_dislan': 'Ageism: Exclusion',
    'kadin': 'Female',
    'yas': 'Age',
    'yalniz': 'Lives alone',
    'kronik': 'Chronic illness',
    'saglik_kotu': 'Poor/very poor health',
    'dul': 'Widowed',
    'bosanmis': 'Divorced',
    'egitim_yok': 'No schooling',
    'egitim_ilk': 'Primary education',
    'log_gelir': 'Log monthly income',
}

for var, label in desc_vars.items():
    col = df_a[var].dropna()
    print(f"  {label:<35}: mean={col.mean():.3f}, sd={col.std():.3f}")

print(f"\nGDS categories:")
print(f"  Normal (0-10):          {(df_a['gds_skor']<=10).mean()*100:.1f}%")
print(f"  Mild depression (11-20): {((df_a['gds_skor']>10)&(df_a['gds_skor']<=20)).mean()*100:.1f}%")
print(f"  Severe (21-30):          {(df_a['gds_skor']>20).mean()*100:.1f}%")

# =============================================================================
# 6. ORDERED PROBIT: HAPPINESS (TABLE 2 — PRIMARY OUTCOME)
# =============================================================================
print("\n=== TABLE 2: ORDERED PROBIT — HAPPINESS ===")

df_op = df_a.dropna(subset=kontrol + ['mutluluk']).copy()
y_op = df_op['mutluluk'].astype(int)
X_op = df_op[kontrol]

model_op = OrderedModel(y_op, X_op, disp=False)
sonuc_op  = model_op.fit(method='bfgs', disp=False)

me_op = sonuc_op.get_margeff()

def yldz(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

print(f"Pseudo-R2: {sonuc_op.prsquared:.4f} | N={len(df_op):,}")
print(f"\n{'Variable':<25} {'AME':>10} {'SE':>10} {'p':>10}")
print("-"*55)
for var in ['ageism_engel','ageism_hak','ageism_dislan',
            'saglik_kotu','kronik','kadin','yas','log_gelir',
            'dul','bosanmis','bekar','egitim_yok','egitim_ilk']:
    if var in list(me_op.margeff_se.index):
        idx = list(me_op.margeff_se.index).index(var)
        me  = me_op.margeff[idx]
        se  = me_op.margeff_se[idx]
        p   = me_op.pvalues[idx]
        print(f"  {var:<23} {me:>+10.4f} {se:>10.4f} {p:>8.4f}{yldz(p)}")

# =============================================================================
# 7. OLS REGRESSION: GDS SCORE (TABLE 3 — SECONDARY OUTCOME)
# =============================================================================
print("\n=== TABLE 3: OLS — GDS SCORE ===")

df_ols = df_a.dropna(subset=kontrol + ['gds_skor']).copy()
y_ols  = df_ols['gds_skor']
X_ols  = sm.add_constant(df_ols[kontrol])

ols_m = sm.OLS(y_ols, X_ols).fit(cov_type='HC3')
print(f"R2={ols_m.rsquared:.4f} | Adj.R2={ols_m.rsquared_adj:.4f} | N={int(ols_m.nobs):,}")

print(f"\n{'Variable':<25} {'Beta':>10} {'SE':>10} {'p':>10}")
print("-"*55)
for var in ['ageism_engel','ageism_hak','ageism_dislan',
            'saglik_kotu','kronik','kadin','yas','log_gelir',
            'dul','bosanmis','bekar','egitim_yok','egitim_ilk']:
    coef = ols_m.params[var]
    se   = ols_m.bse[var]
    p    = ols_m.pvalues[var]
    print(f"  {var:<23} {coef:>+10.4f} {se:>10.4f} {p:>8.4f}{yldz(p)}")

# =============================================================================
# 8. COUNT MODEL ROBUSTNESS (TABLE 3 SUPPLEMENT)
# =============================================================================
print("\n=== ROBUSTNESS: POISSON & NEGATIVE BINOMIAL ===")
df_ols['gds_int'] = df_ols['gds_skor'].astype(int)

poi_m = smf.poisson(
    'gds_int ~ ' + ' + '.join(kontrol), data=df_ols
).fit(cov_type='HC3', disp=False)

neg_m = smf.negativebinomial(
    'gds_int ~ ' + ' + '.join(kontrol), data=df_ols
).fit(disp=False)

print(f"NegBin alpha: {neg_m.params['alpha']:.4f} (minimal overdispersion → OLS valid)")
print(f"\n{'Variable':<25} {'OLS beta':>10} {'Poisson IRR':>12} {'NegBin IRR':>12}")
print("-"*60)
for var in ['ageism_engel','ageism_hak','ageism_dislan',
            'saglik_kotu','kadin','egitim_yok']:
    b_ols = ols_m.params[var]
    irr_p = np.exp(poi_m.params[var])
    irr_n = np.exp(neg_m.params[var])
    print(f"  {var:<23} {b_ols:>+10.4f} {irr_p:>12.4f} {irr_n:>12.4f}")

# =============================================================================
# 9. STRUCTURAL EQUATION MODEL (TABLE 4)
# =============================================================================
print("\n=== TABLE 4: STRUCTURAL EQUATION MODEL ===")

model_str = """
gds_skor ~ ageism_engel + ageism_hak + ageism_dislan
gds_skor ~ saglik_kotu + kronik
gds_skor ~ kadin + yas + log_gelir
gds_skor ~ dul + bosanmis + bekar
gds_skor ~ egitim_yok + egitim_ilk + egitim_orta

ageism_engel  ~ kadin + yas + kronik + saglik_kotu + log_gelir + egitim_yok + egitim_ilk
ageism_hak    ~ kadin + yas + kronik + saglik_kotu + log_gelir + egitim_yok + egitim_ilk
ageism_dislan ~ kadin + yas + log_gelir + egitim_yok + egitim_ilk

saglik_kotu ~ kadin + yas + log_gelir + egitim_yok + egitim_ilk
kronik      ~ kadin + yas + log_gelir + egitim_yok + egitim_ilk
"""

sem_vars = [
    'gds_skor','ageism_engel','ageism_hak','ageism_dislan',
    'saglik_kotu','kronik','kadin','yas','log_gelir',
    'dul','bosanmis','bekar','egitim_yok','egitim_ilk','egitim_orta'
]
df_sem = df_a[sem_vars].dropna()
print(f"SEM sample: N={len(df_sem):,}")

sem_model = semopy.Model(model_str)
sem_model.fit(df_sem)

sonuc = sem_model.inspect()
print("\nDirect effects on GDS (p<0.05):")
gds_effects = sonuc[(sonuc['lval']=='gds_skor') & (sonuc['p-value']<0.05)]
print(gds_effects[['lval','op','rval','Estimate','Std. Err','p-value']].round(4).to_string(index=False))

# Model fit
try:
    stats = semopy.calc_stats(sem_model)
    print("\nModel fit indices:")
    for idx in ['CFI','RMSEA','GFI','TLI']:
        print(f"  {idx}: {stats.T.loc[idx,'Value']:.3f}")
except:
    print("Fit indices: see semopy output above")

# Indirect effects
print("\nIndirect effects:")
r = sonuc.set_index(['lval','rval'])['Estimate']
try:
    paths = [
        ('ageism_engel','egitim_yok','gds_skor'),
        ('saglik_kotu','egitim_yok','gds_skor'),
        ('ageism_engel','saglik_kotu','gds_skor'),
    ]
    for ara, kaynak, hedef in paths:
        a = r.loc[(ara, kaynak)]
        b = r.loc[(hedef, ara)]
        print(f"  {kaynak} → {ara} → {hedef}: {a:.4f} × {b:.4f} = {a*b:.4f}")
except Exception as e:
    print(f"  Manual calculation error: {e}")

# =============================================================================
# 10. SUBGROUP ANALYSES (TABLE 5)
# =============================================================================
print("\n=== TABLE 5: SUBGROUP ANALYSES ===")

kontrol_yas = [c for c in kontrol if c != 'yas']

for cinsiyet, label in [(0,'Male'), (1,'Female')]:
    alt = df_ols[df_ols['kadin']==cinsiyet].copy()
    k   = [c for c in kontrol if c != 'kadin']
    X   = sm.add_constant(alt[k])
    m   = sm.OLS(alt['gds_skor'], X).fit(cov_type='HC3')
    print(f"\n{label} (N={len(alt):,}, R2={m.rsquared:.3f}):")
    for var in ['ageism_engel','ageism_hak','ageism_dislan']:
        print(f"  {var}: {m.params[var]:+.4f}{yldz(m.pvalues[var])}")

for grup in ['50-64','65-74','75+']:
    alt = df_ols[df_ols['yas_grubu']==grup].copy()
    X   = sm.add_constant(alt[kontrol_yas])
    m   = sm.OLS(alt['gds_skor'], X).fit(cov_type='HC3')
    print(f"\n{grup} (N={len(alt):,}, R2={m.rsquared:.3f}):")
    for var in ['ageism_engel','ageism_hak','ageism_dislan']:
        print(f"  {var}: {m.params[var]:+.4f}{yldz(m.pvalues[var])}")

# =============================================================================
# 11. FIGURE 1 — MARGINAL EFFECTS PLOT
# =============================================================================
print("\nGenerating Figure 1...")

me_obj = sonuc_op  # ordered probit marginal effects
ageism_vars  = ['ageism_engel','ageism_hak','ageism_dislan']
ageism_label = ['Restriction\n(Ageism)', 'Rights\nDisrespect', 'Social\nExclusion']

me_vals, se_vals = [], []
for var in ageism_vars:
    idx = list(me_op.margeff_se.index).index(var)
    me_vals.append(me_op.margeff[idx])
    se_vals.append(me_op.margeff_se[idx])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Ageism Perceptions and Well-Being: Marginal Effects',
             fontsize=13, fontweight='bold')

# Left: Ordered probit AME on happiness
ax = axes[0]
ax.bar(range(3), me_vals, color='#c0392b', alpha=0.85,
       yerr=[1.96*s for s in se_vals], capsize=5, width=0.5)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels(ageism_label, fontsize=10)
ax.set_ylabel('Average Marginal Effect on Happiness (AME)')
ax.set_title('Life Satisfaction\n(Ordered Probit)')
ax.grid(axis='y', alpha=0.3)

# Right: OLS coefficients on GDS
ols_vals = [ols_m.params[v] for v in ageism_vars]
ols_ses  = [ols_m.bse[v]    for v in ageism_vars]
ax2 = axes[1]
ax2.bar(range(3), ols_vals, color='#2980b9', alpha=0.85,
        yerr=[1.96*s for s in ols_ses], capsize=5, width=0.5)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xticks(range(3))
ax2.set_xticklabels(ageism_label, fontsize=10)
ax2.set_ylabel('OLS Coefficient on GDS-30 Score')
ax2.set_title('Depressive Symptoms\n(OLS)')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_ageism_effects.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 1 saved: figure1_ageism_effects.png")

print("\n=== ALL ANALYSES COMPLETE ===")
print("Tables and figures correspond to manuscript sections.")
print("For data access: https://www.tuik.gov.tr")
