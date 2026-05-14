# Replication Code: "No Country for Old Men"
## A Structural Investigation of Ageism and Depression in Türkiye

### Data
Data are from the 2023 Türkiye Older Persons Profile Survey (TYPA 2023), 
provided by TurkStat under a restricted access agreement.

**Access:** https://www.tuik.gov.tr (apply for microdata access)

### Requirements
```
pip install pandas numpy statsmodels factor_analyzer semopy matplotlib
```

### Usage
```python
python analysis_replication.py
```

Place `YASLI_PROFIL_FERT_MIKROVERI.csv` in the same directory.

### Contents
| Section | Code | Output |
|---|---|---|
| Scale validation | Cells 4 | GDS-30 KMO, eigenvalues |
| Descriptive stats | Cells 5 | Table 1 |
| Ordered probit | Cells 6 | Table 2 (happiness) |
| OLS regression | Cells 7 | Table 3 (GDS score) |
| Count model robustness | Cells 8 | Table 3 supplement |
| SEM | Cells 9 | Table 4 |
| Subgroup analyses | Cells 10 | Table 5 |
| Figure | Cells 11 | Figure 1 |

### Note
This code is provided for replication purposes. 
All analyses were conducted in Python 3.12.
