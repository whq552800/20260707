"""核验结果首段：204个地点人口加权的全样本率及降幅。"""
import sys; sys.path.insert(0,'../independent_reproduction_20261009')
from paths import *
import pandas as pd
d=pd.read_csv(PANEL); d=d[d.year.isin([1990,2023])]
g=d.groupby(['cause_id','measure_id','year']).apply(lambda x:1e5*x.gbd_number.sum()/x.under5_population.sum(),include_groups=False).unstack('year')
g['decline%']=100*(1-g[2023]/g[1990]); print(d.location_id.nunique()); print(g.round(1))
g.round(2).to_csv('VERIFY_INTRO_GLOBAL_RATES.csv',encoding='utf-8-sig')
