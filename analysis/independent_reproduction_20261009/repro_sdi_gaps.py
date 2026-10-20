import numpy as np, pandas as pd
from paths import PANEL
d=pd.read_csv(PANEL)
b=d[d.year==1990].drop_duplicates('location_id')[['location_id','sdi']]
cuts=np.quantile(b.sdi,[.2,.4,.6,.8])
b['q']=np.searchsorted(cuts,b.sdi,side='left')+1   # equal to cut -> lower group
d=d.merge(b[['location_id','q']],on='location_id')
g=d.groupby(['cause_id','measure_id','year','q']).apply(lambda x: 1e5*x.gbd_number.sum()/x.under5_population.sum(),include_groups=False).unstack('q')
out=pd.DataFrame({'low':g[1],'high':g[5]}); out['RR']=out.low/out.high; out['RD']=out.low-out.high
out=out.reset_index(); out.to_csv('REPRO_SDI_GAPS_ANNUAL.csv',index=False)
e=out[out.year.isin([1990,2023])].pivot_table(index=['cause_id','measure_id'],columns='year',values=['RR','RD'])
print('cuts',cuts, b.q.value_counts().sort_index().tolist()); print(e.round(2).to_string())
# check rate vs number consistency
chk=np.abs(d.gbd_number/d.under5_population*1e5-d.gbd_rate_per_100k)/d.gbd_rate_per_100k
print('max rel diff number/pop vs rate',chk.max())
