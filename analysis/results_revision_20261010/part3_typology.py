"""持续高尾/持续低尾记录的晚期疾病相关短板，按晚期SDI等级分层（Fig 3B）。
规则：候选因素按疾病机制预先指定；某因素晚期“明确不利”即计为短板；不可评价/已达普及/方向不确定不计。
2026-10-11 修订（审稿意见）：UHC指数是包含免疫、卫生设施和就医等指标的综合指数，与其他领域有重叠，因此不再并入“基本服务”，
单独报告（uhc_adverse）；具体因素领域为基本服务（饮水、卫生设施、清洁烹饪）、免疫（DTP3、Hib3、PCV3；DTP3为常规免疫服务的代理）、
营养（生长迟缓、低出生体重）、暴露（环境PM2.5、二手烟）。没有任何可评价具体因素的记录单列为“无可评价因素”，不计入“无短板”。"""
import pandas as pd, numpy as np
pr=pd.read_csv('../independent_reproduction_20261009/REPRO_COUNTRY_DISEASE_PROFILES.csv')
F=pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
T=pd.read_csv('../SDI_PERIOD_TIERS.csv')
LATE='2011–2023'
DOM={'basic_water_pct':'基本服务','basic_sanitation_pct':'基本服务','clean_cooking_pct':'基本服务',
     'dtp3_pct':'免疫','hib3_pct':'免疫','pcv3_pct':'免疫','stunting_u5_pct':'营养','low_birthweight_pct':'营养',
     'gbd_ambient_pm25_ug_m3':'暴露','secondhand_smoke_sev':'暴露'}
REL={'LRI':['gbd_ambient_pm25_ug_m3','secondhand_smoke_sev','clean_cooking_pct','dtp3_pct','hib3_pct','pcv3_pct','stunting_u5_pct','low_birthweight_pct','uhc_service_index'],
     '腹泻病':['basic_water_pct','basic_sanitation_pct','stunting_u5_pct','uhc_service_index'],
     '脑膜炎':['dtp3_pct','hib3_pct','pcv3_pct','uhc_service_index']}
cats=['p1_category','p2_category','p3_category']
pr['persist']=np.where((pr[cats]=='稳健高尾').all(axis=1),'持续高尾',np.where((pr[cats]=='稳健低尾').all(axis=1),'持续低尾',None))
R=pr[pr.persist.notna()].copy()
R['disease']=R.dz.str.replace('发病','').str.replace('死亡','')
R['outcome']=np.where(R.dz.str.endswith('死亡'),'死亡','发病')
FL=F[F.period==LATE].set_index(['location_id','factor'])
tl=T[T.period==LATE].set_index('location_id').tier
def prof(r):
    adv=[];ev=0;uhc='不可评价'
    for f in REL[r.disease]:
        s=FL.loc[(r.location_id,f),'status'] if (r.location_id,f) in FL.index else '不可评价'
        if f=='uhc_service_index': uhc=s; continue
        if s in ('明确不利','明确有利','方向不确定','已达普及'): ev+=1
        if s=='明确不利': adv.append(f)
    doms=sorted(set(DOM[f] for f in adv))
    return pd.Series(dict(n_eval=ev,adv_factors='、'.join(adv),adv_domains='、'.join(doms),n_adv_domains=len(doms),
                          any_specific=len(adv)>0,uhc_status=uhc,uhc_adverse=uhc=='明确不利'))
R=pd.concat([R,R.apply(prof,axis=1)],axis=1)
R['tier_late']=R.location_id.map(tl)
R['band']=R.tier_late.map({'低SDI':'低+中低','中低SDI':'低+中低','中等SDI':'中等+中高','中高SDI':'中等+中高','高SDI':'高'})
R['type']=np.where(R.n_eval==0,'无可评价因素',np.where(R.n_adv_domains==0,'未见短板',np.where(R.n_adv_domains==1,'单一领域','多领域')))
keep=['location_id','location','dz','disease','outcome','persist','tier_late','band','n_eval','n_adv_domains','adv_domains','adv_factors','any_specific','uhc_status','uhc_adverse','type','p1_rate_ratio','p3_rate_ratio']
R[keep].to_csv('PART3_PERSISTENT_RECORD_PROFILES.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',250)
S=R.groupby(['persist','band','type']).size().unstack('type').fillna(0).astype(int); S['合计']=S.sum(axis=1)
print(S); S.to_csv('PART3_TYPE_BY_BAND.csv',encoding='utf-8-sig')
# 2026-10-11：按层与病种汇总（任一具体短板、UHC低于参照），供 Fig 3B 与 S1 Text
def summ(keys):
    g=R.groupby(keys+['persist']).agg(n=('any_specific','size'),n_eval0=('n_eval',lambda x:int((x==0).sum())),
        n_any=('any_specific','sum'),n_uhc_adverse=('uhc_adverse','sum'),uhc_evaluable=('uhc_status',lambda x:int((x!='不可评价').sum())))
    g['any_specific']=(100*g.n_any/(g.n-g.n_eval0)).round(6)            # 分母：至少有一项可评价具体因素的记录
    g['uhc_adverse']=(100*g.n_uhc_adverse/g.uhc_evaluable).round(6)      # 分母：UHC可评价的记录
    return g.reset_index()
summ(['band']).to_csv('PART3_SHORTFALL_BY_BAND.csv',index=False,encoding='utf-8-sig')
summ(['band','disease']).to_csv('PART3_SHORTFALL_BY_BAND_DISEASE.csv',index=False,encoding='utf-8-sig')
print(summ(['band']).to_string()); print(summ(['band','disease']).to_string())
