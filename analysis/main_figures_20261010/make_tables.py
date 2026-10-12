"""Tables of the manuscript as CSV files.
Table1.csv: population-weighted rates in 1990 and 2023 and gaps between low-SDI and high-SDI groups (S1 Text, Table A2).
Table2_EN.csv, Table2_ZH.csv: relative disease positions and late-period factor status of key countries (main text, Table 2).
2026-10-11: persistent values carry the percentage of 200 GBD uncertainty draws in which the record stayed persistent (S1_UI_LABEL_FREQ.csv),
and a column gives the local studies compared with each profile (S2 Text; LIT_CONCORDANCE.csv). Run after s1_sensitivity_20261010/run_ui_mc.py."""
from pathlib import Path
import pandas as pd

H = Path(__file__).resolve().parent; A = H.parent
# ---------- Table 1
g = pd.read_csv(A / 'results_revision_20261010' / 'VERIFY_INTRO_GLOBAL_RATES.csv')
s = pd.read_csv(A / 'sdi_gaps_official_tiers_20261010' / 'SDI_GAPS_OFFICIAL_SUMMARY.csv', header=[0, 1], index_col=[0, 1])
LL = pd.read_csv(A / 'lri_gap_instability_20261010' / 'RR_BY_DESIGN.csv')
KEY = {302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}; MN = {1: '死亡', 6: '发病'}
NE = {302: 'Diarrhoeal disease', 322: 'LRI', 332: 'Meningitis'}
rows = []
for m in [1, 6]:
    for c in [302, 322, 332]:
        r = g[(g.cause_id == c) & (g.measure_id == m)].iloc[0]; k = KEY[c] + MN[m]
        fx = s.loc[('主分析：固定组成员', k)]; sv = s.loc[('敏感性：各期重新赋级', k)]; fl = LL[LL.dz == k].sort_values('year').RR_fixedL.values
        rows.append({'Outcome': NE[c] + {1: ' mortality', 6: ' incidence'}[m],
                     'All locations, 1990': r['1990'], 'All locations, 2023': r['2023'], 'All locations, decrease (%)': 100 * (1 - r['2023'] / r['1990']),
                     'Low-SDI group, 1990': fx[('low', '1990')], 'Low-SDI group, 2023': fx[('low', '2023')], 'Low-SDI group, decrease (%)': fx[('decline_low', '%')],
                     'High-SDI group, 1990': fx[('high', '1990')], 'High-SDI group, 2023': fx[('high', '2023')], 'High-SDI group, decrease (%)': fx[('decline_high', '%')],
                     'Group rate ratio, 1990': fx[('RR', '1990')], 'Group rate ratio, 2023': fx[('RR', '2023')],
                     'Rate difference, 1990': fx[('RD', '1990')], 'Rate difference, 2023': fx[('RD', '2023')],
                     'Rate ratio, membership fixed in 2011-2023, 1990': fl[0], 'Rate ratio, membership fixed in 2011-2023, 2023': fl[1],
                     'Rate ratio, groups reassigned each period, 1990': sv[('RR', '1990')], 'Rate ratio, groups reassigned each period, 2023': sv[('RR', '2023')]})
pd.DataFrame(rows).round(2).to_csv(H / 'Table1.csv', index=False, encoding='utf-8-sig')

# ---------- Table 2: key countries
# Locations discussed in the text, grouped by the pattern they illustrate. Reference ratios are for 2011-2023;
# † marks a location in the same tail in all three periods (persistent), * a tail in 2011-2023 that is not persistent.
GROUPS = [
    ('Persistent high tail, low or low-middle SDI in 1990–2000', '持续高尾（1990—2000年为低或中低SDI）', ['South Africa', 'Nigeria', 'Indonesia', 'Botswana', 'Azerbaijan']),
    ('Persistent low tail, low SDI in 1990–2000', '持续低尾（1990—2000年为低SDI）', ['Bangladesh', 'Viet Nam', 'Nicaragua', 'Solomon Islands']),
    ('Relative position improved over time', '相对位置随时间改善', ['China']),
    ('Excess LRI incidence at high development', '高发展水平下LRI发病偏高', ['Kuwait', 'Qatar', 'Saudi Arabia', 'Poland', 'Czechia', 'Singapore']),
    ('Incidence and mortality in opposite directions', '发病与死亡方向相反', ['Australia', 'Greenland'])]
NAMEZ = {'South Africa': '南非', 'Nigeria': '尼日利亚', 'Indonesia': '印度尼西亚', 'Botswana': '博茨瓦纳', 'Azerbaijan': '阿塞拜疆', 'Bangladesh': '孟加拉国',
         'Viet Nam': '越南', 'Nicaragua': '尼加拉瓜', 'Solomon Islands': '所罗门群岛', 'China': '中国', 'Kuwait': '科威特', 'Qatar': '卡塔尔', 'Saudi Arabia': '沙特阿拉伯',
         'Poland': '波兰', 'Czechia': '捷克', 'Singapore': '新加坡', 'Australia': '澳大利亚', 'Greenland': '格陵兰'}
OUT6 = [('LRI发病', 'LRI incidence', 'LRI发病'), ('LRI死亡', 'LRI mortality', 'LRI死亡'), ('腹泻病发病', 'Diarrhoea incidence', '腹泻病发病'),
        ('腹泻病死亡', 'Diarrhoea mortality', '腹泻病死亡'), ('脑膜炎发病', 'Meningitis incidence', '脑膜炎发病'), ('脑膜炎死亡', 'Meningitis mortality', '脑膜炎死亡')]
FAC = [('basic_water_pct', 'Water', '饮水'), ('basic_sanitation_pct', 'Sanitation', '卫生设施'), ('clean_cooking_pct', 'Clean cooking', '清洁烹饪'),
       ('dtp3_pct', 'DTP3', 'DTP3'), ('hib3_pct', 'Hib3', 'Hib3'), ('pcv3_pct', 'PCV3', 'PCV3'), ('stunting_u5_pct', 'Stunting', '生长迟缓'),
       ('low_birthweight_pct', 'Low birthweight', '低出生体重'), ('uhc_service_index', 'UHC', 'UHC'), ('gbd_ambient_pm25_ug_m3', 'PM2.5', 'PM2.5'),
       ('secondhand_smoke_sev', 'Second-hand smoke', '二手烟')]
TAB = {'EN': {'低SDI': 'L', '中低SDI': 'LM', '中等SDI': 'M', '中高SDI': 'HM', '高SDI': 'H'}, 'ZH': {'低SDI': '低', '中低SDI': '中低', '中等SDI': '中等', '中高SDI': '中高', '高SDI': '高'}}
pr = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_COUNTRY_DISEASE_PROFILES.csv'); TT = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
TAILS = ('稳健高尾', '稳健低尾')
# 2026-10-11（审稿意见R2-M1）：持续尾部值后括注其在200次GBD不确定性抽样中仍为同一持续尾部的百分比
UF = pd.read_csv(A / 's1_sensitivity_20261010' / 'S1_UI_LABEL_FREQ.csv')
DZK = {'LRI发病': (322, 6), 'LRI死亡': (322, 1), '腹泻病发病': (302, 6), '腹泻病死亡': (302, 1), '脑膜炎发病': (332, 6), '脑膜炎死亡': (332, 1)}
UFI = UF.set_index(['location_id', 'cause_id', 'measure_id'])
def keep_pct(lid, dz, tail):
    u = UFI.loc[(lid,) + DZK[dz]]; k = u['persist_high' if tail == '稳健高尾' else 'persist_low']; return 100 if k == u.n_draws else min(99, int(100 * k / u.n_draws + 0.5))   # 四舍五入（半数进位）；100 只表示全部抽样
# 2026-10-11：与地方研究的对照（S2 Text；country_profiles_20261011/LIT_CONCORDANCE.csv 中的编码），每项研究一条，括号内为关系
LIT = pd.read_csv(A / 'country_profiles_20261011' / 'LIT_CONCORDANCE.csv').set_index('id')
REL_CODE = {'Consistent': ('C', '一致'), 'Partly consistent': ('P', '部分一致'), 'Inconsistent': ('I', '不一致'), 'Outside factor set': ('O', '因素集外'), 'Background': ('B', '背景')}
TOPIC = {'S01': ('Hib vaccine', 'Hib疫苗'), 'S02': ('carriage', '鼻咽携带'), 'S03': ('water and sanitation', '饮水与卫生设施'), 'S04': ('child faeces disposal', '儿童粪便处置'),
         'S05': ('shared toilets', '共用厕所'), 'S06': ('unsafe WASH', '不安全的水与卫生'), 'S07': ('water quality', '饮水水质'), 'S08': ('malnutrition', '营养不良'),
         'S09': ('PCV', 'PCV'), 'S10': ('biomass cooking', '生物质燃料烹饪'), 'S11': ('Hib and PCV coverage', 'Hib与PCV覆盖'), 'S12': ('spatial variation in diarrhoea', '腹泻的空间差异'),
         'S13': ('PM2.5', 'PM2.5'), 'S15': ('PM2.5', 'PM2.5'), 'S16': ('field LRI incidence', '现场LRI发病'), 'S17': ('latrines and water storage', '厕所与储水'),
         'S18': ('within-country inequality', '国内不平等'), 'S19': ('Hib vaccine', 'Hib疫苗'), 'S20': ('PCV13', 'PCV13'), 'S21': ('water and sanitation', '饮水与卫生设施'),
         'S22': ('weather and RSV', '气象与RSV'), 'S23': ('oral rehydration', '口服补液'),
         'S26': ('solid fuel use', '固体燃料使用'), 'S27': ('RSV', 'RSV'), 'S28': ('PM2.5', 'PM2.5'), 'S29': ('infant feeding', '婴儿喂养'),
         'S30': ('air pollutants and RSV', '空气污染物与RSV'), 'S31': ('PCV10 introduction', 'PCV10引入')}   # 2026-10-12：补检中国、卡塔尔、沙特、波兰
NOT_SEARCHED = set()   # 2026-10-12：四国已补检
NOTE = {'S17': (', conference abstract', '，会议摘要')}   # 证据等级标注
def lit_cell(en, i):
    ids = [k for k in TOPIC if LIT.loc[k, 'loc_en'] == en]
    if not ids: return ('Not searched', '未检索')[i] if en in NOT_SEARCHED else ('None found', '未检索到')[i]
    fmt = (lambda t, c: f'{t} ({c})') if i == 0 else (lambda t, c: f'{t}（{c}）')
    return ('; ', '；')[i].join(fmt(TOPIC[k][i], REL_CODE[LIT.loc[k, 'class_en']][i] + NOTE.get(k, ('', ''))[i]) for k in ids)
num = lambda v: f'{v:.1f}'.replace('-', '−')
for L in ('EN', 'ZH'):
    i = 0 if L == 'EN' else 1; rows = []
    for ge, gz, locs in GROUPS:
        for en in locs:
            x = pr[pr.location == en].set_index('dz'); lid = int(x.location_id.iloc[0])
            path = ' → '.join(TAB[L][v] for v in TT[TT.location_id == lid].sort_values('period').tier)
            rec = {'Group': (ge, gz)[i], 'Location': en if L == 'EN' else NAMEZ[en], 'SDI level': path}
            for dz, ne, nz in OUT6:
                r = x.loc[dz]; mark = '†' if (r.p1_category == r.p2_category == r.p3_category) and r.p3_category in TAILS else ('*' if r.p3_category in TAILS else '')
                rec[(ne, nz)[i]] = f'{r.p3_rate_ratio:.2f}{mark}' + (f'\n({keep_pct(lid, dz, r.p3_category)})' if mark == '†' else '')
            y = F[(F.location_id == lid) & (F.period == '2011–2023')].set_index('factor')
            adv = [f'{(fe, fz)[i]} {num(y.loc[k].favourable_residual)}' for k, fe, fz in FAC if k in y.index and y.loc[k].status == '明确不利']
            fav = [(fe, fz)[i] for k, fe, fz in FAC if k in y.index and y.loc[k].status == '明确有利']
            sep = '; ' if L == 'EN' else '；'
            rec[('Adverse departures', '不利偏离因素')[i]] = sep.join(adv) or '—'
            rec[('Favourable departures', '有利偏离因素')[i]] = (', ' if L == 'EN' else '、').join(fav) or '—'
            rec[('Local studies (relation)', '地方研究（关系）')[i]] = lit_cell(en, i)
            rows.append(rec)
    pd.DataFrame(rows).to_csv(H / f'Table2_{L}.csv', index=False, encoding='utf-8-sig')
print('Table1.csv, Table2_EN.csv, Table2_ZH.csv written')
