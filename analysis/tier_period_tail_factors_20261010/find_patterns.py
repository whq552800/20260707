"""从分面板数据中提取规律：SDI三层（低+中低、中等+中高、高）内合并计数后的不利/有利比例，高尾与低尾比较；按时期、病种、领域汇总。"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
C = pd.read_csv(HERE / 'TIER_PERIOD_TAIL_COMPOSITION.csv')
C = C[C.name != '家庭PM₂.₅']  # 2026-10-10 更正：与方法一致，只用18项确认版因素（家庭PM2.5属环境条件分析，不计入因素领域）
BAND = {'低SDI': '低+中低', '中低SDI': '低+中低', '中等SDI': '中等+中高', '中高SDI': '中等+中高', '高SDI': '高'}
DOM = {'基本饮水服务覆盖率': 'WaSH', '基本卫生设施覆盖率': 'WaSH', '清洁烹饪覆盖率': '清洁燃料', '百白破疫苗接种覆盖率': '免疫', 'Hib疫苗三剂覆盖率': '免疫', '肺炎球菌结合疫苗覆盖率': '免疫',
       '5岁以下儿童生长迟缓率': '营养', '低出生体重率': '营养', '营养不足人口比例': '营养', '全民健康覆盖服务指数': '卫生系统', '人均卫生支出': '卫生系统', '每千人医院床位数': '卫生系统',
       '每千人护士与助产士数': '卫生系统', '环境PM₂.₅年均暴露': '环境暴露', '家庭PM₂.₅': '环境暴露', '二手烟/环境烟草烟雾': '环境暴露', '人均GDP（购买力平价）': '社会经济', '中等教育毛入学率': '社会经济', '总和生育率': '社会经济'}
C['band'] = C.tier.map(BAND); C['domain'] = C.name.map(DOM); C['death'] = np.where(C.outcome_cn.str.endswith('死亡率'), '死亡', '发病')
def pooled(keys):
    g = C.groupby(keys + ['category'])[['明确不利', '明确有利', '可判定国家数']].sum()
    g['adv'] = 100 * g['明确不利'] / g['可判定国家数']; g['fav'] = 100 * g['明确有利'] / g['可判定国家数']
    w = g[['adv', 'fav', '可判定国家数']].unstack('category')
    out = pd.DataFrame({'高尾不利': w[('adv', '稳健高尾')], '低尾不利': w[('adv', '稳健低尾')], '高尾有利': w[('fav', '稳健高尾')], '低尾有利': w[('fav', '稳健低尾')],
                        'n高': w[('可判定国家数', '稳健高尾')], 'n低': w[('可判定国家数', '稳健低尾')]})
    out['不利差'] = out['高尾不利'] - out['低尾不利']; out['有利差'] = out['低尾有利'] - out['高尾有利']
    return out.round(6)
pd.set_option('display.width', 250)
print('一、领域 × SDI三层（六结局、三期合并）'); A = pooled(['domain', 'band']); print(A.to_string())
print('\n二、领域 × 时期（低+中低层，六结局合并）'); B = pooled(['band', 'domain', 'period']); print(B.loc['低+中低'].to_string())
print('\n二b、高SDI层 × 时期'); print(B.loc['高'].to_string())
print('\n三、病种 × 领域（低+中低层，三期合并）：不利差'); D = pooled(['band', 'outcome_cn', 'domain']); print(D.loc['低+中低']['不利差'].unstack().round(0).to_string())
print('\n三b、病种 × 领域（高SDI层）：不利差'); print(D.loc['高']['不利差'].unstack().round(0).to_string())
print('\n四、发病 vs 死亡 × 领域（低+中低层）'); E = pooled(['band', 'death', 'domain']); print(E.loc['低+中低'][['不利差', '有利差']].unstack(0).round(0).to_string())
print('\n五、对称性：不利差与有利差的相关（因素×结局×三层×时期，各组可判定≥10）')
F = pooled(['outcome_cn', 'band', 'period', 'name']); F = F[(F.n高 >= 10) & (F.n低 >= 10)]
print('n=', len(F), 'Spearman近似（Pearson on ranks）=', round(F[['不利差', '有利差']].rank().corr().iloc[0, 1], 2), '；两者同号比例', round((np.sign(F.不利差) == np.sign(F.有利差)).mean(), 2))
print('\n六、因素层面（三层合并、三期合并、六结局合并），按低+中低层不利差排序'); G = pooled(['band', 'name'])
print(G['不利差'].unstack(0)[['低+中低', '中等+中高', '高']].sort_values('低+中低', ascending=False).round(0).to_string())
for nm, df in [('DOMAIN_BY_BAND', A), ('DOMAIN_BY_BAND_PERIOD', B), ('OUTCOME_DOMAIN_BY_BAND', D), ('FACTOR_BY_BAND', G)]:
    df.to_csv(HERE / f'PATTERN_{nm}.csv', encoding='utf-8-sig')
