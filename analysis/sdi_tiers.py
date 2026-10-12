"""SDI 发展分层（GBD 2023 固定界值）。
界值不随研究地点或时期重新计算；各地点按每一时期年度SDI的算术平均值赋级。
低 [0, 0.5315638185)；中低 [0.5315638185, 0.6304845024)；中等 [0.6304845024, 0.6766624947)；
中高 [0.6766624947, 0.7188265995)；高 [0.7188265995, 1.0]（含上界）。"""
import numpy as np, pandas as pd
CUTS = [0.5315638185, 0.6304845024, 0.6766624947, 0.7188265995]
TIERS = ['低SDI', '中低SDI', '中等SDI', '中高SDI', '高SDI']
HIGH_CUT = CUTS[-1]

def tier(sdi):
    s = np.asarray(sdi, float)
    return np.array(TIERS, dtype=object)[np.searchsorted(CUTS, s, side='right')]

def period_tiers(panel):
    """panel 需含 location_id, year, sdi；返回 location_id, period, sdi_mean, tier。"""
    from repro_common import period_of
    ly = panel.drop_duplicates(['location_id', 'year'])[['location_id', 'year', 'sdi']].copy()
    ly['period'] = ly.year.map(period_of)
    t = ly.groupby(['location_id', 'period']).sdi.mean().rename('sdi_mean').reset_index()
    t['tier'] = tier(t.sdi_mean)
    return t

if __name__ == '__main__':   # writes analysis/SDI_PERIOD_TIERS.csv
    import sys
    from pathlib import Path
    H = Path(__file__).resolve().parent; sys.path.insert(0, str(H / 'independent_reproduction_20261009'))
    from paths import PANEL
    period_tiers(pd.read_csv(PANEL, usecols=['location_id', 'year', 'sdi'])).to_csv(H / 'SDI_PERIOD_TIERS.csv', index=False, encoding='utf-8-sig')
