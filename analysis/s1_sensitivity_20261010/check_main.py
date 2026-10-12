from engine import *
r = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
r2 = classify(r); assert (r2.category.values == r.category.values).all()
import json
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**disease_claims(r), **factor_claims(r, F)}.items()}, indent=0, ensure_ascii=False, default=float))
