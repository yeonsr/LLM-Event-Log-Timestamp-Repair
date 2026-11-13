import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import pandas as pd
import run_audit as RA
import core as N, baselines_fixed as F

rows=[]
M={"B2_nb(emb,log1p,pre5)": N.lstm_repair,
   "B2_fix(tax-features)": F.make_lstm_repair_tax(epochs=30, patience=5, max_len=20)}
for log, ds in (("credit","credit_C1_30"),("pub","pub_C1_30")):
    for mn,fn in M.items():
        try:
            r=RA.score(log,ds,mn,fn); rows+=r
            a=[x for x in r if x["scope"]=="ALL_POLLUTED"][0]
            print("%-14s %-24s MedAE=%10.1f exact=%.3f reg=%.3f %ss"%(ds,mn,a["MedAE_rep"],a.get("exact_match",float('nan')),a.get("regression_rate",float('nan')),a["secs"]),flush=True)
        except Exception as ex:
            print("FAIL",ds,mn,repr(ex),flush=True)
pd.DataFrame(rows).to_csv(RA.OUT/"audit_b2.csv",index=False)
print("done")
