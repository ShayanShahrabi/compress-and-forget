"""Recompute primary manuscript summaries from supplied trial tables (CPU only)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

import argparse

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=ROOT / 'data')
parser.add_argument('--out', type=Path, required=True, help='New output directory; existing files are not overwritten')
args = parser.parse_args()
DATA = args.data.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
d = pd.read_csv(DATA / 'combined_raw.csv.gz')
r = pd.read_csv(DATA / 'reconstructed_trials.csv')
keys = ['model', 'seed', 'attribute', 'interference_level', 'trial_idx']
assert not d.duplicated(keys + ['quant_mode']).any()
assert not r.duplicated(keys).any()
w = d.pivot(index=keys, columns='quant_mode', values='correct')
assert len(w) == 23075 and not w.isna().any().any()
j = d.merge(r, on=keys, suffixes=('', '_replay'), validate='many_to_one')
assert (j.subject == j.subject_replay).all()
assert (j.gold.astype(str) == j.gold_replay.astype(str)).all()
word = j[j.kind == 'word'].copy()
word['intrusion'] = [int(not c and str(e) in str(s).split('|')[:-1])
                     for c, e, s in zip(word.correct, word.model_extracted, word.distractor_sequence)]
intr = word.pivot(index=keys, columns='quant_mode', values='intrusion')
a = int(((intr.fp16 == 1) & (intr.int4 == 0)).sum())
b = int(((intr.fp16 == 0) & (intr.int4 == 1)).sum())
p = binomtest(a, a+b, .5)
ci = binomtest(a, a+b).proportion_ci(method='exact')
report = {'paired_trials': len(w), 'intrusion': {
    'counts': word.groupby('quant_mode').intrusion.sum().to_dict(),
    'rates': word.groupby('quant_mode').intrusion.mean().to_dict(),
    'discordant': [a,b], 'odds_ratio': a/b, 'exact_p': p.pvalue,
    'or_ci': [ci.low/(1-ci.low), ci.high/(1-ci.high)]}, 'key_levels': []}
for model, level in [('qwen2.5-7b',64),('mistral-7b',16),('phi3.5-mini',8)]:
    subset = word[(word.model == model) & (word.interference_level == level)]
    x = subset.pivot(index=keys, columns='quant_mode', values='correct')
    for quant in ['int4','int8']:
        a = int(((x.fp16 == 1) & (x[quant] == 0)).sum())
        b = int(((x.fp16 == 0) & (x[quant] == 1)).sum())
        report['key_levels'].append({'model':model,'k':level,'quant':quant,'n':len(x),
            'fp16':float(x.fp16.mean()),'quant_accuracy':float(x[quant].mean()),
            'discordant':[a,b],'exact_p':binomtest(a,a+b,.5).pvalue})
(OUT / 'primary_verified.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
