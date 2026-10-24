"""Regenerate manuscript graphics from primary data and audited geometry summaries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=ROOT / 'data')
parser.add_argument('--geometry', type=Path, default=ROOT / 'analysis/verified/followup_layerwise_verified.json')
parser.add_argument('--out', type=Path, required=True, help='New output directory; existing figures are not overwritten')
args = parser.parse_args()
DATA = args.data.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
COLORS = {'fp16': '#2a78d6', 'int8': '#1baf7a', 'int4': '#eb6834'}
STYLES = {'fp16': '-', 'int8': ':', 'int4': '--'}
MODELS = [('qwen2.5-7b', 'Qwen2.5-7B', 64), ('mistral-7b', 'Mistral-7B', 16), ('phi3.5-mini', 'Phi-3.5-mini', 8)]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c3c2b7', 'grid.color': '#e1e0d9', 'grid.linewidth': .6,
                     'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e', 'pdf.fonttype': 42})
d = pd.read_csv(DATA / 'combined_raw.csv.gz')
def save(fig, name):
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
def wilson(successes, n):
    p = successes/n; z = 1.95996398454
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half
for kind, name in [('word','fig1_accuracy_by_level'), ('numeric','fig2_numeric_control')]:
    sub = d[d.kind == kind]
    table = sub.groupby(['model','quant_mode','interference_level']).correct.agg(['mean','sum','count']).reset_index()
    table.to_csv(OUT/(name+'.csv'), index=False)
    fig, axes = plt.subplots(1,3,figsize=(10.8,3.1),sharey=True)
    for ax,(model,label,k) in zip(axes,MODELS):
        for q in ['fp16','int8','int4']:
            t=table[(table.model==model)&(table.quant_mode==q)]
            ax.plot(t.interference_level,t['mean'],STYLES[q],color=COLORS[q],label=q.upper(),lw=1.6)
        ax.set(xscale='log',ylim=(-.02,1.03),title=label,xlabel='Total updates, k')
        levels=sorted(sub[sub.model==model].interference_level.unique())
        ax.set_xticks(levels,[str(x) for x in levels]); ax.grid(axis='y'); ax.set_axisbelow(True)
    axes[0].set_ylabel('Exact-match accuracy'); axes[-1].legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,name)
# Key-level marginal intervals: paired inference remains in the tables.
fig,ax=plt.subplots(figsize=(7.3,3.3)); rows=[]
for i,(model,label,k) in enumerate(MODELS):
    for offset,q in zip([-.19,0,.19],['fp16','int8','int4']):
        s=d[(d.model==model)&(d.kind=='word')&(d.interference_level==k)&(d.quant_mode==q)].correct
        p=s.mean();lo,hi=wilson(s.sum(),len(s))
        ax.errorbar(i+offset,p,yerr=[[p-lo],[hi-p]],fmt={'fp16':'o','int8':'s','int4':'^'}[q],
                    color=COLORS[q],ms=6,capsize=3,label=q.upper() if i==0 else None)
        rows.append(dict(model=model,k=k,quant=q,accuracy=p,n=len(s),wilson_low=lo,wilson_high=hi))
ax.set_xticks(range(3),[f'{label}\nk={k}' for _,label,k in MODELS]);ax.set(ylim=(.25,.94),ylabel='Exact-match accuracy')
ax.grid(axis='y');ax.set_axisbelow(True);ax.legend(frameon=False,ncol=3);fig.tight_layout();save(fig,'fig3_key_level_comparison')
pd.DataFrame(rows).to_csv(OUT/'fig3_key_level_comparison.csv',index=False)
r=pd.read_csv(DATA/'reconstructed_trials.csv');keys=['model','seed','attribute','interference_level','trial_idx']
w=d[d.kind=='word'].merge(r[keys+['distractor_sequence']],on=keys,validate='many_to_one')
w['intrusion']=[int(not c and str(e) in str(seq).split('|')[:-1]) for c,e,seq in zip(w.correct,w.model_extracted,w.distractor_sequence)]
fig,ax=plt.subplots(figsize=(4.2,2.9));rows=[]
for i,q in enumerate(['fp16','int8','int4']):
    s=w[w.quant_mode==q].intrusion;p=s.mean();lo,hi=wilson(s.sum(),len(s))
    ax.errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',color=COLORS[q],ms=6,capsize=3)
    rows.append(dict(quant=q,rate=p,n=len(s),wilson_low=lo,wilson_high=hi))
ax.set_xticks(range(3),['FP16','INT8','INT4']);ax.set(xlim=(-.5,2.5),ylim=(.19,.27),ylabel='Same-key intrusion rate')
ax.grid(axis='y');ax.set_axisbelow(True);fig.tight_layout();save(fig,'fig4_intrusion_rate')
pd.DataFrame(rows).to_csv(OUT/'fig4_intrusion_rate.csv',index=False)
g=json.loads(args.geometry.read_text());by={m['model']:m for m in g['models']}
fig,axes=plt.subplots(2,3,figsize=(10.8,5.4),sharex='col');rows=[]
for col,(model,label,k) in enumerate(MODELS):
    m=by[model];c=m['cleaned_distinct_pairs_equal_prompt_seed'];t=pd.DataFrame(c['curves'])
    for q in ['fp16','int4']:
        axes[0,col].plot(t.layer,t[q],STYLES[q],color=COLORS[q],lw=1.6,label=q.upper())
    axes[0,col].set(title=f'{label}: k={k}, n={c["n_prompts"]}',ylim=(0,1.02))
    axes[0,col].legend(frameon=False,fontsize=8)
    axes[1,col].axhline(0,color='#c3c2b7',lw=.7)
    axes[1,col].plot(t.layer,t.delta_int4_minus_fp16,color=COLORS['int4'],lw=1.6)
    axes[1,col].set(xlabel='Hidden-state index (0 = embedding)')
    for ax in axes[:,col]: ax.grid(axis='y');ax.set_axisbelow(True)
    for row in c['curves']: rows.append(dict(model=model,**{key:row[key] for key in ['layer','fp16','int4','delta_int4_minus_fp16','n_prompts','n_distinct_string_pairs']}))
axes[0,0].set_ylabel('Within-prompt pairwise cosine');axes[1,0].set_ylabel('INT4 − FP16 cosine')
fig.tight_layout();save(fig,'fig5_layerwise_cosine');pd.DataFrame(rows).to_csv(OUT/'fig5_layerwise_cosine.csv',index=False)
print('Generated five figures and CSV companions.')
