"""Publication figures from restored, verified participant/summary tables only.

No EEG, trial labels, model fitting, invented uncertainty or synthetic result rows.
Diagram illustrations in Figure 1 are explicitly schematic.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;T=ROOT/'tables'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,'axes.labelsize':9,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
COHORTS=['holdout','pan2023','weibo2014'];N={'holdout':11,'pan2023':14,'weibo2014':10}
LABEL={'holdout':'Stieger historical holdout','pan2023':'Pan','weibo2014':'Weibo'}
COLOR={'profile':'#0072B2','profile_map':'#D55E00','support_ea':'#666666'}
NAME={'profile':'ERFC','profile_map':'Matched MAP','support_ea':'Support EA'}
CC={'holdout':'#0072B2','pan2023':'#009E73','weibo2014':'#CC79A7'}
summary=pd.read_csv(T/'r10_method_budget_summaries_verified.csv')
primary=pd.read_csv(T/'r10_primary_comparisons_verified.csv')
person=pd.read_csv(T/'r10_matched_map_participant_rows.csv')
mechanism=pd.read_csv(T/'r10_exploratory_matched_map_mechanisms.csv')
paired=pd.read_csv(T/'r10_exploratory_paired_values.csv')
assert set(person.method)=={'profile','profile_map'}
assert all(person.n_seeds==3) and len(person)==70

def save(fig,name):
 fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(HERE/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(HERE/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.04)
 with Image.open(HERE/(name+'.png')) as raster:
  raster.convert('RGB').save(HERE/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
 plt.close(fig)

def letter(ax,s):ax.text(-.12,1.07,s,transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
def box(ax,x,y,w,h,text,fc='#EEF5F6',edge='#6F858A',size=7.7):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',fc=fc,ec=edge,lw=.9);ax.add_patch(p)
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size);return p

def arrow(ax,a,b,color='#566970',ls='-'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1,color=color,linestyle=ls))

# Figure 1: compact vector modules; no decorative backgrounds or rounded cards.
f=plt.figure(figsize=(6.70,4.60));gs=f.add_gridspec(3,1,height_ratios=[1.05,1.12,1.0],hspace=.38);f.subplots_adjust(left=.015,right=.985,top=.98,bottom=.03)
def module(ax,x,y,w,h,label,accent=False):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=COLOR['profile'] if accent else '#404040',lw=.8))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8.5,linespacing=1.3)
def link(ax,left,right,y=.52):
 ax.annotate('',xy=(right,y),xytext=(left,y),arrowprops={'arrowstyle':'-|>','lw':.8,'color':'#404040','mutation_scale':8})
a=f.add_subplot(gs[0]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');a.text(.015,.95,'A  Support fit',weight='bold',fontsize=9.5)
module(a,.015,.20,.17,.56,'Support prefix\nEEG + labels\nB = 4–20')
module(a,.24,.20,.21,.56,'Support EA\nFrozen EEGNet\nFeatures + logits')
module(a,.505,.20,.21,.56,'Residual objective J\nMatched MAP fit\nG = J₀ − Ĵ')
module(a,.77,.20,.215,.56,'Retained functions\nJ ≤ Ĵ + cG\nc = 0.1',accent=True)
for left,right in [(.185,.24),(.45,.505),(.715,.77)]:link(a,left,right,.49)
a.text(.5,.10,'Encoder, classifier and reference geometry are fixed.',ha='center',fontsize=8.5)
b=f.add_subplot(gs[1]);b.set(xlim=(0,1),ylim=(0,1));b.axis('off');b.text(.015,.96,'B  Query-specific decision',weight='bold',fontsize=9.5)
module(b,.015,.20,.17,.56,'Query features\nEA anchor\nmargin m₀(x)')
module(b,.24,.20,.21,.56,'Support projection\n+ null completion')
module(b,.505,.20,.21,.56,'Outer correction\ninterval [ℓ(x), u(x)]')
module(b,.77,.20,.215,.56,'δ(x) = proj[ℓ,u](0)\np(x) = σ(m₀ + δ)',accent=True)
for left,right in [(.185,.24),(.45,.505),(.715,.77)]:link(b,left,right,.49)
b.text(.5,.10,'0 ∈ [ℓ(x), u(x)] ⇒ δ(x) = 0; the anchor probability is retained.',ha='center',fontsize=8.5)
c=f.add_subplot(gs[2]);c.set(xlim=(0,1),ylim=(0,1));c.axis('off');c.text(.015,.98,'C  Acquired-prefix evaluation',weight='bold',fontsize=9.5)
start=.03;w=.025
for j in range(20):
 c.add_patch(Rectangle((start+j*w,.43),w*.87,.22,facecolor='#E7ECEF' if j<4 else 'white',edgecolor='#555555',lw=.55))
for end in [4,8,12,16,20]:c.text(start+(end-.5)*w,.27,str(end),ha='center',fontsize=8.5)
c.text(start+10*w,.78,'Acquired positions',ha='center',fontsize=8.5)
link(c,start+20*w+.025,.67,.54);module(c,.68,.39,.305,.30,'Common future query set')
c.text(.5,.05,'Stieger: 11 participants / 105 sessions; Pan: 14 / 28; Weibo: 10 / 10.',ha='center',fontsize=8.3)
save(f,'Fig1_ERFC_framework')

# Figure 2: unaltered cohort mean BA curves plus actual primary uncertainty at B20.
f,axes=plt.subplots(2,2,figsize=(6.70,4.80));curve=summary[summary.method.isin(NAME)].copy();curve['BA_percent']=100*curve.ba
curve.to_csv(HERE/'Fig2_source_budget_curves.csv',index=False)
for ax,cohort,lab in zip(axes.flat[:3],COHORTS,'ABC'):
 sub=curve[curve.cohort==cohort]
 for method in ['profile_map','support_ea','profile']:
  z=sub[sub.method==method].sort_values('budget');ax.plot(z.budget,z.BA_percent,color=COLOR[method],marker={'profile':'o','support_ea':'s','profile_map':'^'}[method],markersize=3.6,lw=1.5,label=NAME[method],alpha=.97)
 ax.set_title(f'{"Stieger" if cohort=="holdout" else LABEL[cohort]} (n = {N[cohort]})',loc='left');ax.set_xlabel('Acquired prefix positions');ax.set_ylabel('Balanced accuracy (%)');ax.set_xticks([4,8,12,16,20]);ax.grid(axis='y',alpha=.2,lw=.6)
 pad=.30 if cohort!='weibo2014' else .6;ax.set_ylim(sub.BA_percent.min()-pad,sub.BA_percent.max()+pad);letter(ax,lab)
axes[0,0].legend(frameon=False,loc='upper left',handlelength=2)
a=axes[1,1];z=primary[primary.name=='profile_vs_ea'].set_index('cohort').loc[COHORTS].reset_index();z.to_csv(HERE/'Fig2_source_primary_effects.csv',index=False)
for i,r in z.iterrows():
 ci=json.loads(r.ci95_pp);a.errorbar(r.effect_pp,i,xerr=[[r.effect_pp-ci[0]],[ci[1]-r.effect_pp]],fmt='o',color=CC[r.cohort],capsize=3,ms=5,lw=1.4)
a.axvline(0,color='#838B90',lw=.8,ls='--');a.set_yticks(range(3),['Stieger','Pan','Weibo']);a.invert_yaxis();a.set_xlabel('Δ balanced accuracy (pp)');a.set_title('ERFC versus EA at B20',loc='left');a.grid(axis='x',alpha=.18,lw=.6);letter(a,'D')
f.subplots_adjust(wspace=.39,hspace=.56,bottom=.12);save(f,'Fig2_accuracy_and_budget')

# Figure 3: paired exploratory effects must show both fewer introduced and fewer corrected errors.
f,axes=plt.subplots(1,3,figsize=(6.70,2.90));metrics=[('balanced_introduced_vs_ea','Introduced errors','ERFC − MAP (pp)'),('balanced_corrected_vs_ea','Corrected errors','ERFC − MAP (pp)'),('updated_query_fraction','Updated probabilities','ERFC − MAP (pp)')]
plotrows=[]
for ax,(metric,title,xlabel),lab in zip(axes,metrics,'ABC'):
 for i,cohort in enumerate(COHORTS):
  r=mechanism[(mechanism.cohort==cohort)&(mechanism.metric==metric)].iloc[0];plotrows.append(r.to_dict())
  ax.errorbar(r.effect_reported,i,xerr=[[r.effect_reported-r.ci95_low],[r.ci95_high-r.effect_reported]],fmt='o',color=CC[cohort],capsize=3,ms=5,lw=1.4)
  ax.annotate('pH='+str(Decimal(str(r.p_holm_18)).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP)),(r.effect_reported,i),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=8)
 ax.axvline(0,color='#838B90',lw=.8,ls='--');ax.set_yticks(range(3),['Stieger (11)','Pan (14)','Weibo (10)'] if ax==axes[0] else []);ax.set_ylim(2.55,-.55);ax.set_title(title,loc='left');ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.18,lw=.6);letter(ax,lab)
axes[0].set_xlim(-1.2,.12);axes[1].set_xlim(-1.95,.12);axes[2].set_xlim(-100,6)
# Inference details are in the caption, not a banner inside the graphic.
f.subplots_adjust(left=.15,right=.99,wspace=.3,bottom=.23,top=.83);pd.DataFrame(plotrows).to_csv(HERE/'Fig3_source_exploratory_effects.csv',index=False);save(f,'Fig3_selectivity_tradeoff')

# Figure 4: complete participant data, not repeated seeds, show the tradeoff distribution.
f,axes=plt.subplots(1,3,figsize=(6.70,3.0));a=axes[0]
for cohort in COHORTS:
 for method,marker in [('profile_map','^'),('profile','o')]:
  z=person[(person.cohort==cohort)&(person.method==method)];a.scatter(100*z.updated_query_fraction,100*z.balanced_introduced_vs_ea,color=CC[cohort],marker=marker,s=18,alpha=.72,edgecolor='white',lw=.25)
a.set_xlabel('Probability updates (%)');a.set_ylabel('Introduced errors vs EA (%)');a.set_title('Update selectivity',loc='left');a.set_xlim(-3,104);a.set_ylim(bottom=-.05);a.grid(alpha=.16,lw=.6);letter(a,'A')
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#666666',label='ERFC',ms=4),Line2D([],[],marker='^',color='none',markerfacecolor='#666666',label='Matched MAP',ms=4)],frameon=False,loc='upper left',fontsize=8)
allscatter=[]
for ax,metric,title,lab in [(axes[1],'ba','Balanced accuracy','B'),(axes[2],'negative_transfer_session_fraction_vs_ea','Negative-transfer frequency','C')]:
 for i,cohort in enumerate(COHORTS):
  d=paired[(paired.cohort==cohort)&(paired.metric==metric)].sort_values('participant_id').copy();d['plot_value']=100*d.difference;d['plot_y']=i+np.linspace(-.13,.13,len(d));allscatter.extend(d.to_dict('records'))
  ax.scatter(d.plot_value,d.plot_y,s=17,color=CC[cohort],alpha=.75,edgecolor='white',lw=.3)
  r=mechanism[(mechanism.cohort==cohort)&(mechanism.metric==metric)].iloc[0];ax.errorbar(r.effect_reported,i+.25,xerr=[[r.effect_reported-r.ci95_low],[r.ci95_high-r.effect_reported]],fmt='D',color='#222222',ms=3,capsize=2,lw=1)
 ax.axvline(0,ls='--',lw=.8,color='#838B90');ax.set_yticks(range(3),['Stieger','Pan','Weibo']);ax.set_ylim(2.6,-.45);ax.set_xlabel('ERFC − MAP (pp)');ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.16,lw=.6);letter(ax,lab)
f.subplots_adjust(left=.08,right=.99,bottom=.23,top=.83,wspace=.57)
person.to_csv(HERE/'Fig4_source_absolute_participant_rates.csv',index=False);pd.DataFrame(allscatter).to_csv(HERE/'Fig4_source_participant_differences.csv',index=False);save(f,'Fig4_participant_tradeoffs')

# Supplementary context: separate earlier protection evidence from new R10 evidence.
d=pd.read_csv(T/'r3_protection_participant_differences.csv').sort_values('delta_ba_pp');f,a=plt.subplots(figsize=(5.9,2.6));a.axhspan(2.39646724142858,6.909417253022452,color='#E5E5E5',alpha=.7);a.axhline(4.658872174687025,color=COLOR['profile'],lw=1.2,label='Participant mean and 95% CI');a.axhline(0,color='#888888',ls='--',lw=.8);a.scatter(range(len(d)),d.delta_ba_pp,color=CC['holdout'],s=24,zorder=3);a.set_xticks(range(len(d)),d.participant_id,rotation=40,ha='right');a.set_ylabel('Protected − unprotected BA (pp)');a.set_title('Earlier tangent-protection contrast (n = 11)',loc='left');a.text(.98,.04,'Holm p = 0.0439 (10 tests)',transform=a.transAxes,ha='right',fontsize=8);a.legend(frameon=False,loc='upper left',fontsize=8);f.subplots_adjust(bottom=.25);d.to_csv(HERE/'FigS1_source_earlier_protection.csv',index=False);save(f,'FigS1_earlier_protection')
print('Saved 4 main figures and 1 supplement as editable SVG, vector PDF, 600 dpi PNG and LZW TIFF')
