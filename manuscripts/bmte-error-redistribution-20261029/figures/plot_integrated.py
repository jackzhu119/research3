"""Publication figures from archived participant predictions and frozen summaries.

All calculations are descriptive; no model is fitted. Input CSVs are local to
this directory. Cohorts keep their own reference, acquisition and averaging
contracts. Bootstrap intervals are read from the independent analysis output.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from PIL import Image,ImageOps,ImageDraw

HERE=Path(__file__).resolve().parent
IN=HERE/'input'
C_COLOR='#009E73';I_COLOR='#D55E00';G_COLOR='#222222'
COHORTS=['holdout','pan2023','weibo2014']
LAB={'holdout':'Stieger','pan2023':'Pan','weibo2014':'Weibo'}
N={'holdout':11,'pan2023':14,'weibo2014':10}
CC={'holdout':'#0072B2','pan2023':'#009E73','weibo2014':'#CC79A7'}
METHODS=['frozen','support_ea','head_bias','head_rank4','head_full','sgd10','sgd100','r3_protected','r4_xfit_select_stack','r8_fiber_bounded','r9_raw_anchored','r9_ea_anchored','profile_map','profile','profile_no_completion','profile_frozen_anchor','profile_quadratic','profile_variance_gate']
MLAB={'frozen':'Frozen source','support_ea':'Support EA','head_bias':'Bias-only head','head_rank4':'Rank-4 head','head_full':'Full residual head','sgd10':'Spatial/head SGD10','sgd100':'Spatial/head SGD100','r3_protected':'Support-KL tangent','r4_xfit_select_stack':'Cross-fit stack','r8_fiber_bounded':'Bounded covariance fiber','r9_raw_anchored':'Raw anchored GP','r9_ea_anchored':'EA anchored GP','profile_map':'Point-fit residual','profile':'Gain-retaining residual','profile_no_completion':'No-completion residual','profile_frozen_anchor':'Frozen-anchor residual','profile_quadratic':'Quadratic residual','profile_variance_gate':'Variance-gated residual'}
CSP_METHODS=['frozen','classifier_only','full_refit','aligned_full_refit']
CSP_SHORT=['Frozen','Classifier','Spatial +\nclassifier','EA + spatial\n+ classifier']
CSP_COLOURS={'frozen':'#777777','classifier_only':'#56B4E9','full_refit':'#6E4D96','aligned_full_refit':'#0072B2'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.1,'axes.titlesize':9,'axes.labelsize':8.1,'xtick.labelsize':7.6,'ytick.labelsize':7.6,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

M=pd.read_csv(IN/'neural_metrics.csv')
U=pd.read_csv(IN/'participant_metrics.csv',dtype={'budget':str})
CS=pd.read_csv(IN/'cohort_summary.csv',dtype={'budget':str})
COMP=pd.read_csv(IN/'component_summary.csv',dtype={'budget':str})
K=U[(U.study=='conventional')&(U.cohort=='Kumar2024')].copy()
K['cohort']=K.analysis_role.map({'main':'remaining12','development':'development6'})
B=U[(U.study=='conventional')&(U.cohort=='BNCI2014-001')].copy()
P=U[U.study=='neural'].copy().rename(columns={'participant':'participant_id'})
P['cohort']=P.cohort.map({'Stieger':'holdout','Pan':'pan2023','Weibo':'weibo2014'})
P['budget']=pd.to_numeric(P.budget)
S=CS[CS.study=='neural'].copy()
S['cohort']=S.cohort.map({'Stieger':'holdout','Pan':'pan2023','Weibo':'weibo2014'})
S['budget']=pd.to_numeric(S.budget)
E=U[U.study=='source_extension'].copy().rename(columns={'participant':'subject','reference_ba':'base_hit','ba':'method_hit'})
assert len(K)==360 and len(B)==36 and len(P)==6300 and len(S)==540 and len(M)==3150 and len(E)==85
for df in [K,B,P,E]:
 assert np.max(np.abs(df.C-df.I-df.G))<1e-12
 assert np.max(np.abs(df.C+df.I-df.D))<1e-12
assert len(P[P.reference.eq('frozen')])==3150

SOURCES={}
def record(name,df):
 df.to_csv(HERE/(name+'_source.csv'),index=False);SOURCES[name]={'rows':len(df),'columns':list(df.columns)}
def save(fig,name):
 for ext in ['pdf','svg']:fig.savefig(HERE/f'{name}.{ext}',bbox_inches='tight',pad_inches=.055)
 fig.savefig(HERE/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.055)
 with Image.open(HERE/f'{name}.png')as im:
  im.convert('RGB').save(HERE/f'{name}.png',dpi=(600,600))
  im.convert('RGB').save(HERE/f'{name}.tif',dpi=(600,600),compression='tiff_lzw')
 plt.close(fig)
def tag(ax,label,x=-.13,y=1.09):ax.text(x,y,label,transform=ax.transAxes,fontsize=10,weight='bold',va='bottom')
def grid(ax):ax.grid(alpha=.15,lw=.4);ax.set_axisbelow(True)
def box(ax,x,y,w,h,text,fc='white',fontsize=8):
 ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec='#444444',lw=.75))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.3)
def arrow(ax,p,q):ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='-|>',lw=.85,color='#555555',mutation_scale=9))
def summary(study,cohort,role,method,budget,reference='frozen'):
 q=CS[(CS.study==study)&(CS.cohort==cohort)&(CS.analysis_role==role)&(CS.method==method)&(CS.budget.astype(str)==str(budget))&(CS.reference==reference)]
 assert len(q)==1,(study,cohort,role,method,budget,reference,len(q))
 return q.iloc[0]
def curve(ax,study,cohort,role,method,budgets,x,colour,label,metric='ba'):
 q=pd.DataFrame([summary(study,cohort,role,method,b)for b in budgets]);m=100*q[metric+'_mean'].to_numpy(float)
 lo=100*q[metric+'_ci95_low'].to_numpy(float);hi=100*q[metric+'_ci95_high'].to_numpy(float)
 ax.fill_between(x,lo,hi,color=colour,alpha=.09,lw=0)
 ax.plot(x,m,'o-',color=colour,ms=3.3,lw=1.3,label=label)
 return q

# One experimental question, separately defined update paradigms.
f=plt.figure(figsize=(7.10,5.55));gs=f.add_gridspec(3,1,height_ratios=[1.1,1.3,1.1],hspace=.18);f.subplots_adjust(left=.025,right=.985,bottom=.065,top=.98)
a=f.add_subplot(gs[0]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
a.text(0,.98,'A  Matched calibration comparisons in five evaluation cohorts',weight='bold',fontsize=9)
box(a,.005,.16,.36,.64,'Conventional CSP/LDA updates\nKumar: 12 participants / 60 sessions\nBNCI: 9 participants / 9 sessions',fc='#F3F7FA',fontsize=7.9)
box(a,.405,.16,.59,.64,'Source-trained EEGNet calibration\nStieger: 11 / 105; Pan: 14 / 28; Weibo: 10 / 10\n18 configurations; three source-model seeds',fc='#F5F5F5',fontsize=8.1)
a.text(.5,.035,'56 participants in the main evaluation; sessions, seeds and configurations are repeated measures.',ha='center',fontsize=7.7)
b=f.add_subplot(gs[1]);b.set(xlim=(0,1),ylim=(0,1));b.axis('off')
b.text(0,.98,'B  Fixed query observations pair each update with its reference',weight='bold',fontsize=9)
box(b,.005,.22,.29,.58,'Current-session support\nKumar: 4–16; full = 19/20 labels\nBNCI: 20 valid labels\nNeural: 4–20 acquired positions',fontsize=7.6)
box(b,.365,.22,.265,.58,'Update chosen components\nCSP: classifier / filters / alignment\nNeural: heads / spatial maps\nFrozen source reference',fontsize=7.4)
box(b,.70,.22,.295,.58,'Matched held-out queries\nFixed support + query identities\nQuery labels used only in scoring\nSupport-EA sensitivity for EEGNet',fontsize=7.5)
arrow(b,(.295,.51),(.365,.51));arrow(b,(.63,.51),(.70,.51))
b.text(.5,.06,'Budget axes retain each acquisition contract; cohort scores are not pooled.',ha='center',fontsize=7.8)
c=f.add_subplot(gs[2]);c.set(xlim=(0,1),ylim=(0,1));c.axis('off')
c.text(0,.98,'C  Resolve net accuracy into changes in individual predictions',weight='bold',fontsize=9)
box(c,.005,.22,.29,.57,'Reference wrong → update correct\nCorrected mass C',fc='#EEF7F2',fontsize=7.5)
box(c,.35,.22,.29,.57,'Reference correct → update wrong\nIntroduced mass I',fc='#FFF1EA',fontsize=7.5)
box(c,.695,.22,.30,.57,'Disagreement D = C + I\nNet BA gain G = C − I\nParticipant-level summaries',fontsize=7.7)
c.text(.322,.51,'+',ha='center',va='center',fontsize=12,color='#555555');arrow(c,(.64,.51),(.695,.51))
c.text(.5,.055,'C and I use class-balanced query weights; all results are in percentage points.',ha='center',fontsize=7.9)
save(f,'Fig1_study_design')

# Full update path, followed by paired adjacent-operation error transitions.
f=plt.figure(figsize=(7.10,5.7));outer=f.add_gridspec(2,2,height_ratios=[1.2,1],wspace=.35,hspace=.62);f.subplots_adjust(left=.09,right=.985,top=.94,bottom=.14)
used=[]
for j,(cohort,people,part,budget,title)in enumerate([('Kumar2024',12,K[K.cohort=='remaining12'],'full','Kumar (N = 12; full support)'),('BNCI2014-001',9,B,'20','BNCI (N = 9; 20 labels)')]):
 ax=f.add_subplot(outer[0,j]);q=part[part.budget.astype(str)==str(budget)]
 mat=q.pivot(index='participant',columns='method',values='ba')[CSP_METHODS]
 for _,v in mat.iterrows():ax.plot(np.arange(4),100*v.to_numpy(float),color='#BBBBBB',lw=.6,alpha=.65,zorder=1)
 means=[]
 for i,m in enumerate(CSP_METHODS):
  z=summary('conventional',cohort,'main',m,budget);used.append(z.to_dict());v=100*z.ba_mean;means.append(v)
  err=[[100*(z.ba_mean-z.ba_ci95_low)],[100*(z.ba_ci95_high-z.ba_mean)]]
  ax.errorbar(i,v,yerr=err,fmt='o',ms=5,color=CSP_COLOURS[m],capsize=2,lw=1.1,zorder=3)
  ax.annotate(f'{v:.2f}',(i,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=7.5,color=CSP_COLOURS[m])
 ax.plot(np.arange(4),means,color='#222222',lw=1.15,zorder=2)
 ax.set_xticks(np.arange(4),CSP_SHORT);ax.set_xlim(-.25,3.25);ax.set_ylabel('Balanced accuracy (%)');ax.set_title(title,loc='left',fontsize=8.8);ax.set_ylim(35,101);grid(ax);tag(ax,'AB'[j])
 sub=outer[1,j].subgridspec(1,2,width_ratios=[2.0,1],wspace=.16);ad=f.add_subplot(sub[0]);ag=f.add_subplot(sub[1],sharey=ad)
 comparisons=['classifier_vs_frozen','spatial_vs_classifier','alignment_vs_full'];z=COMP[(COMP.cohort==cohort)&(COMP.analysis_role=='main')&(COMP.budget.astype(str)==str(budget))].set_index('comparison').loc[comparisons];used.extend(z.reset_index().to_dict('records'))
 yy=np.arange(3);ad.barh(yy,100*z.C_mean,color=C_COLOR,height=.55);ad.barh(yy,100*z.I_mean,left=100*z.C_mean,color=I_COLOR,height=.55)
 ag.axvline(0,color='#999999',lw=.65);ag.hlines(yy,0,100*z.G_mean,color='#555555',lw=.8)
 for i,(_,r)in enumerate(z.iterrows()):
  ag.errorbar(100*r.G_mean,i,xerr=[[100*(r.G_mean-r.G_ci95_low)],[100*(r.G_ci95_high-r.G_mean)]],fmt='D',color=G_COLOR,ms=3,capsize=2,lw=.75)
 ad.set_yticks(yy,['Classifier','Spatial refit','Alignment']);ag.set_yticks(yy);ag.tick_params(labelleft=False)
 for ax in [ad,ag]:ax.set_ylim(2.5,-.55);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15,lw=.4);ax.set_axisbelow(True)
 ad.set_xlim(0,31);ad.set_xticks([0,10,20,30]);ad.set_xlabel('C + I (pp)');ag.set_xlim(-2,10);ag.set_xticks([0,5,10]);ag.set_xlabel('G (pp)');ad.set_title('Adjacent update contributions',loc='left',fontsize=8.3);tag(ad,'CD'[j],x=-.25)
f.legend(handles=[Patch(fc=C_COLOR,label='Corrected C'),Patch(fc=I_COLOR,label='Introduced I'),Line2D([0],[0],marker='D',color=G_COLOR,lw=.7,label='Net gain G')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.54,.015))
record('Fig2_component_error_balance',pd.DataFrame(used));save(f,'Fig2_component_error_balance')

# Equal cohort means and the distribution of participant-level changes.
f,axes=plt.subplots(2,2,figsize=(7.10,5.4));f.subplots_adjust(left=.10,right=.985,top=.93,bottom=.13,wspace=.36,hspace=.58)
z=P[(P.reference=='frozen')&(P.budget==20)&(P.method!='frozen')]
a=axes[0,0]
for c in COHORTS:
 q=z[z.cohort==c];a.scatter(100*q.D,100*q.G,s=8,color=CC[c],alpha=.34,edgecolor='none',label=f'{LAB[c]} (N = {N[c]})')
a.axhline(0,color='#888888',lw=.65);a.set(xlabel='Disagreement D (pp)',ylabel='Net gain G (pp)',title='Participant × configuration summaries');a.legend(frameon=False,loc='upper right',fontsize=7);grid(a);tag(a,'A')
methods=['head_bias','r4_xfit_select_stack'];labels=['Bias-only head','Cross-fit stack'];pan=z[(z.cohort=='pan2023')&z.method.isin(methods)];means=pan.groupby('method')[['C','I','D','G','ba']].mean().loc[methods]
assert abs(means.ba.iloc[0]-means.ba.iloc[1])<1e-12
a=axes[0,1];yy=np.arange(2);a.barh(yy,100*means.C,color=C_COLOR,height=.5);a.barh(yy,100*means.I,left=100*means.C,color=I_COLOR,height=.5)
for i,(_,r)in enumerate(means.iterrows()):a.text(100*r.D+.25,i,f'{100*r.D:.2f}',va='center',fontsize=8)
a.set_yticks(yy,labels);a.set_ylim(1.6,-.9);a.set_xlim(0,13.5);a.set_xlabel('Disagreement D = C + I (pp)');a.set_title('Pan: both BA = 63.36%',loc='left');a.text(.98,.96,'G = +0.18 pp for both',transform=a.transAxes,ha='right',va='top',fontsize=7.7);grid(a);tag(a,'B',x=-.25)
for a,metric,title,label in [(axes[1,0],'D','Disagreement within the same people','D (pp)'),(axes[1,1],'G','Net gains within the same people','G (pp)')]:
 v=pan.pivot(index='participant_id',columns='method',values=metric)[methods]
 for _,r in v.iterrows():a.plot([0,1],100*r.to_numpy(float),'-o',color='#AAAAAA',ms=3,lw=.7,alpha=.75)
 av=100*v.mean().to_numpy(float);a.plot([0,1],av,'D-',color='#222222',ms=5,lw=1.5,label='Participant mean')
 a.set_xticks([0,1],labels);a.set_xlim(-.18,1.18);a.set_ylabel(label);a.set_title(title,loc='left',fontsize=8.7);grid(a)
 if metric=='G':a.axhline(0,color='#888888',lw=.6)
tag(axes[1,0],'C');tag(axes[1,1],'D');axes[1,1].legend(frameon=False,loc='lower left',fontsize=7)
f.legend(handles=[Patch(fc=C_COLOR,label='Corrected C'),Patch(fc=I_COLOR,label='Introduced I')],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.56,.015))
record('Fig3_equal_accuracy_participant_changes',pan);save(f,'Fig3_equal_accuracy_participant_changes')

# The two budget axes retain their original acquisition contracts.
f,axes=plt.subplots(2,2,figsize=(7.10,5.4));f.subplots_adjust(left=.10,right=.985,bottom=.15,top=.85,wspace=.34,hspace=.6)
x=np.arange(5);budgets=['4','8','12','16','full'];used=[]
for m,label in zip(CSP_METHODS,['Frozen','Classifier update','Spatial + classifier','EA + spatial + classifier']):
 q=curve(axes[0,0],'conventional','Kumar2024','main',m,budgets,x,CSP_COLOURS[m],label);used.extend(q.to_dict('records'))
for m,colour,label in [('frozen','#777777','Frozen source'),('support_ea','#0072B2','Support EA')]:
 q=curve(axes[0,1],'neural','Weibo','main',m,['4','8','12','16','20'],x,colour,label);used.extend(q.to_dict('records'))
for ax,title in [(axes[0,0],'Kumar (N = 12)'),(axes[0,1],'Weibo (N = 10)')]:
 ax.set_ylabel('Balanced accuracy (%)');ax.set_title(title,loc='left');grid(ax)
axes[0,1].legend(frameon=False,loc='lower right',fontsize=7)
for ax,study,cohort,m,bs,title in [(axes[1,0],'conventional','Kumar2024','full_refit',budgets,'Spatial + classifier vs frozen'),(axes[1,1],'neural','Weibo','support_ea',['4','8','12','16','20'],'Support EA vs frozen')]:
 for metric,colour,label in [('C',C_COLOR,'Corrected C'),('I',I_COLOR,'Introduced I'),('G',G_COLOR,'Net gain G')]:
  q=curve(ax,study,cohort,'main',m,bs,x,colour,label,metric);used.extend(q.to_dict('records'))
 ax.axhline(0,color='#999999',lw=.6);ax.set_ylabel('Class-balanced change (pp)');ax.set_title(title,loc='left',fontsize=8.5);grid(ax)
for i in range(2):
 axes[i,0].set_xticks(x,['4','8','12','16','Full*']);axes[i,1].set_xticks(x,['4','8','12','16','20'])
for ax in axes.flat:ax.set_xlim(-.12,4.12)
axes[1,0].set_xlabel('Valid support labels (*19/20 at full support)');axes[1,1].set_xlabel('Acquired positions before quality control')
for ax,l in zip(axes.flat,'ABCD'):tag(ax,l)
f.legend(handles=[Line2D([0],[0],color=C_COLOR,marker='o',label='Corrected C'),Line2D([0],[0],color=I_COLOR,marker='o',label='Introduced I'),Line2D([0],[0],color=G_COLOR,marker='o',label='Net gain G')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.54,.015))
f.legend(handles=[Line2D([0],[0],color=CSP_COLOURS[m],marker='o',label=l)for m,l in zip(CSP_METHODS,['Frozen','Classifier update','Spatial + classifier','EA + spatial + classifier'])],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,1.01),title='Kumar update actions',title_fontsize=7.7)
record('Fig4_budget_profiles',pd.DataFrame(used).drop_duplicates());save(f,'Fig4_budget_profiles')

# Full neural grid, all eighteen configurations and all five budgets.
f,axes=plt.subplots(2,3,figsize=(7.10,8.0));f.subplots_adjust(left=.275,right=.965,bottom=.12,top=.93,wspace=.24,hspace=.3)
limit=float(np.ceil(100*S.G_mean.abs().max()/2)*2)
for ri,reference in enumerate(['frozen','support_ea']):
 for ci,c in enumerate(COHORTS):
  ax=axes[ri,ci];q=S[(S.reference==reference)&(S.cohort==c)];mat=q.pivot(index='method',columns='budget',values='G_mean').loc[METHODS,[4,8,12,16,20]]*100
  im=ax.imshow(mat,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_yticks(range(18),[MLAB[m]for m in METHODS]if ci==0 else[]);ax.set_xticks(range(5),['4','8','12','16','20']);ax.tick_params(axis='y',length=0,labelsize=7);ax.set_title(f'{LAB[c]} (N = {N[c]})',fontsize=8.5);ax.set_xlabel('Acquired positions')
  tag(ax,'ABCDEF'[ri*3+ci],x=-.18)
  if ci==0:ax.text(-1.42,.5,('Frozen-source reference'if ri==0 else'Support-EA reference'),transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8.4,weight='bold')
bar=f.colorbar(im,ax=axes,orientation='horizontal',fraction=.025,pad=.065,shrink=.55);bar.set_label('Net gain G (class-balanced pp)')
record('FigS1_neural_full_grid',S);save(f,'FigS1_neural_full_grid')

# Absolute performance retains the complete probability-score profile.
means=M.groupby(['cohort','method','budget'],as_index=False)[['ba','balanced_nll']].mean()
f,axes=plt.subplots(2,3,figsize=(7.10,8.0));f.subplots_adjust(left=.275,right=.96,bottom=.115,top=.935,wspace=.25,hspace=.35)
for ri,(metric,label,cmap,scale)in enumerate([('ba','Balanced accuracy (%)','viridis',100),('balanced_nll','Balanced negative log-likelihood','magma_r',1)]):
 lo=scale*means[metric].min();hi=scale*means[metric].max()
 for ci,c in enumerate(COHORTS):
  ax=axes[ri,ci];mat=means[means.cohort==c].pivot(index='method',columns='budget',values=metric).loc[METHODS,[4,8,12,16,20]]*scale
  im=ax.imshow(mat,aspect='auto',cmap=cmap,vmin=lo,vmax=hi);ax.set_yticks(range(18),[MLAB[m]for m in METHODS]if ci==0 else[]);ax.set_xticks(range(5),['4','8','12','16','20']);ax.tick_params(axis='y',length=0,labelsize=7);ax.set_title(LAB[c],fontsize=8.6);ax.set_xlabel('Acquired positions');tag(ax,'ABCDEF'[ri*3+ci],x=-.18)
 cb=f.colorbar(im,ax=list(axes[ri]),orientation='horizontal',fraction=.03,pad=.10,shrink=.6);cb.set_label(label,fontsize=8)
record('FigS2_neural_probability_scores',means);save(f,'FigS2_neural_probability_scores')

# Previously exposed development participants retain a distinct analysis role.
f,axes=plt.subplots(2,2,figsize=(7.10,5.2));f.subplots_adjust(left=.11,right=.985,top=.93,bottom=.18,wspace=.37,hspace=.6);used=[]
for j,(role,title)in enumerate([('main','Primary cohort (N = 12)'),('development','Development cohort (N = 6)')]):
 for m in CSP_METHODS:
  q=curve(axes[0,j],'conventional','Kumar2024',role,m,budgets,x,CSP_COLOURS[m],m);used.extend(q.to_dict('records'))
 axes[0,j].set_title(title,loc='left',fontsize=8.5);axes[0,j].set_ylabel('Balanced accuracy (%)');axes[0,j].set_xticks(x,['4','8','12','16','Full*']);axes[0,j].set_xlabel('Valid support labels');grid(axes[0,j])
 q=pd.DataFrame([summary('conventional','Kumar2024',role,m,'full')for m in CSP_METHODS[1:]]);used.extend(q.to_dict('records'));yy=np.arange(3);a=axes[1,j]
 a.barh(yy,100*q.C_mean,color=C_COLOR,height=.55);a.barh(yy,100*q.I_mean,left=100*q.C_mean,color=I_COLOR,height=.55);a.scatter(100*q.G_mean,yy,c=G_COLOR,marker='D',s=20,zorder=3)
 a.set_yticks(yy,['Classifier','Spatial refit','Aligned refit']);a.set_ylim(2.55,-.5);a.set_xlim(-2,40);a.set_xlabel('Change vs frozen (pp)');a.set_title('Full-support error transitions',loc='left',fontsize=8.5);a.axvline(0,color='#888888',lw=.6);grid(a)
for ax,l in zip(axes.flat,'ABCD'):tag(ax,l,x=-.18)
handles=[Line2D([0],[0],color=CSP_COLOURS[m],marker='o',label=l)for m,l in zip(CSP_METHODS,['Frozen','Classifier update','Spatial + classifier','EA + spatial + classifier'])]
f.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.55,.01))
record('FigS3_conventional_development',pd.DataFrame(used).drop_duplicates());save(f,'FigS3_conventional_development')

# Source-model replacement is a separate extension, not another target budget.
f,axes=plt.subplots(1,2,figsize=(7.10,3.4));f.subplots_adjust(left=.09,right=.98,top=.88,bottom=.19,wspace=.3)
a=axes[0];a.scatter(100*E.I,100*E.C,s=17,color='#0072B2',alpha=.7,edgecolor='none');lim=100*max(E.C.max(),E.I.max())+2;a.plot([0,lim],[0,lim],color='#888888',lw=.8,ls='--');a.scatter([100*E.I.mean()],[100*E.C.mean()],color='#222222',marker='D',s=42,zorder=3);a.set(xlim=(0,lim),ylim=(0,lim),xlabel='Introduced errors I (pp)',ylabel='Corrected errors C (pp)',title='Dreyer source-model replacement (N = 85)');grid(a);tag(a,'A')
a=axes[1];a.scatter(100*E.D,100*E.G,s=17,color='#0072B2',alpha=.7,edgecolor='none');a.axhline(0,color='#888888',lw=.7);a.scatter([100*E.D.mean()],[100*E.G.mean()],color='#222222',marker='D',s=42,zorder=3);a.set(xlabel='Prediction disagreement D (pp)',ylabel='Net BA change G (pp)',title='Same observations, expanded source cohort');a.text(.97,.97,f'Mean C = {100*E.C.mean():.2f} pp\nMean I = {100*E.I.mean():.2f} pp\nMean G = +{100*E.G.mean():.2f} pp',transform=a.transAxes,ha='right',va='top',fontsize=7.8);grid(a);tag(a,'B')
record('FigS4_source_model_extension',E);save(f,'FigS4_source_model_extension')

proof={'new_model_fitting':False,'new_hypothesis_tests':False,'renderer':'Matplotlib; deterministic numeric and vector graphics','source_counts':{'kumar_participant_rows':len(K),'bnci_participant_rows':len(B),'neural_participant_reference_rows':len(P),'neural_config_cohort_budget_reference_rows':len(S),'source_extension_participants':len(E)},'primary_participants_by_cohort':{'Kumar':12,'BNCI':9,'Stieger':11,'Pan':14,'Weibo':10},'primary_participant_total':56,'development_participants_separate':6,'source_extension_participants_separate':85,'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(IN.glob('*.csv'))},'figure_sources':SOURCES,'Pan_example':{'BA_percent':100*pan.groupby('method').ba.mean().iloc[0],'head_bias_D_pp':100*pan[pan.method=='head_bias'].D.mean(),'stack_D_pp':100*pan[pan.method=='r4_xfit_select_stack'].D.mean()},'formats':['editable SVG','vector PDF','600dpi RGB PNG','600dpi RGB LZW TIFF'],'generative_images':False,'unpooled_cohorts':True,'intervals':'Pointwise10000 participant-bootstrap intervals read from the independent cohort_summary/component_summary outputs; not calculated by the plotting script.'}
(HERE/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
files=list(sorted(HERE.glob('Fig[1-4]_*.png')))+list(sorted(HERE.glob('FigS*.png')))
thumbs=[]
for p in files:
 with Image.open(p)as im:
  t=ImageOps.contain(im.convert('RGB'),(850,650));panel=Image.new('RGB',(900,700),'white');panel.paste(t,((900-t.width)//2,38));ImageDraw.Draw(panel).text((15,10),p.stem,fill='black');thumbs.append(panel)
montage=Image.new('RGB',(1800,700*((len(thumbs)+1)//2)), '#EEEEEE')
for i,t in enumerate(thumbs):montage.paste(t,((i%2)*900,(i//2)*700))
montage.save(HERE/'montage_review.jpg',quality=92)
print(json.dumps({'main_figures':4,'supplementary_figures':4,'input_counts':proof['source_counts'],'all_numeric_identities_passed':True}))
