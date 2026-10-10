"""Portfolio figures for synthetic data; plotting is optional for core models."""
from pathlib import Path
import numpy as np


def create_figures(directory,metrics,summary,group_prediction,test,prediction,cv_records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelcolor':'#334155','text.color':'#172B43','axes.edgecolor':'#CCD5DE',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','svg.fonttype':'none'})
    navy,blue,teal='#193D64','#4083B8','#178982'
    names=['Displacement','Stress']; units=['mm','MPa']
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    fig.suptitle('Predicted scenario means match held-out synthetic responses',x=.075,ha='left',fontsize=16,fontweight='bold')
    for r,ax in enumerate(axes):
        observed=summary['mean'][:,r]; pred=group_prediction['mean'][:,r]
        low=min(observed.min(),pred.min()); high=max(observed.max(),pred.max()); pad=(high-low)*.1
        ax.plot([low-pad,high+pad],[low-pad,high+pad],color='#9BAABB',lw=1.2,ls='--',label='Perfect agreement')
        ax.scatter(observed,pred,c=navy if r==0 else teal,s=45,zorder=3,edgecolors='white',linewidths=.5)
        ax.set(xlabel=f'Observed scenario mean ({units[r]})',ylabel=f'Predicted mean ({units[r]})',
            xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),title=f"{names[r]}  |  R² = {metrics[r]['scenario_mean_r2']:.4f}")
        ax.grid(color='#E7ECF1',lw=.7); ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.text(.075,.025,'SYNTHETIC DATA · 20 held-out scenarios · Separate training, calibration and test groups',fontsize=9,color='#66788A')
    fig.subplots_adjust(left=.075,right=.98,top=.8,bottom=.19,wspace=.3)
    for ext in ('png','svg'): fig.savefig(directory/f'scenario_means.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,7.2))
    fig.suptitle('Model comparison and calibrated prediction intervals',x=.075,ha='left',fontsize=16,fontweight='bold')
    for r in range(2):
        name=metrics[r]['response']
        best=[min(a['cv_scenario_mean_rmse'] for a in cv_records if a['response']==name and a['degree']==degree) for degree in (1,2,3)]
        ax=axes[0,r]
        ax.bar(['Linear','Quadratic','Cubic'],best,color=[blue,teal,navy],width=.55)
        ax.set(title=f'{names[r]} · training-only CV',ylabel=f'Scenario-mean RMSE ({units[r]})')
        ax.grid(axis='y',color='#E7ECF1',lw=.7); ax.set_axisbelow(True)
        for i,v in enumerate(best): ax.annotate(f'{v:.3f}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        ax.set_ylim(0,max(best)*1.3)
        group_mean=group_prediction['mean'][:,r]
        order=np.argsort(group_mean)
        x=np.arange(len(order))
        lower=group_prediction['lower_90'][order,r]; upper=group_prediction['upper_90'][order,r]
        ax=axes[1,r]
        ax.fill_between(x,lower,upper,color=blue,alpha=.17,label='90% prediction interval')
        ax.plot(x,group_mean[order],color=navy,lw=1.8,label='Predicted mean')
        for rank,index in enumerate(order):
            observed=test['y'][test['groups']==summary['groups'][index],r]
            ax.scatter(np.full(len(observed),rank),observed,s=6,color=teal,alpha=.45,rasterized=False)
        ax.set(title=f"{names[r]} · held-out coverage {metrics[r]['coverage_90']:.1%}",xlabel='Test scenarios ordered by predicted mean',ylabel=f'Response ({units[r]})')
        ax.grid(color='#E7ECF1',lw=.7); ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
        if r==0: ax.legend(loc='upper left',fontsize=8,frameon=False)
    fig.text(.075,.015,'SYNTHETIC DATA · CV chooses models on training groups; calibration sets intervals; test groups only report performance.',fontsize=8.5,color='#66788A')
    fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.095,hspace=.58,wspace=.3)
    for ext in ('png','svg'): fig.savefig(directory/f'validation_intervals.{ext}',dpi=180)
    plt.close(fig)
