import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import calendar
import warnings
import os
from sklearn import preprocessing
from IPython.core.display import display,HTML
import datetime

display(HTML("<style>.container {width:98% !important;}</style>"))
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.style as style
style.available
style.use('seaborn-darkgrid')
plt.show()

class SAMOutputAnalysis(object):

    def __init__(self,model):
        self.model = model
        self.execute()

    def execute(self):
        self.read_output()
        self.get_figure_label()
        self.summarize_data()
        self.create_analysis_folder()
        self.plot_sample_of_data(7)
        self.plot_means_and_std()
        self.plot_histogram_dist()
        self.plot_kde_dist()
        self.plot_shape_diff_from_avg()
        self.plot_sorted_shape()
        self.plot_monthly_diff_from_avg()
        self.plot_bin_percentages()
        print (f'Finished analysis for {self.model.lat},{self.model.lon}.')

    def read_output(self):
        df = pd.read_csv(self.model.outputpath)
        self.df = df[df.columns[1:]]

    def get_figure_label(self):
        self.name = f'lat{np.round(self.model.lat,3)}_lon{np.round(self.model.lon,3)}'

    def summarize_data(self):
        self.df_summary = self.df.describe()
        self.df_summary.T

    @staticmethod
    def autolabel1(ax,rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=20)

    @staticmethod
    def autolabel(ax,rects,sign,i):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height < 0:
                offset = -25
            else:
                offset = 3
            if i == '':
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=20)
            else:
                ax[i].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=20)

    def create_analysis_folder(self):
        self.dirpath = os.path.join(self.model.fpath,'analysis')
        try:
            os.mkdir(self.dirpath)
        except:
            pass

    def plot_sample_of_data(self,days):
        df = self.df[:24*days]
        df.index.name = 'Hour'
        df.columns.name = 'Weather Year'
        fig, ax = plt.subplots(figsize=(30, 12))
        sns.lineplot(data=df,dashes=False,linewidth=3,marker='o',markersize=10,alpha=0.5)
        plt.title(self.name + ' Sample of 1 Week of Hourly Shape Values (Jan 1 - Jan 7)')
        plt.savefig(os.path.join(self.dirpath,'sample_plot_of_data_' + self.name + '.png'))

    def plot_means_and_std(self):
        dfa = self.df_summary.loc[['mean','std'],:].T
        dfa['type'] = 'SAM Model'
        dfa['colors'] = 'b'
        fig, ax = plt.subplots(figsize=(30, 12))
        df = dfa.round(2)
        df['Year'] = df.index.values
        df = df.sort_values('std')
        ymax = max(max(df['std']),max(df['mean']))
        rects1 = plt.bar(df['Year'],df['std'],alpha=0.5,color = df['colors'])
        plt.ylim(0, ymax+0.1)
        ax1 = ax.twinx()
        df.plot(kind='line',x='Year', y='mean',ax=ax1,legend=False,color='r',alpha=0.5,linewidth=8,marker='o',markersize=20)
        plt.ylim(0, ymax+0.1)
        self.autolabel(ax,rects1,'pos','')
        ax.set(xlabel="Year", ylabel="Std Deviation")
        ax1.set(xlabel="Year", ylabel="Mean")
        ax.figure.legend()
        plt.title(self.name + ' Hourly Shape Means and StdDev by Weather Year')
        plt.savefig(os.path.join(self.dirpath,'mean_and_stdev_' + self.name + '.png'))

    def plot_histogram_dist(self):
        self.cols = list(self.df.columns.values)
        fig, ax = plt.subplots(figsize=(30, 15))
        for col in self.cols:
            ax = sns.distplot(self.df[[col]], rug=False,kde=False, label=col, bins=20,
                          hist_kws={"histtype": "step", "linewidth": 4,"alpha": .5})
        # Plot formatting
        plt.legend(loc='upper right',
                   ncol=3, borderaxespad=0.1,fontsize=20)
        plt.title(self.name + ' Hourly Shape Distribution')
        plt.xlabel('Hourly Generation Shape Value',size=24)
        plt.ylabel('Counts per Year',size=24)
        plt.rc('axes', titlesize=20)     # fontsize of the axes title
        plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)
        plt.savefig(os.path.join(self.dirpath,'histogram_distribution_' + self.name + '.png'))

    def plot_kde_dist(self):
        fig, ax = plt.subplots(figsize=(30, 15))
        for col in self.cols:
            ax = sns.distplot(self.df[[col]], rug=False,kde=True, hist=False,label=col,
                          kde_kws={"linewidth": 4,"alpha": .5})
        # Plot formatting
        plt.legend(loc='best',
                   ncol=3, borderaxespad=0.1,fontsize=20)
        plt.title(self.name + ' Hourly Shape Distribution')
        plt.xlabel('Hourly Generation Shape Value',size=24)
        plt.ylabel('Value Counts per Year',size=24)
        plt.rc('axes', titlesize=20)     # fontsize of the axes title
        plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)
#         plt.xlim(0, 1)
        plt.savefig(os.path.join(self.dirpath,'kernel_distribution_' + self.name + '.png'))

    def plot_shape_diff_from_avg(self):
        df_diff = pd.DataFrame()
        avg = self.df.mean(axis=1)
        for col in self.cols:
            df_diff[col] = self.df[col] - avg # positive values are greater than avg, while negative are less than mean!
        df_diff['hour'] = df_diff.index.values
        self.df_diff = df_diff
        df = df_diff[:-1].copy()
        df_diff_grouped = pd.DataFrame()
        cumulative_pos = []
        cumulative_neg = []
        mcols = []
        for col in self.cols:
            mcols.append((col,'Pos'))
            mcols.append((col,'Neg'))
            cumulative_pos.append(df[col][df[col]>0].sum())
            cumulative_neg.append(df[col][df[col]<0].sum())
        df_diff_grouped["pos"] = cumulative_pos
        df_diff_grouped["neg"] = cumulative_neg
        df_diff_grouped["net"] = df_diff_grouped["pos"] + df_diff_grouped["neg"]
        df_diff_grouped.index = self.cols
        df_diff_grouped.index.name = 'Cumulative Total'
        fig, ax = plt.subplots(figsize=(30, 12))
        df = df_diff_grouped.round(2)
        df = df.sort_values('net')
        rect1=plt.bar(df.index, df.pos.values,alpha=0.5,color='g',width=0.9,label='Positive Errors')
        rect2=plt.bar(df.index, df.neg.values,alpha=0.5,color='r',width=0.9,label='Negative Errors')
        rect3=plt.bar(df.index, df.net.values,alpha=0,color='y',width=0.4)
        plt.plot(df.index, df.net.values,color='cyan', linewidth=8,label='Net Errors')
        ax.set(xlabel="Year", ylabel="Cumulative Hourly Shape Difference from Avg")
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Hourly Shape Difference')
        ax.set_title(self.name + ' Cumulative Difference from Average Hourly Shape',fontdict={'fontsize':25})
        ax.legend()
        self.autolabel(ax,rect1,'pos',"")
        self.autolabel(ax,rect2,'neg',"")
        self.autolabel(ax,rect3,'neg',"")
        plt.savefig(os.path.join(self.dirpath,'shape_difference_from_avg_' + self.name + '.png'))

    def plot_sorted_shape(self):
        df = self.df.copy()
        df.columns.name = 'Weather Year'
        fig, ax = plt.subplots(figsize=(30, 12))
        plt.title(self.name + ' Sorted Hourly Shape Values')
        for col in self.cols:
            df = self.df.sort_values(col)
            sns.lineplot(data=df[col].values,dashes=False,linewidth=5,markersize=2,alpha=0.5,ax=ax,label=col)
        plt.savefig(os.path.join(self.dirpath,'sorted_hourly_values_' + self.name + '.png'))

    def plot_monthly_diff_from_avg(self):
        df_diff_monthly = pd.DataFrame()
#         df = self.df_diff_monthly
        dt = (pd.date_range(start=datetime.datetime(2019, 1, 1),
                                     periods=self.df.shape[0],
                                     freq='h'))
        months = [i.month for i in dt]
        self.df_diff['Month'] = months
        mcols = []
        for col in self.cols:
            mcols.append((col,'Pos'))
            mcols.append((col,'Neg'))
            mcols.append((col,'NET'))
            cumulative_neg = (self.df_diff.groupby(['Month'])[col].agg([('neg' , lambda x : x[x < 0].sum())]))
            cumulative_pos = (self.df_diff.groupby(['Month'])[col].agg([('pos' , lambda x : x[x > 0].sum())]))
            df_diff_monthly[col+"_pos"] = cumulative_pos['pos']
            df_diff_monthly[col+"_neg"] = cumulative_neg['neg']
            df_diff_monthly[col+"_net"] = cumulative_pos['pos'] + cumulative_neg['neg']
        df_diff_monthly.columns=pd.MultiIndex.from_tuples(mcols)
        df_diff_monthly.index.name = 'Month'
        self.df_diff_monthly = df_diff_monthly
        fig, ax = plt.subplots(len(self.cols[:]),figsize=(30,len(self.cols)*3))
        fig.suptitle(self.name + ' Monthly Cost Difference from Average Hourly Shape',fontdict={'fontsize':25})
        plt.tight_layout(pad=3)
        for i in range(len(self.cols)):
            df = df_diff_monthly[self.cols[i]].round(0).astype(int)
            labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            pos = df['Pos'].values
            neg = df['Neg'].values
            net = df['NET'].values
            x = np.arange(len(labels))  # the label locations
            width = .9  # the width of the bars
            rects1 = ax[i].bar(x, pos, width, label='Gain',alpha=0.5,color='green')
            rects2 = ax[i].bar(x, neg, width, label='Loss',alpha=0.5,color='red')
            rects3 = ax[i].bar(x, net, width, label='Net',alpha=0,color='blue')
            ax[i].plot(x, net, width, label='Net',alpha=1,color='cyan')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax[i].set_ylabel('$ Cost Difference')
            ax[i].set_title(self.cols[i] + ' - Difference From Average Shape',fontdict={'fontsize':16})
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(labels,fontdict={'fontsize':19})
            ax[i].legend()
        fig.suptitle(self.name + ' Cumulative Monthly Difference from Average Shape',fontdict={'fontsize':20},y=1)
        plt.savefig(os.path.join(self.dirpath,'monthly_shape_difference_from_avg_' + self.name + '..png'))

    def plot_bin_percentages(self):
        ## find percentage of counts in each bind
        BINS=20
        df_bins=pd.DataFrame()
        df_counts = pd.DataFrame()
        for col in self.cols:
            df_bins[col] = pd.cut(self.df[col],BINS) #,labels=[x for x in range(BINS)])
            df_counts[col] = df_bins[col].value_counts().sort_index().values/len(self.df)*100
        self.df_bins = df_bins
        self.df_counts = df_counts.round(1)
        # set width of bar
        barWidth = 0.8
        fig,ax = plt.subplots(figsize =(30, 15))
        # set height of bars
        for year in self.cols:
            ys1 = self.df_counts[f'{year}']
            br1 = np.arange(len(self.df_counts.iloc[:,0]))
            rects1 = ax.step(br1, ys1, label = f'{year}',alpha=0.5)
#         self.autolabel1(ax,rects1)
        # Adding Xticks  and labels
        xs = list(np.sort(self.df_bins.iloc[:,0].unique()))
        xs = [str(i) for i in xs]
        plt.xlabel('Bin Upper Range', fontweight ='bold')
        plt.ylabel('% of value counts', fontweight ='bold')
        plt.xticks([r  for r in range(len(xs))], xs,rotation='vertical')  # + barWidth
        plt.title(f'Histogram of shape value distributons [{BINS} Bins].',fontweight ='bold')
        plt.legend()
        plt.savefig(os.path.join(self.dirpath,'bin_percentages_' + self.name + '.png'))
        # plt.tight_layout()
        # plt.show()
