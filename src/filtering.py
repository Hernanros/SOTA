import pandas as pd

from nltk.tokenize import word_tokenize


class filter_annotators ( ):
    def __init__(labelers_col = 'annotator', random_col = 'random', duration_col = 'total_seconds',
                label_col = 'labels', reduced_col = 'reduced_label',
                time_method = 'seconds', time_limit = 300,
                sentences_id = 'pair_id', min_labels = 4, unpopular_share = 0.5):
        
        self.df = None
        self.labelers = None
        
        self.labelers_col = labelers_col
        self.random_col = random_col
        self.duration_col = duration_col
        self.label_col = label_col
        self.reduced_col = reduced_col
        self.time_method = time_method
        self.time_limit = time_limit
        self.sentences_id = sentences_id
        self.min_labels = min_labels
        self.unpopular_share = unpopular_share
        
        
        self.ba = {'duration': None,
                    'high_random': None}
                    'no_variance':None,
                    'unpopular': None,
                    'sentiment_inconsistent': None,
                    'ba_combined': None}

    def fit (df):
        labelers = None

        self.df = df
        if self.reduced_col is None:
            self.df['reduced_label'] = self.df[self.label_col].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0) 
            self.reduced_col = 'reduced_label'
       
        # set up random honey pot and no varience filters       
        if self.random_col is not None:
            
            self.ba['high_random'] = self.random_honey_pot()
            self.ba['no_variance'] = self.no_variance()

        # set up time outliers filter
        if self.duration_col is not None:     
            self.ba['duration'] = self.time_outliers (method = self.time_method, limit = self.time_limit)
            
        #set up unpopular vote
        if self.sentences_id is not None:
            self.ba['unpopular'] = self.unpopular_voter(self.min_labels,self.unpopular_share)

        if labelers is not None:
            self.labelers = labelers



    def time_outliers(method = 'seconds', limit = 300):
        '''
        filter annotators based on the time it took to label a pair of sentences
        argument in:
            - duration series {series, array or list} - time it took to annotate
            - method {str} - weather to filter based on a cap value or cap percentile default is seconds (options: seconds, percentile)
            - limit {int, float or None} - if number: the ceiling value to cap the duartion (default 300), \
                                                    if method = None does it on a percentile based value.
        Returns:
            time outliers list (index)
        
        '''
        df.groupby(labelers_col).mean()
            if self.labelers is not None:
                self.labelers = self.labelers.join(pd.DataFrame(self.df.groupby(self.labelers_col).mean(),columns = [self.duration_col] ))           
            else:
                self.labelers = pd.DataFrame(self.df.groupby(self.labelers_col).mean())
        
        if method =='seconds':
            return list(self.labelers[self.labelers[self.duration_col]>limit].index.values)
        else:
            return list(self.labelers[self.labelers[self.duration_col] > self.labelers[self.duration_col].quantile(limit)].index.values)

    def random_honey_pot ():
        '''
        identify which labelers have a higher mean for random pairs than non-random pairs
        Returns:
            list of suspicious labelers
        '''
        labelers = self.df[self.df[self.random_col]==0].groupby(self.labelers_col)[self.label_col].agg(['mean','std','size'])
        labelers = labelers[labelers['size']>1]
        labelers_rand = self.df[self.df[self.random_col]==1].groupby(self.labelers_col)[self.label_col].agg(['size','mean','std'])
        labelers_rand = labelers_rand[labelers_rand['size']>1]
        
        labelers = labelers.join(labelers_rand, rsuffix = '_rand')
        labelers['mean_random_gap'] = labelers['mean']-labelers['mean_rand']
        if self.labelers is not None:
            self.labelers = self.labelers.join(labelers, rsuffix = '_labels')
        else:
            self.labelers = labelers
        
        return list(self.labelers[self.labelers['mean_random_gap']<0])

    def no_variance (min_var = 1):
        '''
        Identify which labelers doesn't vary enough in their labels
        Args:
            min_var {float or int} - minimal standard deviation value to compare to.
        return:
            list of suspicious labeles
        '''
        total_std = self.df.groupby(self.[self.labelers_col][self.label_col].std()
        total_std.name = 'total_std'
        if self.labelers is not None:
            labelers = labelers.join(total_std)
        else:
            self.labelers = labelers
        return list(self.labelers[self.labelers['total_std']<0])

    def unpopular_voter(min_labels = 4 , unpopular_share = 0.5):
        '''
        Identify which labelers tend to be on the unpopular opinion
        Args:
            min_labels {int} - minimal number of labels for labeler
            unpopular share {float} - threshold share of labels that were on the unpopular size
        Returns:
            list of suspicious labeles
        '''

        # count number of different answers for each sentence pair
        uniquelabels = self.df.groupby(self.sentences_id)[self.reduced_col].nunique()
        # reduce dataset to only have 2 unique answers
        pairs_two_agree = uniquelabels[uniquelabels==2].index.values
        df_twoagree = self.df[self.df[self.sentences_id].isin(pairs_twoagree)]

        # set the generally agreed labels as popular vote 
        df_id_reducedlabel = twoagree.groupby(self.sentences_id)[self.reduced_col].median()
        df_twoagree['generally_accepted_label']  = df_id_reducedlabel.values.repeat(3)

        #count for each labeler how many times they were on the unpopular vote
        df_unpopularopinion = df_twoagree[df_twoagree.reduced_label != df_twoagree.generally_accepted_label].groupby(self.labelers_col).size()
        df_unpopularopinion.columns = ['unpopular_opinion_times']

        # count nummber of labels per labeler
        total_opinion = self.df.groupby(self.labelers_col).size()
        total_opinion.colums = ['total_opinion']
        
        #create df with both columns
        total_opinion.join(df_unpopularopinion).fillna(0)
        total_opinion['unpopular_share'] = total_opinion.unpopular_opinion_times/total_opinion.total_opinion
        
        if self.labelers is not None:
            self.labelers = self.labelers.join(total_opinion['unpopular_share'])
        else:
            self.labelers = labelers

        return (list(total_opinion[(total_opinion.total_opinion>min_labels) & (total_opinion.unpopular_share>unpopular_share)]))
