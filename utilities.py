## MEAN - TARGET ENCODING
def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):#,missing_correction=True
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


## SAFE MEMORY REDUCTION

def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
    """
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == 'float'
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ['float16', 'float32']

    if na_count <= na_loss_limit:
        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == 'int'):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
            return col_tmp

    # field can't be converted
    return col

##MEMORY ENCODING IMPLEMENTATION ON TRAIN AND SET OOF.
#   for f in tqdm_notebook(Cat):
#       gc.collect()
#       train[f+'_mean_target']=None
#       test[f+'_mean_target']=None
#       for trn_idx, val_idx in folds.split(train.values, train.target.values):
#           gc.collect()
#           trn_f, trn_tgt = train[f].iloc[trn_idx], train.target.iloc[trn_idx]
#           val_f, val_tgt = train[f].iloc[val_idx], train.target.iloc[val_idx]
#           trn_tf, val_tf = target_encode(trn_series=trn_f, 
#                                          tst_series=val_f, 
#                                          target=trn_tgt, 
#                                          min_samples_leaf=10, 
#                                          smoothing=2,
#                                          noise_level=0)
        
#           train.loc[val_idx,f+'_mean_target']=val_tf
#           del trn_f, trn_tgt, val_f, val_tgt, trn_tf, val_tf
#       gc.collect()
#       trn_tf, val_tf = target_encode(trn_series=train[f], 
#                                  tst_series=test[f], 
#                                  target=train.target, 
#                                  min_samples_leaf=10, 
#                                  smoothing=2,
#                                  noise_level=0)
#       test[f+'_mean_target']=val_tf
#       del trn_tf, val_tf

###################################################################################

def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in tqdm_notebook(df.columns):
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)
        
        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')
        
        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if (na_count != new_na_count):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if (n_uniq != new_n_uniq):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df
####################################################################################
#COUNT ENCODING TRAIN AND TEST
train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

#TRAIN 
train[feature].map(train[feature].value_counts(dropna=False))




##########################################################################################
#IMPORTANCE CALCULATOR
class Importance_calculator:
    def __init__(self,X,y,param_list,num_round=1000,metric=roc_auc_score,cv=5,random_state=0,modeltype='tree',scale=False):
        
        self.X=X
        self.y=y
        self.cv=cv
        self.scale=scale
        self.param_list=param_list
        self.metric=metric
        self.num_round=num_round
        self.random_state=random_state
        self.modeltype=modeltype
        
    def scorer(self,y_true,y_pred):
        return(self.metric(y_true,y_pred))
  
    def permutate_column_predict(self,model,valid_x,valid_y):
        perm_pred = []
        np.random.seed(self.random_state)
        for col in tqdm_notebook(valid_x.columns):
            value = valid_x[col].copy()
            valid_x[col] = np.random.permutation(valid_x[col].values)
            perm_pred=perm_pred+[self.scorer(valid_y,self.pred_wrapper(model,valid_x,self.modeltype))] #predict
            valid_x[col] = value
        return(perm_pred)
    
    def pred_wrapper(self,model,x,modeltype='logit'):
        if modeltype is 'logit':
            return(model.predict_proba(x)[:,1])
        else:
            return(model.predict(x))
        
    def scaler(self,train,valid):
        train_mean , train_std = train.mean(axis=0),train.std(axis=0)
        #rescale inside the cycle to not overfit
        train-=train_mean
        valid-=train_mean

        train/=train_std
        valid/=train_std
        return(train,valid)
    
    def build_neural(self):
        Input = layers.Input(shape=(self.X.shape[1],))
        x = layer.Dense(self.param_list['number'],activation=self.param_list['activation'])(Input)
        pred=layers.Dense(1,activation='softmax')(x)
        self.NN_model=Model(inputs=Input,outputs=pred)
        
    def cv_score_importance(self):
        N=self.X.shape[1]
        folds = StratifiedKFold(n_splits=self.cv, shuffle=True,random_state=self.random_state)
        print('Inizio train e scoring:\n')
        self.importance_permutation_score=[0]*N
        Error=0
        for trn_idx, val_idx in tqdm_notebook(folds.split(self.X, self.y)):
                train_x, train_y = self.X.iloc[trn_idx], self.y.iloc[trn_idx]
                valid_x, valid_y = self.X.iloc[val_idx], self.y.iloc[val_idx]
                if self.scale is True:
                    train_x,valid_x=self.scaler(train_x,valid_x)
                    
                print('Inizio train.\n')
                if self.modeltype is 'logit':
                    model = LogisticRegression(**self.param_list).fit(train_x, train_y)
                if self.modeltype is 'neural':
                    self.build_neural(param= self.param_list)
                    self.NN_model.compile(optimizer='adam',loss='binary_crossentropy')
                    model = self.NN_model.fit(train_x, train_y,epochs=100)
                if self.modeltype is 'tree':
                    model = lgb.train(self.param_list,lgb.Dataset(train_x, label=train_y),self.num_round)
                
                print('inizio calcolo permutation.\n')
                perm_pred = self.permutate_column_predict(model,valid_x,valid_y)
                Pred=self.scorer(valid_y,self.pred_wrapper(model,valid_x,self.modeltype))
                Error+=Pred
                print('AUC-ROC cv : {}\n'.format(Pred))
                base_pred = [Pred] * N
                tmp_diff=[base_pred[i]-perm_pred[i] for i in range(N)]
                self.importance_permutation_score=[self.importance_permutation_score[i]+tmp_diff[i] for i in range(N)]
        print('AUC-ROC cv: {}\n'.format(Error/self.cv))
        return([self.importance_permutation_score[i]/np.float(self.cv) for i in range(N)])
#########################################################################################################

# NEGATIVE SAMPLING 
class Negative_Sampler:
  
  def __init__(self,vector,prob_pos=1,prob_neg=.5,seed=0):
    self.prob_pos=prob_pos
    self.prob_neg=prob_neg
    self.vector=vector
    self.seed=seed
    
  def negative_sample(self):
    np.random.seed(self.seed)
    Positive = np.where(self.vector==1)[0]
    Negative = np.where(self.vector==0)[0]
    Positive_sample= np.random.choice(Positive,np.int(np.round((self.prob_pos)*len(Positive))),replace=False).tolist()
    Negative_sample= np.random.choice(Negative,np.int(np.round((self.prob_neg)*len(Negative))),replace=False).tolist()
    result=np.sort(Positive_sample+Negative_sample)
    return(result)


#############################################################################################################
# CALC BEST CENTROID


from sklearn.model_selection import KFold
from sklearn_extra.cluster import KMedoids
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
import warnings
import gc

class centroid_finder:

    """
    Class used to calculate best number of cluster and the cluster
    """
    
    def __init__(self, range_search = np.arange(2, 8 + 1),
                 model_forecast = LogisticRegression, model_forecast_parameter = {'max_iter': 1000},
                 model_aggregation = KMedoids, model_aggregation_parameter = {'metric': 'euclidean'},
                 score_fn = adjusted_rand_score, score_argument = {}, verbose = False):
        
        self.range_search = range_search
        
        self.model_forecast = model_forecast
        self.model_forecast_parameter = model_forecast_parameter
        
        self.model_aggregation = model_aggregation
        self.model_aggregation_parameter = model_aggregation_parameter
        
        self.score_fn = score_fn
        self.score_argument = score_argument
        
        self.verbose = verbose
        self.metric = model_aggregation_parameter['metric']

    def number_centroid(self, data, n_folder = 5, repeat = 3):
        """
        Calculates the cross validation score with repetition for each number of cluster

        return the best number of cluster
        """
        
        score = []
        for i in self.range_search:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp_score = self.run_cv(data = data, n_cluster = i, n_folder = n_folder, repeat = repeat)
            if self.verbose:
                print(f'# Cluster:  {i}, Score: {temp_score}')

            score += [temp_score]

        self.best_number = self.range_search[np.argmax(score)]

    def run_cv(self, data, n_cluster, n_folder, repeat):
        """
        Given a dataset of embedding, the number of cluster to test, the number of folder and the number of repetition
        It calculates the cv scores to select best number of cluster by calculating adjusted_rand_score.
        http://statweb.stanford.edu/~gwalther/predictionstrength.pdf
        """
        score_f = 0
        for r_s in range(repeat):
            
            kf = KFold(n_splits = n_folder, random_state = r_s, shuffle = True)
            
            score = 0
            for train_index, valid_index in kf.split(data):

                X_train, X_valid = data[train_index,:], data[valid_index,:]

                clustering_model = self.model_aggregation(n_clusters = n_cluster, **self.model_aggregation_parameter )
                clustering_model.fit(X_train)

                y_train = clustering_model.labels_

                #If there is only one cluster then skip and return 0 score
                if len(np.unique(y_train)) == 1:
                    return 0
                
                centroid_train = clustering_model.cluster_centers_

                distance = scipy.spatial.distance.cdist(X_valid, centroid_train, self.metric)
                y_valid = np.argmin(distance, axis = 1)

                model = self.model_forecast(**self.model_forecast_parameter)
                model.fit(X_train, y_train)

                predict = model.predict(X_valid)

                score += self.score_fn(y_valid, predict, **self.score_argument)/n_folder
            
            del X_train, X_valid, clustering_model, y_train, y_valid
            gc.collect()
            score_f += score/repeat
            
        return score

    def run_cluster(self, data):
        """
        After calculation of best number of cluster calculates the cluster

        return the cluster centroid, labels and cluster indices
        """
        
        clustering_model = self.model_aggregation(n_clusters = self.best_number, **self.model_aggregation_parameter)
        clustering_model.fit(data)
        
        self.labels_ = clustering_model.labels_
        self.cluster_centers_ = clustering_model.cluster_centers_
        self.medoid_indices_ = clustering_model.medoid_indices_
