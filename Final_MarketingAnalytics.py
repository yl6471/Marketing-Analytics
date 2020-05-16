#!/usr/bin/env python
# coding: utf-8

# # Final Project —— Yanqing Li (Net ID: yl6471)

# ## Middle Eastern Video on Demand

# ## PART I. Data Preprocess and Merge

# In[1]:


import matplotlib.pyplot as plt
import scipy.stats as scs
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
subscriber=pd.read_pickle('subscribers')
subscriber.head(10)


# In[2]:


subscriber.shape


# In[3]:


subscriber.columns


# In[4]:


subscriber[ 'weekly_consumption_hour'].unique()


# In[6]:


subscriber.isnull().sum()


# In[7]:


engagement = pd.read_pickle(r'engagement')


# In[8]:


engagement.head()


# In[9]:


customer_service = pd.read_pickle(r'customer_service_reps')


# In[10]:


customer_service_sorted = customer_service.sort_values(by=['account_creation_date'])


# In[11]:


customer_service_sorted


# In[12]:


customer_service_sorted = customer_service_sorted[['subid','current_sub_TF']]


# In[13]:


churn_data = customer_service_sorted.drop_duplicates(keep = 'last')


# In[14]:


churn_data 


# In[15]:


merged_sub = subscriber.merge(churn_data, how = 'left',left_on='subid', right_on='subid')


# In[16]:


merged_sub_churn = merged_sub[['current_sub_TF']]


# left join customer service data onto subscriber data on key "subid". if a user does not have customer service information, we assume he/she will not churn by default.

# In[17]:


merged_sub[['current_sub_TF']] = merged_sub_churn.fillna(True)


# In[18]:


merged_sub


# In[19]:


merged_sub[['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','revenue_net','join_fee']] = merged_sub[['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','revenue_net','join_fee']].fillna(0)


# In[20]:


merged_sub[['package_type','preferred_genre','intended_use','male_TF','attribution_survey','payment_type']] = merged_sub[['package_type','preferred_genre','intended_use','male_TF','attribution_survey','payment_type']].fillna('Unknown')


# In[21]:


merged_sub = merged_sub[merged_sub['age']<=120] 


# In[22]:


merged_sub[['age']] = merged_sub[['age']].fillna(np.average(merged_sub['age'].dropna()))


# In[23]:


merged_sub = merged_sub[merged_sub.columns.drop('op_sys')]


# In[24]:


drop_list = ['creation_until_cancel_days','country', 'account_creation_date', 'trial_end_date', 'language'] 
for i in drop_list:
    merged_sub = merged_sub[merged_sub.columns.drop(i)]


# In[25]:


merged_sub_base_uae_14_day_trial = merged_sub.loc[merged_sub['plan_type'] == 'base_uae_14_day_trial']
merged_sub_low_uae_no_trial = merged_sub.loc[merged_sub['plan_type'] == 'low_uae_no_trial']


# In[26]:


merged_sub_base_uae_14_day_trial = merged_sub_base_uae_14_day_trial[['current_sub_TF','plan_type']]
merged_sub_low_uae_no_trial = merged_sub_low_uae_no_trial[['current_sub_TF','plan_type']]


# In[27]:


merge_AB = pd.concat([merged_sub_base_uae_14_day_trial , merged_sub_low_uae_no_trial], axis=0)


# In[28]:


merge_AB= pd.get_dummies(merge_AB, prefix=['plan_type'],drop_first= True)


# In[29]:


merge_AB


# ## PART II. AB Test (Plan type: low uae no trial vs. base uae 14 day trial )

# In[30]:


import imblearn
print(imblearn.__version__)
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
# make them same size group
X_under, y_under = undersample.fit_resample(merge_AB[['plan_type_low_uae_no_trial']],merge_AB[['current_sub_TF']])


# In[31]:


merge_AB_under = pd.concat([X_under, y_under], axis=1)


# In[32]:


merge_AB_under


# In[33]:


import HW1 as ABTesting


# In[34]:


groupA =  merge_AB_under.loc[merge_AB_under['plan_type_low_uae_no_trial'] == 0] #14 day trial
groupB =  merge_AB_under.loc[merge_AB_under['plan_type_low_uae_no_trial'] == 1] 


# In[35]:


A_list = list(groupA['current_sub_TF'])
B_list = list(groupB['current_sub_TF'])


# In[36]:


import scipy
norm = scipy.stats.norm()
ABTesting.t_test(A_list, B_list,0.95)


# B>A. It means that current subscribers with low uae no trail are more than ones with base uae 14 day trail. The difference is statistically significant. 

# ## PART III. Customer Segmentation

# In[37]:


import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy
import seaborn as sns


# In[38]:


engagement['date'] = pd.to_datetime(engagement['date'])
engagement


# In[41]:


feature_list = ['app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']


# In[42]:


customer_average = pd.pivot_table(engagement, values=feature_list, index='subid', aggfunc=np.mean)
customer_average.reset_index(drop=False, inplace=True)
customer_average


# In[43]:


sub = pd.read_pickle(r'subscribers')

sub


# In[44]:


sub_dummy = sub[['package_type', 'preferred_genre', 'male_TF']]

sub_dummy = pd.get_dummies(sub_dummy)

sub_dummy


# In[45]:


sub_seg = pd.merge(sub[['subid']], sub_dummy, left_index=True, right_index = True, how='left')

sub_seg.shape


# In[46]:


sub_seg = pd.merge(sub_seg, customer_average, on= 'subid', how='left')

sub_seg.shape


# In[47]:


sub_seg.dropna(axis=0, inplace=True)

sub_seg.set_index('subid',inplace=True)
sub_seg.shape


# In[48]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def fitting(df):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return 


# In[49]:


fitting(sub_seg)


# In[50]:


kmeans = KMeans(n_clusters = 4, random_state=0).fit(sub_seg)


# In[51]:


a = list(kmeans.cluster_centers_)

seg_result = pd.DataFrame(a, columns=sub_seg.columns)

seg_result


# ## PART IV. Churn Model (Random Forest)

# In[52]:


merged_sub = pd.get_dummies(merged_sub, prefix=['package_type', 'preferred_genre', 'intended_use','male_TF',  'attribution_technical', 'attribution_survey', 'plan_type',  'payment_type'], columns=['package_type', 'preferred_genre', 'intended_use','male_TF',  'attribution_technical', 'attribution_survey', 'plan_type',  'payment_type'])


# In[53]:


merged_sub 


# In[54]:


merged_sub.to_csv('churnmodel.csv',index=False)


# In[55]:


from sklearn.model_selection import train_test_split
X = merged_sub[merged_sub.columns.drop('current_sub_TF')]
X = X[X.columns.drop('subid')]
y = merged_sub['current_sub_TF'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[56]:


from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier()


# In[57]:


clf.fit(X_train,y_train)


# In[58]:


y_pred = clf.predict(X_test)


# In[59]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))


# In[60]:



importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
features = list()
for f in range(X.shape[1]):
    if importances[indices[f]] >= 0.01:
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        features.append(list(X.columns)[indices[f]])
print('Most important features: ',features)


# ## PART V. Expected CLV

# In[61]:


clf.fit(X,y)


# In[62]:


y_pred = clf.predict(X)


# In[63]:


y_pred = y_pred.astype(int)


# In[64]:


revenue = np.array(X['revenue_net'])


# In[65]:


revenue 


# In[66]:


expected_revenue = np.dot(y_pred,revenue)


# In[67]:


expected_revenue


# In[68]:


clv = expected_revenue/len(y_pred)


# In[69]:


clv


# In[70]:


X = merged_sub[merged_sub.columns.drop('current_sub_TF')]
y = merged_sub['current_sub_TF'].ravel()


# In[71]:


df_pred= pd.concat([X, pd.DataFrame(y_pred,columns = ['current_sub'])], axis=1).dropna()


# In[72]:


df_pred_not_churn = df_pred[df_pred['current_sub'] == 1]
df_pred_churn = df_pred[df_pred['current_sub'] == 0]


# In[73]:


df_pred_churn


# In[74]:


df_pred_churn.to_csv('predchurn.csv',index=False)


# In[75]:


df_pred_not_churn


# In[76]:


df_comp= pd.concat([pd.DataFrame(y,columns = ['current_sub']), pd.DataFrame(y_pred,columns = ['predicted_sub'])], axis=1).dropna()


# In[77]:


df_comp= pd.concat([X[['subid']], df_comp], axis=1).dropna()


# In[79]:


y_equal = list()
for i in range(len(y)):
    if y[i] == y_pred[i]:
        y_equal.append(1)
    else:
        y_equal.append(0)
df_comp= pd.concat([df_comp,pd.DataFrame(y_equal,columns = ['equal'])], axis=1).dropna()


# In[80]:


df_comp.to_csv('predComp.csv',index=False)


# In[ ]:




