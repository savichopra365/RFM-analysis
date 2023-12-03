#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt
import pandas as pd


# In[2]:


df= pd.read_csv("C:\\Users\\ACER\\OneDrive - DAV Institute of Engineering and Technology, Jalandhar\\Documents\\Sample - Superstore.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()
# data is clean


# # extracting date and time for recency
# 

# In[6]:


df['Order Date']=pd.to_datetime(df['Order Date'])


# In[7]:


df['newcol1']=df['Order Date'].dt.year


# In[8]:


df['month']=df['Order Date'].dt.month


# In[9]:


df['day']=df['Order Date'].dt.day


# In[10]:


df.head()


# In[11]:


print(df['Order Date'].min())


# In[12]:


print(df['Order Date'].max())


# In[13]:


df.rename(columns={'Order Date': 'ORDER_date','Customer ID':'cust_id', 'Order ID':'order_id'}, inplace=True)


# In[14]:


presnt_date=dt.datetime(2017,12,31)


# In[15]:


df.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


le=LabelEncoder()


# In[18]:


df['cust_id']=le.fit_transform(df['cust_id'])


# In[19]:


df.head()


# In[20]:


df.shape


# In[21]:


rfm = df.groupby('cust_id').agg({
                            'ORDER_date': lambda ORDER_date: (presnt_date - ORDER_date.max()).days,  # Recency
                            'order_id': lambda order_id: order_id.nunique(),  # Frequency
                            'Sales': lambda Sales: Sales.sum()  # Monetary
})

# Rename the columns to recency, frequency, and monetary
rfm.columns = ["recency", "frequenvy" ,"monetary"]
rfm.head(10)


# In[22]:


rfm.shape


# In[23]:


rfm=rfm.sort_values(by='monetary', ascending=False)


# In[24]:


rfm


# # catagorizing our data into of RFM into 4 catgories using THE QUARTILES
# quartiles divide our data into 4 parts 

# In[25]:


quartiles= rfm.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))


# In[26]:


quartiles=quartiles.to_dict()
quartiles


# In[27]:


series = pd.Series(quartiles)


# In[28]:


# converting these quartiles into the dictonary
dictionary=series.to_dict()
print(dictionary)


# In[29]:


rfm.shape


# In[30]:


# writing functions for classification

def RClass(x,p,d):
    if x<= d[p][0.25]:
        return 1
    
    elif x<= d[p][0.50]:
        return 2
    
    elif x<=d[p][0.75]:
        return 3
    else:
        return 4

    
    
    
def FClass(x,p,d):
       if x<= d[p][0.25]:
        return 4
    
       elif x<= d[p][0.50]:
            return 3
    
       elif x<=d[p][0.75]:
            return 2
       else:
            return 1

    
      


# In[31]:


rmf_segm=rfm


# In[32]:


rfm


# In[33]:


rfmSeg = rfm
rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
rfmSeg['F_Quartile'] = rfmSeg['frequenvy'].apply(FClass, args=('frequenvy',quartiles,))
rfmSeg['M_Quartile'] = rfmSeg['monetary'].apply(FClass, args=('monetary',quartiles,))


# In[34]:


rfmSeg['recency']


# In[35]:


rfmSeg.head()


# In[36]:


# conact method for viewing R F M scores tOGETGER
rfmSeg['RFM_class']=rfmSeg.R_Quartile.map(str) \
                    +rfmSeg.F_Quartile.map(str) \
                    +rfmSeg.M_Quartile.map(str)


# .map(str) is used to convert these values to strings. This step is necessary because you are going to concatenate these values with other strings, and all components being concatenated should be of the same data type (string in this case).
# +:
# 
# The + operator is used for string concatenation in Python.
# 
# 
# \ is a line continuation character used for better readability
# 
# 
# 
# 
# 

# In[37]:


rfmSeg


# In[38]:


rfmSeg['score']=rfmSeg[['R_Quartile','F_Quartile','M_Quartile']].sum(axis=1)


# In[39]:


import seaborn as sns
sns.distplot(rfmSeg['recency'])


# In[40]:


sns.distplot(rfmSeg['frequenvy'])


# In[41]:


sns.distplot(rfmSeg['monetary'])


# In[42]:


from sklearn.preprocessing import  PowerTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np


# In[43]:


pt=FunctionTransformer(func=np.log1p)


# In[44]:


X=rfmSeg['recency']
Y=rfmSeg['frequenvy']
Z=rfmSeg['monetary']


# In[45]:


x_pt=pt.fit_transform(X)
y_pt=pt.fit_transform(Y)
z_pt=pt.fit_transform(Z)


# In[46]:


sns.distplot(x_pt)


# In[47]:


sns.distplot(y_pt)


# In[48]:


sns.distplot(z_pt)


# hence by using log tranform we normalized the the skewed data 
# the next step before k means is to standardize the data

# rfmSeg

# In[49]:


rfmTable = rfmSeg.drop(['R_Quartile', 'F_Quartile', 'M_Quartile', 'RFM_class', 'score'], axis="columns")


# In[50]:


segment_values=['low-val','mid-val', 'high-val']
rfmSeg['value segment']=pd.qcut(rfmSeg['score'], q=3, labels=segment_values)


# # explanation of pd.qcut d.qcut, it specifically divides a continuous variable into quantiles, ensuring that each quantile contains approximately the same number of data points.
# x: This is the input data that you want to discretize. It can be a Pandas Series, NumPy array, or any iterable containing continuous data.
# 
# q: This parameter specifies the number of quantiles (intervals or bins) into which you want to divide the data. For example, if you set q=4, it will create quartiles, dividing the data into four equal parts. If you set q=10, it will create deciles, dividing the data into ten equal parts.
# 
# labels: This parameter allows you to specify labels for the resulting quantiles or bins. You can provide a list of labels with a length equal to q - 1. These labels will be assigned to each quantile in order. If you don't provide labels, the quantiles will be labeled with integers.
# 
# retbins: When set to True, this parameter returns the quantile bin edges along with the discretized data.
# 
# duplicates: By default, pd.qcut will raise an error if there are duplicate values in the input data. You can set duplicates='drop' to drop the duplicates, or duplicates='raise' to raise an error.

# In[51]:


rfmSeg["value segment"].value_counts()


# In[52]:


rfmSeg['score'].max()
rfmSeg['score'].min()


# In[53]:


segment_counts=rfmSeg['value segment'].value_counts().reset_index()
segment_counts.columns=['value segment', 'count']


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


# plotting the bar graph

blue_colors = ["#0074D9", "#3498DB", "#89C4F4", "#6C7A89", "#96A7A2"]

plt.bar(segment_counts['value segment'], segment_counts['count'], color=blue_colors)

# Add labels and a title
plt.xlabel('Value Segment')
plt.ylabel('Count')
plt.title('RFM Value Segment Distribution')

# Show the bar chart
plt.show()


# In[56]:


rfmSeg['RFMcustomerSegments'] = ''

# Define score limits for evenly distributed segments
champions_limit = 12
potential_loyalists_limit = 8
at_risk_customers_limit = 6
cant_loose_limit = 4

rfmSeg.loc[rfmSeg['score'] == champions_limit, 'RFMcustomerSegments'] = 'champions'
rfmSeg.loc[(rfmSeg['score'] >= potential_loyalists_limit) & (rfmSeg['score'] < champions_limit), 'RFMcustomerSegments'] = 'potential loyalists'
rfmSeg.loc[(rfmSeg['score'] >= at_risk_customers_limit) & (rfmSeg['score'] < potential_loyalists_limit), 'RFMcustomerSegments'] = 'at risk customers'
rfmSeg.loc[(rfmSeg['score'] >= cant_loose_limit) & (rfmSeg['score'] < at_risk_customers_limit), 'RFMcustomerSegments'] = 'cant lose'
rfmSeg.loc[rfmSeg['score'] < cant_loose_limit, 'RFMcustomerSegments'] = 'lost'


# In[57]:


rfmSeg['RFMcustomerSegments'].value_counts()


# In[58]:


rfmSeg.head()


# In[59]:


rfmSeg.head()


# In[60]:


segment_product_counts = rfmSeg.groupby(['value segment', 'RFMcustomerSegments']).size().reset_index(name='Count')


# In[61]:


pip install plotly


# In[62]:


import plotly.express as px
import plotly.graph_objs as go


#  Plotly Express to create a treemap     visualization. A treemap is a type of chart that displays hierarchical data as nested rectangles. It's commonly used to represent data with a tree-like structure, where each rectangle (or "tile") represents a node in the hierarchy, and the size or color of the rectangles can represent some quantitative or categorical data.

# In[63]:


fig_treemap= px.treemap(segment_product_counts,
                        path=['value segment','RFMcustomerSegments'],
                         values='Count',
                         color='value segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                         title='RFM CUTSOMERS ANALYSIS')
fig_treemap.show()


# # Analysing the corelation between recency freuency and mometary within each clas

# In[64]:


champions_seg=rfmSeg[rfmSeg['RFMcustomerSegments']=='champions']
pot_loyalist=rfmSeg[rfmSeg['RFMcustomerSegments']=='potential loyalists']
lost=rfmSeg[rfmSeg['RFMcustomerSegments']=='lost']



# In[65]:


corr_matrix_champ=champions_seg[['recency','frequenvy','monetary']].corr()
heatmap_champ=go.Figure(data=go.Heatmap(
                    z=corr_matrix_champ.values,
                    x=corr_matrix_champ.columns,
                    y=corr_matrix_champ.columns,
                    colorscale="RdBu",
                    ))
heatmap_champ.update_layout(title="Champions segment")
heatmap_champ.show()


# # THE segment of champions 
# as expected show a relation between frequency and monetary
# corelation coeff of 0.45 which shows that customers of this segment shop frequent and spend money heavily
# means that, in general, as customers spend more (monetary value increases), they tend to have been less recently active as customers. In other words, customers who spent more are less recent in their activity.

# In[66]:


corr_matrix_loyal=pot_loyalist[['recency','frequenvy','monetary']].corr()
heatmap_loyal=go.Figure(data=go.Heatmap(
                    z=corr_matrix_loyal.values,
                    x=corr_matrix_loyal.columns,
                    y=corr_matrix_loyal.columns,
                    colorscale="RdBu",
                    ))
heatmap_loyal.update_layout(title="potenial loyalist")

heatmap_loyal.show()


# # the segment of potential loyalists
# IT shows that the segment of potential loaylists 
# There is no direct corelation between the 3 variables
# recency and monetary do show a corelation of 0.25 but there is no relationship established between them as the corelation coefficient is low.The variables have relationship that is non linear.

# In[67]:


corr_matrix_lost=lost[['recency','frequenvy','monetary']].corr()
heatmap_lost=go.Figure(data=go.Heatmap(
                    z=corr_matrix_lost.values,
                    x=corr_matrix_lost.columns,
                    y=corr_matrix_lost.columns,
                    colorscale="RdBu",
                    ))
heatmap_lost.update_layout(title="lost customers")

heatmap_lost.show()


# # the segment of lost customers
# 
# In summary, a correlation coefficient of 0.185 suggests a weak positive relationship between frequency and monetary variables. 
# Strong corelation does not exists and the heatmap clearly shows that the customers neither are active nor spend money frequently. Hence we consider this catahgory as the lost customer catagory.

# In[68]:


segment_counts=rfmSeg['RFMcustomerSegments'].value_counts()
blue_colors = ["#0074D9", "#3498DB", "#89C4F4", "#6C7A89", "#96A7A2"]

fig= go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                           marker=dict(color=blue_colors))])
fig.show()


# use plotly it is more user interactivity over matplotlib 

# In[69]:


segment_scores=rfmSeg.groupby('RFMcustomerSegments')['R_Quartile','F_Quartile','M_Quartile'].mean().reset_index()
fig=go.Figure()

fig.add_trace(go.Bar(
  x=segment_scores['RFMcustomerSegments'],
  y=segment_scores['R_Quartile'],
  name="RECENCY",
  marker_color='rgb(94,158,217)'
))


fig.add_trace(go.Bar(
  x=segment_scores['RFMcustomerSegments'],
  y=segment_scores['F_Quartile'],
  name="frequenvy",
  marker_color='rgb(94,158,217)'

))

fig.add_trace(go.Bar(
  x=segment_scores['RFMcustomerSegments'],
  y=segment_scores['M_Quartile'],
  name="monetary",
  marker_color='rgb(32,102,148)'
))



fig.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True
)

fig.show()


# In[ ]:





# In[ ]:




