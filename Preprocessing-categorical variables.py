import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# assume we have four features and feature 1 and feature 4 are categorical variables
X=[['good',1,10,'spring'],
   ['median',2,14,'summer'],
   ['bad',5,10,'winter'],
   ['median',10,15,'fall']]
items=['quality','size','length','season']
df=pd.DataFrame(X,columns=items)

#The LabelEncoder can transform the string variables to numerical variables
df_values=df.values
label=LabelEncoder()
df_values[:,0]=label.fit_transform(df_values[:,0])
df_values[:,3]=label.fit_transform(df_values[:,3])

#The OneHotEncoder can transform the numerical variables into dummy variables
#worth to note, if sparse is True, use toarray()
dummy=OneHotEncoder(categorical_features=[0,3],sparse=False)
X_dummy=dummy.fit_transform(df_values)
print(X_dummy)
print(dummy.get_feature_names())