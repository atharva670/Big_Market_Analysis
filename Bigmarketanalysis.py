#Basic Statistics
import pandas as pd
import numpy as np
import math
# Data cleaning

data = pd.read_csv('train.csv')
df = pd.DataFrame(data)
# mean imputation
df['Item_Weight'] = pd.to_numeric(df['Item_Weight'], errors='coerce')
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
# mode imputation
mod = df['Outlet_Size'].mode().iloc[0]
print("Mode=", mod)
df['Outlet_Size'] = df['Outlet_Size'].apply(
    lambda x: np.nan if (str(x).isdigit() or x == '') else x
)
df['Outlet_Size'] = df['Outlet_Size'].fillna('Medium')
# Checking for duplicate rows
print(df.duplicated().sum())
# Outlier Detection and Removal
columns=df.select_dtypes(include=['int64','float64']).columns
for i in columns:
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    low=Q1-(1.5*IQR)
    high=Q3+(1.5*IQR)
    sum=0
    for j in df[i]:
        if(j<low or j>high):
            sum=sum+1
    print(i,"Outlier=",sum)


df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df.to_csv('train.csv', index=False)
# Descriptive statistics
df1 = df[df['Outlet_Type'] == 'Supermarket Type1']
val = df1['Item_Outlet_Sales'].mean()
print("Average Sales from Supermarket Type1 is ", val, 'INR')
df2 = df[df['Outlet_Type'] == 'Supermarket Type2']
val2 = df2['Item_Outlet_Sales'].mean()
print("Average Sales from Supermarket Type2 is ", val2, 'INR')
df3 = df[df['Outlet_Type'] == 'Supermarket Type3']
val3 = df3['Item_Outlet_Sales'].mean()
print("Average Sales from Supermarket Type3 is ", val3, 'INR')
df4 = df[df['Outlet_Type'] == 'Grocery Store']
val4 = df4['Item_Outlet_Sales'].mean()
print("Average Sales from Grocery Store is ", val4, 'INR')
print("Maximum People  go in Supermarket Type 3")
l = []
for i in df3['Item_MRP']:
    l.append(i)
l.sort()

print(
    "Minimum expense that one person can do in Supermarket Type 3 is ",
    l[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Supermarket Type 3 is ",
    l[len(l) - 1],
    'INR',
)
mod = df['Outlet_Location_Type'].mode().iloc[0]
print("Most of the Outlets are in", mod, "Cities")
print(
    "Difference between minimum expense and maximum expense in Supermarket 3 is",
    l[len(l) - 1] - l[0],
)
# Standard Deviation and Variance is used for machine learning models
l2 = []
for i in l:
    if i < np.median(l):
        l2.append(i)

l3 = []
for i in l:
    if i > np.median(l):
        l3.append(i)
print(
    "25% of products in Supermarket 3  having cost less than or equal to",
    np.median(l2),
)
print(
    "Half of the  products in Supermarket 3 having cost less than or equal to",
    np.median(l),
)
print(
    "75% of products  in Supermarket 3 having cost less than or equal to",
    np.median(l3),
)
print(
    "25% of products in Supermarket 3 having cost greater than", np.median(l3)
)
# SuperMarket 2
l4 = []
for i in df2['Item_MRP']:
    l4.append(i)
l4.sort()

print(
    "Minimum expense that one person can do in Supermarket Type 2 is ",
    l4[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Supermarket Type 2 is ",
    l4[len(l4) - 1],
    'INR',
)
print(
    "Difference between minimum expense and maximum expense in Supermarket 2 is",
    l4[len(l4) - 1] - l4[0],
)
# Standard Deviation and Variance is used for machine learning models
l5 = []
for i in l:
    if i < np.median(l4):
        l5.append(i)

l6 = []
for i in l4:
    if i > np.median(l4):
        l6.append(i)
print(
    "25% of products in Supermarket 2 having cost less than or equal to",
    np.median(l5),
)
print(
    "Half of the  products in Supermarket 2 having cost less than or equal to",
    np.median(l4),
)
print(
    "75% of products  in Supermarket 2 having cost less than or equal to",
    np.median(l6),
)
print(
    "25% of products in Supermarket 2 having cost greater than", np.median(l6)
)
# SuperMarket 1
l7 = []
for i in df1['Item_MRP']:
    l7.append(i)
l7.sort()

print(
    "Minimum expense that one person can do in Supermarket Type 1 is ",
    l7[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Supermarket Type 1 is ",
    l7[len(l7) - 1],
    'INR',
)
print(
    "Difference between minimum expense and maximum expense in Supermarket 1 is",
    l7[len(l7) - 1] - l7[0],
)

l8 = []
for i in l7:
    if i < np.median(l7):
        l8.append(i)

l9 = []
for i in l7:
    if i > np.median(l7):
        l9.append(i)
print(
    "25% of products in Supermarket 1 having cost less than or equal to",
    np.median(l8),
)
print(
    "Half of the  products in Supermarket 1 having cost less than or equal to",
    np.median(l7),
)
print(
    "75% of products  in Supermarket 1 having cost less than or equal to",
    np.median(l9),
)
print(
    "25% of products in Supermarket 1 having cost greater than", np.median(l9)
)
# Grocery Store
l10 = []
for i in df4['Item_MRP']:
    l10.append(i)
l10.sort()
print(
    "Minimum expense that one person can do in Grocery Store is ",
    l10[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Grocery Store is ",
    l10[len(l10) - 1],
    'INR',
)
print(
    "Difference between minimum expense and maximum expense in Grocery Store is",
    l10[len(l10) - 1] - l10[0],
)
# Standard Deviation and Variance is used for machine learning models
l11 = []
for i in l10:
    if i < np.median(l10):
        l11.append(i)

l12 = []
for i in l10:
    if i > np.median(l10):
        l12.append(i)
print(
    "25% of products in Grocery Store  having cost less than or equal to",
    np.median(l11),
)
print(
    "Half of the  products in Grocery Store having cost less than or equal to",
    np.median(l10),
)
print(
    "75% of products  in Grocery Store having cost less than or equal to",
    np.median(l12),
)
print(
    "25% of products in Grocery Store having cost greater than", np.median(l12)
)
print(
    "For  maximum shopping with minimum cost People should go to Supermarket 3"
)
# Standard Deviation and Variance is used for machine learning models
# Inferential statistics
# Hypothesis Testing

print(
    "Let H0=Half of the  products in Supermarket 3 having cost less than or equal to",
    np.median(l),
)
print(
    "Let H1=Half of the  products in Supermarket 3 is not having cost less than or equal to",
    np.median(l),
)
n = 935 / 2
num = int(n)

sum = 0
for i in range(num):

    sum = sum + df3['Item_MRP'].iloc[i]


X = sum / num  # Sample mean
print("Sample mean", X)
Y = df3['Item_MRP'].mean()  # Population mean
print("Population mean", Y)
sum1 = 0
for i in range(len(df3)):
    sum1 = sum1 + (
        (df3['Item_MRP'].iloc[i] - Y) * (df3['Item_MRP'].iloc[i] - Y)
    )
sd = math.sqrt(sum1 / len(df3))
print("Standard Deviation=", sd)
sample_size = len(df3)
print("Sample size=", sample_size)
d = sd / math.sqrt(sample_size)
Z = (X - Y) / d
print("Z-Value=", Z)
print("Since Z-value is near to Zero , p-value is very High")
print("Fail to reject H0")
#Data Visualisation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('train.csv')
df=pd.DataFrame(data)
df1=df[df['Outlet_Type']=='Supermarket Type1']
df2=df[df['Outlet_Type']=='Supermarket Type2']
df3=df[df['Outlet_Type']=='Supermarket Type3']
gs=df[df['Outlet_Type']=='Grocery Store']
#Univariate data Visualization
plt.hist(df1['Item_Outlet_Sales'],color='red',bins=4,label='Sales vs Store')
plt.legend()
plt.grid(True)
plt.show()
plt.hist(df2['Item_Outlet_Sales'],color='red',bins=4,label='Sales vs Store')
plt.legend()
plt.grid(True)
plt.show()
plt.hist(df3['Item_Outlet_Sales'],color='red',bins=4,label='Sales vs Store')
plt.legend()
plt.grid(True)
plt.show()
plt.hist(gs['Item_Outlet_Sales'],color='red',bins=4,label='Sales vs Store')
plt.legend()
plt.grid(True)
plt.show()
sns.countplot(df['Outlet_Type'],color='red')
plt.show()
plt.boxplot(df1['Item_Outlet_Sales'],label='Sales of Supermarket Type1')
plt.legend()
plt.show()
plt.boxplot(df2['Item_Outlet_Sales'],label='Sales of Supermarket Type2')
plt.legend()
plt.show()
plt.boxplot(df3['Item_Outlet_Sales'],label='Sales of Supermarket Type3')
plt.legend()
plt.show()
plt.boxplot(gs['Item_Outlet_Sales'],label='Sales of Grocery Store')
plt.legend()
plt.show()
#Bivariate Data Visualization
plt.plot(df1['Item_Outlet_Sales'],df1['Item_MRP'],color='red',label='Sales vs MRP of Supermarket1')
plt.legend()
plt.show()
plt.plot(df2['Item_Outlet_Sales'],df2['Item_MRP'],color='red',label='Sales vs MRP of Supermarket2')
plt.legend()
plt.show()
plt.plot(df3['Item_Outlet_Sales'],df3['Item_MRP'],color='red',label='Sales vs MRP of Supermarket3')
plt.legend()
plt.show()
plt.plot(gs['Item_Outlet_Sales'],gs['Item_MRP'],color='red',label='Sales vs MRP of Grocery Store')
plt.legend()
plt.show()
plt.scatter(df1['Item_Outlet_Sales'],df1['Item_MRP'],color='red',label='Sales vs MRP of Supermarket1')
plt.legend()
plt.show()
plt.scatter(df2['Item_Outlet_Sales'],df2['Item_MRP'],color='red',label='Sales vs MRP of Supermarket2')
plt.legend()
plt.show()
plt.scatter(df3['Item_Outlet_Sales'],df3['Item_MRP'],color='red',label='Sales vs MRP of Supermarket3')
plt.legend()
plt.show()
plt.scatter(gs['Item_Outlet_Sales'],gs['Item_MRP'],color='red',label='Sales vs MRP of Grocery Store')
plt.legend()
plt.show()
sns.barplot(x=df1['Item_Outlet_Sales'],y=df1['Item_MRP'],label='Sales vs MRP of Supermarket1')
plt.xlim(1000,1005)
plt.ylim(100,105)
plt.legend()
plt.show()
sns.barplot(x=df2['Item_Outlet_Sales'],y=df2['Item_MRP'],label='Sales vs MRP of Supermarket2')
plt.xlim(1000,1005)
plt.ylim(100,105)
plt.legend()
plt.show()
sns.barplot(x=df3['Item_Outlet_Sales'],y=df3['Item_MRP'],label='Sales vs MRP of Supermarket3')
plt.xlim(1000,1005)
plt.ylim(100,105)
plt.legend()
plt.show()
sns.barplot(x=gs['Item_Outlet_Sales'],y=gs['Item_MRP'],label='Sales vs MRP of Grocery Store')
plt.xlim(1000,1005)
plt.ylim(100,105)
plt.legend()
plt.show()
#Multivariate data distribution
l1=df1.select_dtypes(include=['int64','float64']).columns
new_frame1=df1[l1]
corr1=new_frame1.corr()
sns.heatmap(corr1,annot=True)
plt.show()
l2=df2.select_dtypes(include=['int64','float64']).columns
new_frame2=df2[l2]
corr2=new_frame2.corr()
sns.heatmap(corr2,annot=True)
plt.show()
l3=df3.select_dtypes(include=['int64','float64']).columns
new_frame3=df3[l3]
corr3=new_frame3.corr()
sns.heatmap(corr3,annot=True)
plt.show()
l4=gs.select_dtypes(include=['int64','float64']).columns
new_frame4=gs[l4]
corr4=new_frame4.corr()
sns.heatmap(corr4,annot=True)
plt.show()
sns.pairplot(data=df1)
plt.show()
sns.pairplot(data=df2)
plt.show()
sns.pairplot(data=df3)
plt.show()
sns.pairplot(data=gs)
plt.show()
#Categorical Data visualization
sns.countplot(df['Outlet_Type'],color='red')
plt.show()
df['Outlet_Type'].value_counts().plot.pie()
plt.savefig('plot1.png',dpi=300,bbox_inches='tight')
plt.show()
#Supervised Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import math
import numpy as np
# Load data
df = pd.read_csv('train.csv')
#Decision Tree


le_x = LabelEncoder()
le_y = LabelEncoder()


X = le_x.fit_transform(df['Item_Type'])
y = le_y.fit_transform(df['Outlet_Type'])



X = pd.DataFrame(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=0)


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)
Y_test_decoded = le_y.inverse_transform(Y_test)
y_pred_decoded = le_y.inverse_transform(y_pred)



df1 = pd.DataFrame({
    'Actual': Y_test_decoded,
    'Predicted': y_pred_decoded
})

print(df1)
print(confusion_matrix(Y_test,y_pred))

print("Accuracy:", accuracy_score(y_pred, Y_test)*100,'%')
#Logistic regression

le_x1 = LabelEncoder()
le_y1 = LabelEncoder()


X1 = le_x1.fit_transform(df['Item_Type'])
y1 = le_y1.fit_transform(df['Outlet_Type'])



X1=pd.DataFrame(X1)


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, y1, test_size=0.5, random_state=0)


model1 = LogisticRegression()
model1.fit(X1_train, Y1_train)


y_pred1 = model1.predict(X1_test)
Y1_test_decoded = le_y1.inverse_transform(Y1_test)
y1_pred_decoded = le_y1.inverse_transform(y_pred1)



df2 = pd.DataFrame({
    'Actual': Y1_test_decoded,
    'Predicted': y1_pred_decoded
})

print(df2)
print(confusion_matrix(Y1_test,y_pred1))

print("Accuracy:", accuracy_score(y_pred1, Y1_test)*100,'%')
#Support Vector Machines
obj1=LabelEncoder()
obj2=LabelEncoder()
fat=obj1.fit_transform(df['Item_Fat_Content'])
type1=obj2.fit_transform(df['Item_Type'])
fat=pd.DataFrame(fat)
x_train,x_test,y_train,y_test=train_test_split(fat,type1,test_size=0.5,random_state=0)
mod=SVC(kernel='rbf')
mod.fit(x_train,y_train)
predicted=mod.predict(x_test)
new_test=obj2.inverse_transform(predicted)
new_test1=obj2.inverse_transform(y_test)
dr=pd.DataFrame({'Actual':new_test,
                 'Predicted':new_test1
                })
print(dr)
print(confusion_matrix(new_test1,new_test))

print("Accuracy:", accuracy_score(new_test1, new_test)*100,'%')
#Linear Regression

pc=df[df['Outlet_Type']=='Supermarket Type1']
X=pc[['Item_MRP']]
Y=pc['Item_Outlet_Sales']
X5_train,X5_test,Y5_train,Y5_test=train_test_split(X,Y,test_size=0.5,random_state=0)
module=LinearRegression()
module.fit(X5_train,Y5_train)
predc=module.predict(X5_test)
s=pd.DataFrame({'Actual':Y5_test,'Predicted':predc})
print(s)

input2=float(input('Enter MRP='))

l=[]

l.append(input2)

df_2d=[l]
print('Sales=',module.predict(df_2d))
print('MAE=',mean_absolute_error(predc,Y5_test))
print('MSE=',mean_squared_error(predc,Y5_test))
print('RMSE=',math.sqrt(mean_squared_error(predc,Y5_test)))
print('R2 Score=',r2_score(Y5_test,predc))
print(pc)
#Unsupervised Learning
from sklearn.cluster import KMeans,DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
data=pd.read_csv('train.csv')
df=pd.DataFrame(data)
#K-means Clustering
df1=df[['Item_Weight','Item_MRP']].copy()
new_df1=df1.copy()
obj=KMeans(n_clusters=6,random_state=0)
obj.fit(df1)
print(obj.predict(df1))
print(obj.cluster_centers_)
df1['Cluster'] = obj.labels_
print(df1)
plt.scatter(df1['Item_Weight'], df1['Item_MRP'], c=df1['Cluster'])
plt.xlabel("Item_Weight")
plt.ylabel("Item_MRP")
plt.title("Clusters Visualization")
plt.show()
sil_score = silhouette_score(new_df1, obj.labels_)
db_score = davies_bouldin_score(new_df1, obj.labels_)
ch_score = calinski_harabasz_score(new_df1, obj.labels_)
print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)
print("Calinski-Harabasz Index:", ch_score)
#Hierarchical Clustering
df2=df1[['Item_Weight','Item_MRP']].copy()
Z = linkage(df2, method='ward')
labels = fcluster(Z, t=6, criterion='maxclust')
df2['Cluster2'] = labels
print(df2)
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
sil_score = silhouette_score(new_df1, labels)
db_score = davies_bouldin_score(new_df1, labels)
ch_score = calinski_harabasz_score(new_df1, labels)
print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)
print("Calinski-Harabasz Index:", ch_score)
#DBSCAN
df3=df2[['Item_Weight','Item_MRP']].copy()
db = DBSCAN(eps=2.5, min_samples=2)
db.fit_predict(df3)
df3['Cluster3']=db.labels_
print(df3)
plt.scatter(df3['Item_Weight'],df3['Item_MRP'] , c=labels)
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
sil_score = silhouette_score(new_df1, db.labels_)
db_score = davies_bouldin_score(new_df1, db.labels_)
ch_score = calinski_harabasz_score(new_df1, db.labels_)
print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)
print("Calinski-Harabasz Index:", ch_score)
#SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('train.csv')
le_y = LabelEncoder()
le_x = LabelEncoder()
y= le_y.fit_transform(df['Outlet_Type'])
X= le_x.fit_transform(df['Outlet_Size'])
X=pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
Xv=le_x.inverse_transform(X_train_res)
Yv=le_y.inverse_transform(y_train_res)
X1=pd.DataFrame(Xv,columns=['Outlet_Size'])
Y1=pd.DataFrame(Yv,columns=['Outlet_Type'])
df2=pd.concat([X1,Y1],axis=1)
df3=pd.concat([df,df2],axis=1)
print(df3)
#Time Series
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
df = pd.read_excel('AirPassengers.xlsx')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.index.freq='MS'
data = df['#Passengers']   
plt.plot(data)
plt.title("Time Series Data")
plt.show()
train = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
pred = model_fit.forecast(steps=len(test))
plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.show() 
#Deep Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


df = pd.read_csv('train.csv')


df = df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)


df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


df = pd.get_dummies(df, drop_first=True)


X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)


pred = model.predict(X_test)


mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)



print("\n MODEL PERFORMANCE SUMMARY")
print("-" * 40)

print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(" On average, prediction is off by this much sales value")

print(f"\n Root Mean Squared Error (RMSE): {rmse:.2f}")
print(" Penalizes large errors more strongly")

print(f"\n R² Score: {r2:.3f}")

if r2 > 0.8:
    print(" Excellent model (very accurate)")
elif r2 > 0.6:
    print(" Good model (acceptable predictions)")
elif r2 > 0.4:
    print(" Average model (needs improvement)")
else:
    print(" Poor model (needs tuning)")



comparison = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': pred.flatten(),
    'Error': y_test.values - pred.flatten()
})

print("\n SAMPLE PREDICTIONS (First 10 Rows)")
print("-" * 40)
print(comparison.head(10))



avg_actual = np.mean(y_test)
avg_pred = np.mean(pred)

print("\n INSIGHTS")
print("-" * 40)

print(f" Average Actual Sales: {avg_actual:.2f}")
print(f" Average Predicted Sales: {avg_pred:.2f}")

if avg_pred > avg_actual:
    print(" Model is slightly OVER-predicting sales")
else:
    print(" Model is slightly UNDER-predicting sales")







     
















