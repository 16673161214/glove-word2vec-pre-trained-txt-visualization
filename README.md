# glove-word2vec-pre-trained-txt-visualization
* How to use? Just modify the path of the txt.  
```python
import numpy as np
f=open(r'C:\Users\Administrator\Desktop\vectors.txt','r')
weights=[]
word=[]
for line in f:
    values=line.split()
    word.append(values[0])
    weights.append(values[1:])
print()
from sklearn.manifold import TSNE
X_tsne=TSNE(n_components=2,learning_rate=100).fit_transform(weights)
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_tsne[:,0],X_tsne[:,1])
for i in range(len(X_tsne)):
    x=X_tsne[i][0]
    y=X_tsne[i][1]
    plt.text(x,y,word[i],size=16)
plt.show()
```
* The result likes this:  
![wow!](https://github.com/16673161214/glove-word2vec-pre-trained-txt-visualization/blob/master/gloveresult.jpg)
* The following code can classify 3 clusters of the vectors
```python
#聚类
from sklearn.cluster import KMeans

import numpy as np
f=open(r'C:\Users\Administrator\Downloads\vectors.txt','r',encoding='utf-8')
weights=[]
word=[]
for line in f:
    values=line.split()
    word.append(values[0])
    weights.append(values[1:])

from sklearn.manifold import TSNE
X_tsne=TSNE(n_components=2,learning_rate=100).fit_transform(weights)

# 10 clusters
n_clusters = 3
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=1)
y_pred_kmeans = kmeans.fit_predict(X_tsne)

X_tsne0 = []
X_tsne1 = []
X_tsne2 = []
for i in range(len(y_pred_kmeans)):
    if y_pred_kmeans[i]==0:
        X_tsne0.append(list(X_tsne[i]))
    elif y_pred_kmeans[i]==1:
        X_tsne1.append(list(X_tsne[i]))
    elif y_pred_kmeans[i]==2:
        X_tsne2.append(list(X_tsne[i]))
X_tsne0=np.asarray(X_tsne0)
X_tsne1=np.asarray(X_tsne1)
X_tsne2=np.asarray(X_tsne2)

from matplotlib.font_manager import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
print(plt.rcParams)
plt.figure(figsize=(100,80))
plt.scatter(X_tsne0[:,0],X_tsne0[:,1],c = 'r',marker = 'o')
plt.scatter(X_tsne1[:,0],X_tsne1[:,1],c = 'g',marker = 'o')
plt.scatter(X_tsne2[:,0],X_tsne2[:,1],c = 'b',marker = 'o')
for i in range(len(X_tsne)):
    x=X_tsne[i][0]
    y=X_tsne[i][1]
    plt.text(x,y,word[i],size=6)
plt.show()
```
* The result likes this:  
![wow!](https://github.com/16673161214/glove-word2vec-pre-trained-txt-visualization/blob/master/glove_cluster.jpg)
