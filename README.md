# glove-word2vec-pre-trained-txt-visualization
## Glove
* How to use? Just modify the path of the txt.  
```python
#聚类
from sklearn.cluster import KMeans
#word
import numpy as np
from tqdm import tqdm
f=open(r'C:\Users\Administrator\Desktop\glove and w2v\glovewordvectors.txt','r',encoding='utf-8')
weights=[]
word=[]
for line in f:
    values=line.split()
    word.append(values[0])
    weights.append(values[1:])

from sklearn.manifold import TSNE
X_tsne=TSNE(n_components=3,learning_rate=100).fit_transform(weights)

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

plt.figure(figsize=(100,80))
plt.scatter(X_tsne0[:,0],X_tsne0[:,1],c = 'r',marker = 'o')
plt.scatter(X_tsne1[:,0],X_tsne1[:,1],c = 'g',marker = 'o')
plt.scatter(X_tsne2[:,0],X_tsne2[:,1],c = 'b',marker = 'o')
for i in tqdm(range(len(X_tsne))):
    x=X_tsne[i][0]
    y=X_tsne[i][1]
    plt.text(x,y,word[i],size=6)
plt.savefig('./glove_wordclusterupdate.jpg')
plt.show()
```
* The result likes this:  
![wow!](https://github.com/16673161214/glove-word2vec-pre-trained-txt-visualization/blob/master/glove_wordclusterupdate.jpg)
## Word2vec
* The following code can classify 3 clusters of the vectors
```python
#聚类
from sklearn.cluster import KMeans
#word
import numpy as np
from tqdm import tqdm
f=open(r'C:\Users\Administrator\Desktop\glove and w2v\wordmodel.txt','r',encoding='utf-8')
text=[]
for line in f:
    text.append(line)
text=text[1:]

weights=[]
word=[]
for line in text:
    values=line.split()
    word.append(values[0])
    weights.append(values[1:])

from sklearn.manifold import TSNE
X_tsne=TSNE(n_components=3,learning_rate=100).fit_transform(weights)

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
for i in tqdm(range(len(X_tsne))):
    x=X_tsne[i][0]
    y=X_tsne[i][1]
    plt.text(x,y,word[i],size=6)
plt.savefig('./word2vec_clusterupdate.jpg')
plt.show()
```
* The result likes this:  
![wow!](https://github.com/16673161214/glove-word2vec-pre-trained-txt-visualization/blob/master/glove_cluster.jpg)
