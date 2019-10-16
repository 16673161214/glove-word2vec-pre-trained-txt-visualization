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
