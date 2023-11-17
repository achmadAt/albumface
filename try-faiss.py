import faiss
import numpy as np
d = 3
index = faiss.IndexFlatIP(d)


#Test add embeddings vector to index
test = np.array([[0.1, 0.2, 0.3]])
index.add(test)

print(index.ntotal)



k = 1
searched = np.array([[0.1,0.2,0.3]])
dist, val = index.search(searched, k=k)

print("dist:", dist, "val:", val)