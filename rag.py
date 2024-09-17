from PyPDF2 import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)

reader = PdfReader("./doc/attention.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
print("Pass : 0")
segment = splitter.split_text(text)
print("???")
print(segment)
df = pd.DataFrame(segment, columns = ["Text"])

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Pass : 1")

df['Embedding'] = df['Text'].apply(model.encode)

vector = model.encode(df['Text'])

dim = vector.shape[1]


import faiss
index = faiss.IndexFlatL2(dim)
index.add(vector)
print("Pass : 2")



input_query = "What is Attention"
encode_pre = model.encode(input_query)
# encode_pre.shape

#FAISS expects 2d array, so next step we are converting encode_pre to a 2D array
import numpy as np
svec = np.array(encode_pre).reshape(1,-1)

print("Pass : 3")
#We will get euclidean distance and index of the 2 nearest neighbours
distance,pos = index.search(svec,k=2)


print(df.Text.iloc[pos[0]])
print("Pass : 4")