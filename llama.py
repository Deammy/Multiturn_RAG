from together import Together
from PyPDF2 import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd


# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (

    Settings
)
# from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.core.memory import ChatMemoryBuffer
# from unicodedata import normalize

# from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import numpy as np
d = 1536  # dimension
# faiss_index = faiss.IndexFlatL2(d)
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )

import os
import json
from dotenv import load_dotenv


load_dotenv()  

# uri = os.environ.get("MONGO_DB")
# Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))

# def connect_mongo(uri):
#     try:
#         client = MongoClient(uri, server_api=ServerApi('1'))
#         print("Connect Success")
#         return client
#     except:
#         print("Connection Error")
#         return None

# connect_mongo(uri)
model = SentenceTransformer('all-MiniLM-L6-v2')

"""
    class : MultiturnRAG
        init :
            document_path : Path to document.
            history : Keep chat history with list
            context_length : Determine length of context that query from retriever.

        def initial_retriever:
            initialize retriever for retriever document.

        def prepare_context:
            get context from retriever

        def update_history:
            store history chat to self.history

        def generate_response:
            get response from llm model.

        def get_response:
            get user input and prepare data, then get response from LLM.
"""



class MultiturnRAG:
    def __init__(self,model, client, document_path: list, context_length : int = 20):
        self.document_path = document_path
        self.history = []
        self.context_length = context_length
        self.model = model
        self.client = client
        self.context = ""
        

    def initial_retriever(self):

        """Initila Retriever

        input: -      
        output: -
    
        """

        splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)
        
        text = ""
        all_segment = []
        for path in self.document_path:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            segment = splitter.split_text(text)
            for seg in segment:
                all_segment.append(seg)

        df = pd.DataFrame(all_segment, columns = ["Text"])
        print("Pass : 1")
        df['Embedding'] = df['Text'].apply(model.encode)
        vector = model.encode(df['Text'])
        dim = vector.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vector)
        self.retriever = index
        self.df = df


    def prepare_context(self, query):

        """Prepare Context

        input:
        query : input text for query document in retriever.
        
        output:
        context : context from document that relate to query.
    
        """

        encode_pre = model.encode(query)
        svec = np.array(encode_pre).reshape(1,-1)
        print("Pass : 3")
        distance,pos = self.retriever.search(svec,k=2)

        return self.df.Text.iloc[pos[0]]
    
    def update_history(self, user_input, bot_response):

        """Update History

        input:
        user_input : input text from user.
        sys_response : response from llm.
        
        """
        

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": bot_response})
        if(len(self.history) == 6):
            print("Rechat : ")
            chat = "\n".join([str(history) for history in self.history])
            qa_prompt_str = f"""
                Context information is below.
                ---------------------
                {self.context}
                ---------------------
                Conclude this conversation
                {chat}
            """

            
            message = [{"role": "system", "content": "You are a helpful assistant. You must conclude this conversation.", "context" : self.context}]
            for history in self.history:
                message.append(history)
            message.append({"role": "user", "content": qa_prompt_str})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                max_tokens=128,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["[/INST]","</s>"],
            )
            print(dict(dict(dict(response)['choices'][0])['message'])['content'].split("im_end")[0].replace("im_start", "").replace("assistant","").replace("<","").replace(">","").replace("|",""))
            self.context = dict(dict(dict(response)['choices'][0])['message'])['content'].split("im_end")[0].replace("im_start", "").replace("assistant","").replace("<","").replace(">","").replace("|","")
            self.history = []

        
    def generate_response(self, input, query_context):

        """Generate response
        
        input :
        input : text that will send to llm.
        context : context document from retriever.
        
        output
        response : response from llm.
        """

        input = input.replace("\b", "")

        print(self.context)
        if(self.context == ""):
            message = [{"role": "system", "content": "You are a helpful assistant."}] #You communicate in Thai language.
        else:
            message = [{"role": "system", "content": f"You are a helpful assistant. The context is {self.context}"}]

        for history in self.history:
            message.append(history)

        message.append({"role": "user", "content": input, "context" : query_context})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=64,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["[/INST]","</s>"],
        )
        print("Get response")

        return dict(dict(dict(response)['choices'][0])['message'])['content'].split("im_end")[0].replace("im_start", "").replace("assistant","").replace("<","").replace(">","").replace("|","")
            
    def get_response(self, user_input):

        """Get response
        input : 
        user_input : input from user.

        output :
        response : responser from model.

        """
        try:
            response = self.generate_response(input = user_input)
        except:
            return "Error"
        
        self.update_history(user_input = user_input,bot_response = response)
        
        # response = self.retriever.chat(user_input)
        return response

if __name__ == "__main__":
    document = ["./doc/attention.pdf"]
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    #Use for get api with llm
    client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
    # Start RAG OOP
    rag = MultiturnRAG(document_path=document, model=model, client=client)
    rag.initial_retriever()
    print("Input : ")
    Input = str(input())
    print(rag.prepare_context(Input))
    # Input = " "
    # print("Start : \n")
    # while(Input != "Exit"):
    #     Input = str(input())
    #     response = rag.get_response(Input)
    #     print(response)








