import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

class RetrievalModel:
    def __init__(self, index_name: str = None):
        self.index_name = "bao-tri" if index_name is None else index_name
        token = os.getenv('PINE_CONE_API')
        self.pc = Pinecone(
            api_key=token,  
        )
        self.index = self.pc.Index(self.index_name)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
    def augment_query(self, query_text):
        query_embedding = self.model.encode(query_text).tolist()
        results = self.index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
        context = ""
        for match in results['matches']:
            if match['score'] <0.6: continue 
            context += match["metadata"]["content"] + "\n"
        
            

        # retrieved_chunks = [match["metadata"]["content"] for match in results['matches']]

        # context = "\n".join(retrieved_chunks)
        prompt = f"""Dựa trên thông tin kỹ thuật dưới đây, hãy trả lời câu hỏi sau:

# Thông tin kỹ thuật:
{context}

# Câu hỏi:
{query_text}
"""
        return prompt

def main():
    model = RetrievalModel()
    
    text = "Lỗi rơi dao trong máy CNC"
    prompt = model.augment_query(text)
    print(prompt)
    
if __name__ == "__main__":
    load_dotenv()
    main()

