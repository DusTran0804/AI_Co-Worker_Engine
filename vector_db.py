from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

# Sample simulated knowledge base for Gucci Simulation
KNOWLEDGE_BASE = [
    Document(page_content="Gucci Group DNA revolves around balancing creative autonomy for each of the 9 brands while driving group-wide growth.", metadata={"source": "gucci_dna.txt"}),
    Document(page_content="The Gucci Competency Framework consists of 4 core themes: Vision, Entrepreneurship, Passion, and Trust.", metadata={"source": "hr_framework.txt"}),
    Document(page_content="Any group-wide HR policy must be implemented as a flexible framework rather than a rigid, standardized mandate to respect brand autonomy.", metadata={"source": "implementation_guidelines.txt"}),
    Document(page_content="Confidential Financial Metric: Gucci Group's targeted YoY growth for the next quarter is strictly confidential and protected by NDA.", metadata={"source": "nda.txt"})
]

class GucciKnowledgeBase:
    def __init__(self):
        # Requires OPENAI_API_KEY. If mock_key is present, we won't actually call the embedding API
        # to save costs and prevent crashes during offline testing.
        self.api_key = os.environ.get("OPENAI_API_KEY", "mock_key")
        self.vector_store = None
        
        # Only initialize real FAISS if a real key is provided (or if we want to mock it)
        if self.api_key != "mock_key":
            try:
                embeddings = OpenAIEmbeddings(api_key=self.api_key)
                self.vector_store = FAISS.from_documents(KNOWLEDGE_BASE, embeddings)
            except Exception as e:
                print(f"Warning: Failed to initialize FAISS VectorDB. {e}")

    def search(self, query: str, k: int = 1) -> str:
        """
        Searches the VectorDB for the query.
        Returns a formatted string of the most relevant results.
        """
        # 1. Mock Behavior (if no real API key)
        if not self.vector_store:
            # Simple keyword matching as a fallback mock
            query_lower = query.lower()
            for doc in KNOWLEDGE_BASE:
                if any(word in doc.page_content.lower() for word in query_lower.split()):
                    return f"[Mocked VectorDB Result] Found in {doc.metadata['source']}: {doc.page_content}"
            return "[Mocked VectorDB Result] No relevant documents found."
            
        # 2. Real FAISS Vector Search
        try:
            results = self.vector_store.similarity_search(query, k=k)
            if not results:
                return "No relevant documents found."
            
            formatted_results = "\n".join([f"Found in {res.metadata['source']}: {res.page_content}" for res in results])
            return formatted_results
        except Exception as e:
            return f"Error during VectorDB search: {str(e)}"

# Example usage
if __name__ == "__main__":
    db = GucciKnowledgeBase()
    result = db.search("What are the 4 themes of the competency framework?")
    print("Search Result:\n", result)
