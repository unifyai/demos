from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



def create_retriever():    
    def split_text(text): 
        """Splits the given text into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)

        chunks = text_splitter.split_text(text=text)
        return chunks

    def faiss_vector_storage(chunks):
        """Creates a FAISS vector store from the given text chunks.

        Args:
            text_chunks: A list of text chunks to be vectorized.

        Returns:
            FAISS: A FAISS vector store.
        """
        if "vector_store" not in st.session_state:
        st.session_state.vector_store = None  # or some initial value
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts= chunks, embedding=embeddings)
        return vector_store

    def create_retrieval_chain(retriever, question_answer_chain):
        st.session_state.vector_store = vector_store
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs = {"k": 5}
            )
        return retriever
