from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from tubescriber import YouTubeTranscriber
    
MODEL_ID = 'llama3.1'
BASE_URL = 'http://127.0.0.1:11434'
VIDEO_ID = 'zjkBMFhNj_g'

def main():
    video_id = VIDEO_ID
    llm = Ollama(model=MODEL_ID, base_url=BASE_URL)
    
    result = transcribe(video_id)
    
    if result:
        print("Transcription Successful.  Creating a Vector Store...")
        vector_store_retriever = make_vector_store_retriever(result)

        print("Combining docs...")
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        combine_docs_chain = create_stuff_documents_chain(
            llm, retrieval_qa_chat_prompt
        )

        retrieval_chain = create_retrieval_chain(vector_store_retriever, combine_docs_chain) 

        print("Chain complete.  Running Invocations...")
        invocations(retrieval_chain)

    else:
        print("ERROR: Failed to transcribe the video.")

def transcribe(video_id):
    tx = YouTubeTranscriber()
    return tx.transcribe(video_id)

def make_vector_store_retriever(result):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_text(result)
    embed_model = OllamaEmbeddings(model=MODEL_ID,
        base_url=BASE_URL
    )
    return Chroma.from_texts(chunks, embed_model).as_retriever()

def invocations(retrieval_chain):
    response1 = retrieval_chain.invoke({"input": "Create an outline and a summary of this message for someone who wants to know the main points but only has a few minutes to read it."})

    print(f"{response1['answer']}\n")

if __name__ == "__main__":
    main()
