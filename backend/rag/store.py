from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyMuPDFLoader("challengers_scraper.pdf")
documents = loader.load()



splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
    
)
splits = splitter.split_documents(documents)

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db=FAISS.from_documents(splits, model)

db.save_local("faiss_index")

# db = FAISS.load_local(
#     "faiss_index",
#     model,
#     allow_dangerous_deserialization=True
# )

# query = input("Ask: club name ? ")

# results = db.similarity_search(query, k=3)

# for i, doc in enumerate(results):
#     print(f"\n--- Result {i+1} ---\n")
#     print(doc.page_content)
#     print("Page:", doc.metadata.get("page"))




print("Pages:", len(documents))
print("Chunks:", len(splits))


print("\nSample chunk:\n")
for i, chunk in enumerate(splits):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)
