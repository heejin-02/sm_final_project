from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:/Users/smhrd1\Desktop/final_project/langchain/pdf/2025_insect.pdf")
data_nyc = loader.load()
print(data_nyc)