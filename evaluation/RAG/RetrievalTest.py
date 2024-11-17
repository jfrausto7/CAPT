from langchain_together import Together
from agents.retrieval.MultiVectorstoreRetriever import MultiVectorstoreRetriever


def create_retrieval_chain(
    vector_dir: str,
    model,
    show_progress: bool = True
) -> None:
    # Initialize the retriever
    retriever = MultiVectorstoreRetriever(
        vector_dir=vector_dir,
        k=4,  # Number of documents to retrieve per query
        score_threshold=0.7,  # Minimum similarity score
        show_progress=show_progress
    )
    
    # Create a retrieval chain
    from langchain.chains import RetrievalQA
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True  # Include source docs in response
    )
    
    return qa_chain

# Initialize your model
model = Together(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    temperature=0.3,
    max_tokens=512
)

# Create the retrieval chain
qa_chain = create_retrieval_chain(
    vector_dir="app/vectorstore/store",
    model=model
)

# Query across all collections
# response = qa_chain("What are the safety guidelines for ketamine therapy?")
# answer = response['result']
# sources = response['source_documents']

prompt1 = "How many individuals have received MDMA in phase 1 or phase 2 clinical trials without any unexpected drug-related serious adverse events?" # Over 1100
prompt2 = "How many administrations of psilocybin with psychological support produces antidepressant effects in patients with cancer and in those with treatment-resistant depression?" # 1-2
prompt3 = "In 'Effects of Psilocybin-Assisted Therapy on Major Depressive Disorder: A Randomized Clinical Trial', what was the age range of participants? Where did the study take place?" #  Johns Hopkins University School of Medicine. The age range of participants was 21-75 years.
prompt4 = "What is a substantial public health burden?" # Major Depressive Disorder (MDD)

# Query specific collections
response = qa_chain.invoke(
    prompt3,
    collections=['articles']
)

print(response['result'])