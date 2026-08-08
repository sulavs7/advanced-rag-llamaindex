from llama_index.llms.groq import Groq

def generate_answer(llm: Groq, query: str, context: str) -> str:
    prompt = f"""Answer the question using ONLY the context below. \
    If the answer isn't in the context, say you don't know.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    response = llm.complete(prompt)
    return str(response)