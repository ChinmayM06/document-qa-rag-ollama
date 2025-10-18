def get_qa_prompt_template() -> str:
    """
    Return the prompt template for question answering.
    Uses context from retrieved documents to answer questions.
    """
    template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 

    If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

    Keep your answer concise and relevant to the question.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    return template


def get_qa_prompt_with_sources_template() -> str:
    """
    Return the prompt template for question answering with source citation.
    """
    template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

    If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

    Keep your answer concise and relevant to the question. If applicable, mention which part of the context supports your answer.

    Context:
    {context}

    Question: {question}

    Detailed Answer:"""
    
    return template


def get_standalone_question_template() -> str:
    """
    Template for converting follow-up questions into standalone questions.
    Useful for chat history context.
    """
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}

    Follow-up question: {question}

    Standalone question:"""
    
    return template


def format_context(documents: list, include_metadata: bool = False) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        documents: List of (Document, score) tuples
        include_metadata: Whether to include document metadata
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(documents, 1):
        if include_metadata and doc.metadata:
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}\n")
        else:
            context_parts.append(f"[Context {i}]\n{doc.page_content}\n")
    
    return "\n".join(context_parts)


# Test function
if __name__ == "__main__":
    from langchain.schema import Document
    
    # Test prompt templates
    print("=== QA Prompt Template ===")
    template = get_qa_prompt_template()
    print(template)
    
    print("\n=== QA with Sources Template ===")
    template_sources = get_qa_prompt_with_sources_template()
    print(template_sources)
    
    # Test context formatting
    print("\n=== Context Formatting Test ===")
    sample_docs = [
        (Document(page_content="Python is a programming language.", 
                 metadata={"source": "doc1.pdf"}), 0.85),
        (Document(page_content="It is widely used in data science.", 
                 metadata={"source": "doc1.pdf"}), 0.72)
    ]
    
    context = format_context(sample_docs, include_metadata=True)
    print(context)