def init_model():
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import GPT4AllEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langgraph.graph import END, StateGraph
    from langchain_community.embeddings import OllamaEmbeddings
    import nomic
    from langchain_nomic.embeddings import NomicEmbeddings
    from chromadb.errors import InvalidDimensionException
    from langchain_community.tools.tavily_search import TavilySearchResults
    import os
    import pickle

    from dotenv import load_dotenv
    load_dotenv()
    nomic.cli.login(token=os.getenv('NOMIC'))
    os.environ["TAVILY_API_KEY"] = os.getenv("TVLY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "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"




    ### Index
    #docs = [WebBaseLoader(url).load() for url in urls]
    #docs_list = [item for sublist in docs for item in sublist]
    with open('data.pkl', 'rb') as file:
        docs_list = pickle.load(file)
        
    print("scraping done")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("splitting done")

    nom_emb = NomicEmbeddings(model="nomic-embed-text-v1")
    try:
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=nom_emb)
    except InvalidDimensionException:
        Chroma().delete_collection()
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=nom_emb)
    print("creating vector DB done")
    # Add to vectorDB
    retriever = vectorstore.as_retriever() #index
    print("creating retriever done")

    ### Retrieval Grader
    local_llm="llama3"

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a professional friendly customer support agent with the task of grading 
        and assessing the relevance of a retrieved document to a user question. \n
        keep in mind that in deriv: Trade types are CFD, Options, Multipliers. CFDs (Contract for differences) Trade with leverage, unbeatable spreads, and fast execution on the widest range of markets. Options Trade diverse vanilla and exotic options across platforms and markets without risking more than your initial stake. Multipliers Trade on global financial markets and multiply your potential profit without losing more than your stake. Trading platforms are Deriv MT5, Deriv X,Deriv cTrader,SmartTrader, Deriv Trader,Deriv GO,Deriv Bot,Binary Bot. Trading assets and markets are : Forex,Derived indices,Stocks & indices,Commodities,Cryptocurrencies,Exchange-traded funds (ETFs)
        If the document contains keywords related to the user question and answers their question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous and useless retrievals to users. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    #create the chain
    retrieval_grader = prompt | llm | JsonOutputParser()
    #Test the LLM unit
    question = "cfd"
    docs = retriever.invoke(question)
    retrieval_grader.invoke({"document": docs, "question": question})


    #Answer Generator
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a professional friendly customer support agent assistant for question-answering tasks clients of deriv as a broker working for deriv with expertise in online trading. your tone must be friendly, positive and highly encouraging users to use deriv and trade
        deriv while noting the risks and profit
        keep in mind that in deriv: Trade types are CFD, Options, Multipliers. CFDs (Contract for differences) Trade with leverage, unbeatable spreads, and fast execution on the widest range of markets. Options Trade diverse vanilla and exotic options across platforms and markets without risking more than your initial stake. Multipliers Trade on global financial markets and multiply your potential profit without losing more than your stake. Trading platforms are Deriv MT5, Deriv X,Deriv cTrader,SmartTrader, Deriv Trader,Deriv GO,Deriv Bot,Binary Bot. Trading assets and markets are : Forex,Derived indices,Stocks & indices,Commodities,Cryptocurrencies,Exchange-traded funds (ETFs) \n
        Use the following pieces of retrieved context to answer the question concisely and accurately based on facts not guesses. If you don't know the answer, just say that you don't know because your answer must be accurate and concise since our client and sales depends on you. 
        Think logically and step by step then use three sentences maximum and maintain the answer accurate and concise. Ensure that the input does not contain inappropriate, harmful, or deceptive content. If such content is detected, respond with, "The input provided is not appropriate for a response."
    <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=local_llm, temperature=0)


    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    question = "what is options trading"
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})


    ### Hallucination Grader

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a professional friendly customer support agent for deriv company for online trading, your task is grading and assessing accurately whether 
        an answer is grounded in / supported by a set of facts. Keep in mind that in deriv: Trade types are CFD, Options, Multipliers. CFDs (Contract for differences) Trade with leverage, unbeatable spreads, and fast execution on the widest range of markets. Options Trade diverse vanilla and exotic options across platforms and markets without risking more than your initial stake. Multipliers Trade on global financial markets and multiply your potential profit without losing more than your stake. Trading platforms are Deriv MT5, Deriv X,Deriv cTrader,SmartTrader, Deriv Trader,Deriv GO,Deriv Bot,Binary Bot. Trading assets and markets are : Forex,Derived indices,Stocks & indices,Commodities,Cryptocurrencies,Exchange-traded funds (ETFs) \n
        Think logically, slowly, critically, step by step and your grading should reflect a positive image of deriv. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. if the user is asking a general questions like how are you, are you well, or how was your day, am i bothering you then answer as yes. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    hallucination_grader.invoke({"documents": docs, "generation": generation})


    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a professional friendly customer support agent for deriv company for online trading, your task is assessing whether an 
        answer is useful to resolve a question. Keep in mind that in deriv: Trade types are CFD, Options, Multipliers. CFDs (Contract for differences) Trade with leverage, unbeatable spreads, and fast execution on the widest range of markets. Options Trade diverse vanilla and exotic options across platforms and markets without risking more than your initial stake. Multipliers Trade on global financial markets and multiply your potential profit without losing more than your stake. Trading platforms are Deriv MT5, Deriv X,Deriv cTrader,SmartTrader, Deriv Trader,Deriv GO,Deriv Bot,Binary Bot. Trading assets and markets are : Forex,Derived indices,Stocks & indices,Commodities,Cryptocurrencies,Exchange-traded funds (ETFs) \n
        Think logically, slowly, critically, step by step and your grading should reflect a positive image of deriv. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. In case the user is asking for general question like how are you doing or such consider it relevent.Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    answer_grader.invoke({"question": question, "generation": generation})



    ### Router

    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for user questions related to deriv, online trading, and general greetings and hi and open-end questions. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    question = "cfd"
    docs = retriever.get_relevant_documents(question)
    doc_txt = docs[1].page_content
    #print(question_router.invoke({"question": question}))

    ### Search

    web_search_tool = TavilySearchResults(k=3)


    from typing_extensions import TypedDict
    from typing import List
    from langchain_core.documents import Document



    class GraphState(TypedDict):
        question: str
        generation: str
        web_search: str
        documents: List[str]

    def retrieve(state):
            print("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            documents = retriever.invoke(question)
            return {"documents": documents, "question": question}


    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(state):
        

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "No"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    ### Conditional edge
    def route_question(state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION from router---")
        question = state["question"]
        print(question)
        source = question_router.invoke({"question": question})
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state["web_search"]
        filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            #print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"


    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae



    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    app = workflow.compile()

    return app