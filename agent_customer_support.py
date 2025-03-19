import os
import sys
from sqlalchemy import create_engine, text, inspect
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
import uuid
import logging

#################### Retriever ################
from rag_pipeline import rag_pipeline

################## Utilities ###########
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

######### Assistant ##############
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from datetime import datetime

######## Graph Executer ##########
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
DB_URL = os.getenv("DB_URL")

engine = create_engine(DB_URL)
inspector = inspect(engine)

table_names = [
    'client_client', 'project_projects', 'payment_paymentstatus', 'project_projectstatus',
    'project_projectfile', 'project_projectsassigned', 'payment_payment', 'freelancer_freelancer',
    'project_applyproject', 'register_user', 'register_address', 'register_freelancer',
    'register_applyproject', 'register_projects', 'register_client', 'register_notification',
    'register_paymentstatus', 'register_payment', 'register_projectfile', 'register_projectstatus',
    'register_projectsassigned'
]

#################### Retriever ################
path = 'doc/'  
loader = DirectoryLoader(
    path,
    glob='*.txt',
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load_and_split(RecursiveCharacterTextSplitter(
    chunk_overlap=200,
    chunk_size=1000,
))

# Creating vectorstore
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vector_db = FAISS.from_documents(documents, embedding)

@tool
def retrieve_company_info(query: str, k: int = 3) -> list:
    """
    Retrieve relevant gokap company related information, including general details, steps, policies, stories, and company history.

    Args:
        query (str): The search query related to gokap company policies, general information, or stories.
        k (int): Number of relevant documents to retrieve (default: 3).

    Returns:
        list: A list of relevant document contents matching the query.
    """
    try:
        # Use the optimized RAG pipeline for retrieval
        retrieved_docs = rag_pipeline.get_retrieval_content(query, k=k)
        
        if not retrieved_docs:
            # If no documents found, try with a more general query
            logger.info(f"No documents found for '{query}'. Trying with a more general query.")
            broader_query = " ".join(query.split()[:3]) if len(query.split()) > 3 else query
            retrieved_docs = rag_pipeline.get_retrieval_content(broader_query, k=k)
            
        if not retrieved_docs:
            return ["No relevant information found for the query."]
            
        # Add source information if available
        enhanced_docs = []
        for i, doc in enumerate(retrieved_docs):
            enhanced_docs.append(f"Document {i+1}:\n{doc}")
            
        return enhanced_docs
        
    except Exception as e:
        logger.error(f"Error retrieving company info: {str(e)}")
        return [f"Error retrieving information: {str(e)}"]

@tool
def get_user_detail(user_id: int) -> str:
    """
    Retrieve user details by user ID.

    Args:
        user_id (int): The ID of the user.

    Returns:
        str: A message with user details or an error message.
    """
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT * FROM register_user WHERE id = {user_id}"))
        user = result.fetchone()

    if user:
        return f"User details: {user}"
    else:
        return f"No user exists with ID {user_id}"

@tool
def get_all_projects() -> list[dict]:
    """
    Retrieve all projects.

    Returns:
        list[dict]: A list of all projects.
    """
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM project_projects"))
        projects = result.mappings().fetchall()

    return [dict(row) for row in projects] if projects else []

@tool
def get_project_by_id(project_id: int) -> str:
    """
    Retrieve project details by project ID.

    Args:
        project_id (int): The ID of the project.

    Returns:
        str: Project details or an error message.
    """
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT * FROM project_projects WHERE id = {project_id}"))
        project = result.fetchone()

    if project:
        return f"Project details: {project}"
    else:
        return f"No project found with ID {project_id}"

@tool
def get_projects_by_client_id(client_id: int) -> list[dict]:
    """
    Retrieve all projects associated with a specific client id.

    Args:
        client_id (int): The ID of the client.

    Returns:
        list[dict]: A list of projects associated with the client.
    """
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT * FROM project_projects WHERE client_id = {client_id}"))
        projects = result.mappings().fetchall()

    return [dict(row) for row in projects] if projects else []

@tool
def update_user_profile(user_id: int, firstname: str = None, lastname: str = None, email: str = None) -> str:
    """
    Update user profile details.

    Args:
        user_id (int): The ID of the user.
        firstname (str, optional): New first name.
        lastname (str, optional): New last name.
        email (str, optional): New email.

    Returns:
        str: Success or failure message.
    """
    updates = []
    if firstname:
        updates.append(f"firstname = '{firstname}'")
    if lastname:
        updates.append(f"lastname = '{lastname}'")
    if email:
        updates.append(f"email = '{email}'")

    if not updates:
        return "No updates provided."

    update_query = f"UPDATE register_user SET {', '.join(updates)} WHERE id = {user_id}"

    with engine.connect() as connection:
        result = connection.execute(text(update_query))
        connection.commit()

    if result.rowcount > 0:
        return "User profile updated successfully."
    else:
        return f"No user found with ID {user_id}"

@tool
def get_freelancer_detail(user_id: int) -> str:
    """
    Retrieve freelancer details by user ID.

    Args:
        user_id (int): The ID of the freelancer.

    Returns:
        str: Freelancer details or an error message.
    """
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT * FROM freelancer_freelancer WHERE user_id = {user_id}"))
        freelancer = result.fetchone()

    if freelancer:
        return f"Freelancer details: {freelancer}"
    else:
        return f"No freelancer found with user ID {user_id}"

@tool
def get_project_status(project_id: int) -> str:
    """
    Retrieve the status of a project.

    Args:
        project_id (int): The ID of the project.

    Returns:
        str: Project status or an error message.
    """
    with engine.connect() as connection:
        result = connection.execute(text(f"""
            SELECT ps.project_status
            FROM project_projectstatus ps
            JOIN project_projects p ON ps.id = p.project_status_id
            WHERE p.id = {project_id}
        """))
        status = result.fetchone()

    if status:
        return f"Project status: {status[0]}"
    else:
        return f"No status found for project ID {project_id}"

@tool
def get_user_address(user_id: int) -> str:
    """
    Retrieve the address of a user.

    Args:
        user_id (int): The ID of the user.

    Returns:
        str: User address or an error message.
    """
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT country, state, city, zip_code FROM register_address WHERE user_id = {user_id}"))
        address = result.fetchone()

    if address:
        return f"User address: Country: {address[0]}, State: {address[1]}, City: {address[2]}, ZIP Code: {address[3]}"
    else:
        return f"No address found for user ID {user_id}"

# Error handling for tools
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Create a tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Print event
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


########## cohere ############
CO_API_KEY = os.environ['COHERE_API_KEY']

cohere_llm = ChatCohere(
    cohere_api_key=CO_API_KEY,
    temperature=0.5,
    max_tokens=500
)

llm = cohere_llm


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for gokap innotech company. "
            " Use the provided tools to search for projects, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

llm_tools = [
    get_user_detail,
    get_all_projects,
    get_project_by_id,
    get_projects_by_client_id,
    update_user_profile,
    get_freelancer_detail,
    get_project_status,
    get_user_address,
    retrieve_company_info,
]


assistant_runnable = primary_assistant_prompt | llm.bind_tools(llm_tools)

######## Graph Executer ##########
builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(llm_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")


memory = MemorySaver()
executer = builder.compile(checkpointer=memory)


###### Main #######
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # user_id to be passed to each tool
        "user_id": "1",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

def query_llm(query=None):
    if query is None:
        print("query: is required")
        return
    _printed = set()
    
    events = executer.stream(
        {"messages": ("user", query)}, config, stream_mode="values"
    )
    response = ""
    for event in events:
        response = event
    return response.get('messages')[-1].content