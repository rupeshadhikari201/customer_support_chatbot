import speech_recognition as sr
from flask import Flask, request, render_template, session, jsonify
import os
import soundfile as sf
from agent_customer_support import query_llm
from tts import generate_audio
from datetime import datetime
import uuid
from datetime import datetime
from langchain_cohere import ChatCohere
from langchain_anthropic import ChatAnthropic   
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from agent_customer_support import get_user_detail, get_all_projects, get_project_by_id, get_projects_by_client_id, update_user_profile, get_freelancer_detail, get_project_status, get_user_address, retrieve_company_info, Assistant, create_tool_node_with_fallback
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI

app = Flask(__name__)
app.secret_key = "uuid:1232"
UPLOAD_FOLDER = "static/audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Initialize available LLMs
def get_llm(model_name):
    if model_name == "cohere":
        return ChatCohere(
            cohere_api_key=os.environ['COHERE_API_KEY'],
            temperature=0.5,
            max_tokens=500
        )
    elif model_name == "anthropic":
        return ChatOpenAI(
        openai_api_key = os.environ['OPENROUTER_API_KEY'],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="anthropic/claude-3-haiku:beta",
        temperature= 0.7, 
        )
    elif model_name == "mistral":
        return ChatMistralAI(
            mistral_api_key=os.environ['MISTRAL_API_KEY'],
            temperature=0.5,
            max_tokens=500
        )
    elif model_name == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=os.environ['GOOGLE_API_KEY'],
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,   
            max_retries=2,
        )
    elif model_name == "deepseek":
        return ChatOpenAI(
            openai_api_key = os.environ['OPENROUTER_API_KEY'],
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="deepseek/deepseek-r1-zero:free",
            temperature= 0.7, 
        )
    elif model_name == "llama":
        return ChatOpenAI(
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-2-13b-chat",
        temperature= 0.7, 
)
    else:
        return ChatCohere(
            cohere_api_key=os.environ['COHERE_API_KEY'],
            temperature=0.5,
            max_tokens=500
        )

# Default LLM
current_llm = get_llm("cohere")

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

# llm tools
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

assistant_runnable = primary_assistant_prompt | current_llm.bind_tools(llm_tools)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Create graph executer
def create_executer(llm):

    # Graph builder
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
    return builder.compile(checkpointer=memory)

# Initialize executer with default LLM
executer = create_executer(current_llm)

# Function to query LLM
def query_llm(query=None, model_name="cohere"):
    global current_llm, executer
    
    # Check if model has changed
    if model_name != session.get("current_model", "cohere"):
        current_llm = get_llm(model_name)
        executer = create_executer(current_llm)
        session["current_model"] = model_name
    
    if query is None:
        print("query is required")
        return "Please provide a query."
    
    thread_id = session.get("thread_id", str(uuid.uuid4()))
    session["thread_id"] = thread_id
    
    config = {
        "configurable": {
            "user_id": "1",
            "thread_id": thread_id,
        }
    }
    
    events = executer.stream(
        {"messages": ("user", query)}, config, stream_mode="values"
    )
    
    response = ""
    for event in events:
        response = event
    
    return response.get('messages')[-1].content


@app.route('/', methods=['POST', 'GET'])
def home():
    if "chat_history" not in session:
        session["chat_history"] = [{'sender': 'assistant', 'text': "How can I help you today?"}]
    context = session['chat_history']
    return render_template('index.html', messages=context)


@app.route("/send_message", methods=["POST"])
def send_message():
    print(request.json)
    user_message = request.json.get("message", "")
    selected_model = request.json.get("model", "cohere")
    print("user_message", user_message) 
    print("selected_model", selected_model)
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Call agent Model
    response = query_llm(user_message, selected_model)
    
    # Generate audio response
    audio_data = generate_audio(response)
    audio_url = None
    
    if audio_data is not None:
        sample_rate = 22050
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        audio_filename = f"{unique_id}_{timestamp}.wav"
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        
        # Save audio file
        sf.write(audio_path, audio_data, sample_rate, format="WAV")
        audio_url = f"/static/audio/{audio_filename}"
    
    # Append new conversation turn
    context = session.get("chat_history", [])
    context.append({"text": user_message, "sender": 'user'})
    context.append({"text": response, 'sender': 'assistant', 'audio_url': audio_url})
    session['chat_history'] = context
    
    return jsonify({
        "response": response,
        "audio_url": audio_url,
        "model": selected_model
    })


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = [{'sender': 'assistant', 'text': "How can I help you today?"}]
    session["thread_id"] = str(uuid.uuid4())  
    return jsonify({"status": "success"})


# run flask app
if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True)