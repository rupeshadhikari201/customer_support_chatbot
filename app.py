from flask import Flask, request, render_template, session, jsonify
import os
import soundfile as sf
import numpy as np
import logging
import uuid
import tempfile
from datetime import datetime
import json
from werkzeug.utils import secure_filename

# Import custom modules
from agent_customer_support import query_llm as agent_query_llm
from tts import generate_audio
from rag_pipeline import rag_pipeline
from rag_evaluation import rag_evaluator

# Import LLM modules
from langchain_cohere import ChatCohere
from langchain_anthropic import ChatAnthropic   
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

# Import LangGraph components
from langchain_core.prompts import ChatPromptTemplate
from agent_customer_support import get_user_detail, get_all_projects, get_project_by_id, get_projects_by_client_id, update_user_profile, get_freelancer_detail, get_project_status, get_user_address, retrieve_company_info, Assistant, create_tool_node_with_fallback
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "uuid:1232"
UPLOAD_FOLDER = "static/audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create a folder for evaluation results
EVAL_RESULTS_FOLDER = "evaluation_results"
os.makedirs(EVAL_RESULTS_FOLDER, exist_ok=True)

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
        openai_api_key=os.environ['OPENROUTER_API_KEY'],
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

# Enhanced system prompt for better customer support
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for gokap innotech company. "
            "Your goal is to provide accurate, helpful, and concise responses to user queries. "
            "Use the provided tools to search for projects, company policies, and other information to assist the user's queries. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If a search comes up empty, expand your search before giving up."
            "Always maintain a professional and friendly tone."
            "When providing information about team members, projects, or company policies, be specific and accurate."
            "If you don't know the answer, be honest and suggest alternative resources."
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
        logger.error("Query is required")
        return "Please provide a query."
    
    thread_id = session.get("thread_id", str(uuid.uuid4()))
    session["thread_id"] = thread_id
    
    config = {
        "configurable": {
            "user_id": "1",
            "thread_id": thread_id,
        }
    }
    
    # Store the query and context for evaluation
    if "evaluation_data" not in session:
        session["evaluation_data"] = {"questions": [], "contexts": [], "answers": []}
    
    # Get context from RAG pipeline for evaluation
    try:
        context = rag_pipeline.get_retrieval_content(query)
        session["evaluation_data"]["questions"].append(query)
        session["evaluation_data"]["contexts"].append(context)
    except Exception as e:
        logger.error(f"Error getting context for evaluation: {str(e)}")
        context = []
    
    # Stream the response
    events = executer.stream(
        {"messages": ("user", query)}, config, stream_mode="values"
    )
    
    response = ""
    for event in events:
        response = event
    
    answer = response.get('messages')[-1].content
    logger.info(f"Generated answer: {answer[:100]}...")
    
    # Store the answer for evaluation
    if context:  # Only store if we have context
        session["evaluation_data"]["answers"].append(answer)
    
    return answer

@app.route('/', methods=['POST', 'GET'])
def home():
    if "chat_history" not in session:
        session["chat_history"] = [{'sender': 'assistant', 'text': "How can I help you today?"}]
    context = session['chat_history']
    return render_template('index.html', messages=context)

@app.route("/send_message", methods=["POST"])
def send_message():
    logger.info(f"Received message request: {request.json}")
    user_message = request.json.get("message", "")
    selected_model = request.json.get("model", "cohere")
    
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
    # Clear evaluation data
    session["evaluation_data"] = {"questions": [], "contexts": [], "answers": []}
    return jsonify({"status": "success"})

@app.route("/evaluate_rag", methods=["POST"])
def evaluate_rag():
    """Endpoint to evaluate the RAG system"""
    if "evaluation_data" not in session or not session["evaluation_data"]["questions"]:
        return jsonify({"error": "No evaluation data available. Please have some conversations first."}), 400
    
    try:
        # Prepare evaluation data
        eval_data = session["evaluation_data"]
        dataset = rag_evaluator.prepare_evaluation_data(
            questions=eval_data["questions"],
            contexts=eval_data["contexts"],
            answers=eval_data["answers"]
        )
        
        # Run evaluation
        model_name = session.get("current_model", "default")
        results = rag_evaluator.evaluate(dataset)
        
        # Save results
        rag_evaluator.save_results(results, model_name)
        
        return jsonify({
            "status": "success",
            "results": results,
            "model": model_name
        })
    except Exception as e:
        logger.error(f"Error during RAG evaluation: {str(e)}")
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

@app.route("/compare_models", methods=["GET"])
def compare_models():
    """Endpoint to compare different models"""
    try:
        # Get all evaluation results
        results_files = [f for f in os.listdir(EVAL_RESULTS_FOLDER) if f.endswith('.json')]
        
        if not results_files:
            return jsonify({"error": "No evaluation results available"}), 404
        
        # Load results
        model_results = {}
        for file in results_files:
            with open(os.path.join(EVAL_RESULTS_FOLDER, file), 'r') as f:
                data = json.load(f)
                model_name = file.split('_')[2]  # Extract model name from filename
                model_results[model_name] = data
        
        # Compare models
        comparison_df = rag_evaluator.compare_models(model_results)
        
        # Generate chart
        chart_path = os.path.join(EVAL_RESULTS_FOLDER, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        rag_evaluator.generate_comparison_chart(comparison_df, chart_path)
        
        return jsonify({
            "status": "success",
            "comparison": comparison_df.to_dict(),
            "chart_url": f"/{chart_path.replace(os.path.sep, '/')}"
        })
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500

# run flask app
if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True)