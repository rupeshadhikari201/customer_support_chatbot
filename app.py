from flask import Flask, request, render_template, session, jsonify
import os
import numpy as np
import logging
import uuid
import tempfile
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import threading

# Import custom modules
from agent_customer_support import query_llm as agent_query_llm
# from tts import generate_audio
from rag_pipeline import rag_pipeline
# from rag_evaluation import rag_evaluator

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

# Import LangSmith tracking
from langsmith import Client as LangSmithClient
from langsmith.run_helpers import traceable
try:
    from langsmith import run_tracking
except ImportError:
    # Fallback for older versions of langsmith
    from langsmith.run_helpers import run_tracking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LangSmith client if API key is available
langsmith_client = None
if os.environ.get("LANGCHAIN_API_KEY"):
    langsmith_client = LangSmithClient()

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
            max_tokens=500,
            timeout=60  # 60 second timeout
        )
    elif model_name == "anthropic":
        return ChatOpenAI(
            openai_api_key=os.environ['OPENROUTER_API_KEY'],
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3-haiku:beta",
            temperature=0.7,
            headers={"HTTP-Referer": "https://customer-support-chatbot.example", "X-Title": "Customer Support Chatbot"},
            request_timeout=60  # 60 second timeout
        )
    elif model_name == "mistral":
        return ChatMistralAI(
            mistral_api_key=os.environ['MISTRAL_API_KEY'],
            temperature=0.5,
            max_tokens=500,
            timeout=60  # 60 second timeout
        )
    elif model_name == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=os.environ['GOOGLE_API_KEY'],
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=60,   
            max_retries=2
        )
    elif model_name == "deepseek":
        return ChatOpenAI(
            openai_api_key=os.environ['OPENROUTER_API_KEY'],
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="deepseek/deepseek-r1-zero:free",
            temperature=0.7,
            headers={"HTTP-Referer": "https://customer-support-chatbot.example", "X-Title": "Customer Support Chatbot"},
            request_timeout=60  # 60 second timeout
        )
    elif model_name == "llama":
        return ChatOpenAI(
            openai_api_key=os.environ['OPENROUTER_API_KEY'],
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="meta-llama/llama-2-13b-chat",
            temperature=0.7,
            headers={"HTTP-Referer": "https://customer-support-chatbot.example", "X-Title": "Customer Support Chatbot"},
            request_timeout=60  # 60 second timeout
        )
    else:
        return ChatCohere(
            cohere_api_key=os.environ['COHERE_API_KEY'],
            temperature=0.5,
            max_tokens=500,
            timeout=60  # 60 second timeout
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
@traceable(name="customer_support_query")
def query_llm(query=None, model_name="cohere"):
    global current_llm, executer
    
    # Check if model has changed
    if model_name != session.get("current_model", "cohere"):
        try:
            current_llm = get_llm(model_name)
            executer = create_executer(current_llm)
            session["current_model"] = model_name
        except Exception as e:
            logger.error(f"Error initializing {model_name} model: {str(e)}")
            return f"Sorry, there was an error initializing the {model_name} model. Please try another model or try again later."
    
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
    context = []
    try:
        context = rag_pipeline.get_retrieval_content(query)
        session["evaluation_data"]["questions"].append(query)
        session["evaluation_data"]["contexts"].append(context)
    except Exception as e:
        logger.error(f"Error getting context for evaluation: {str(e)}")
        # Continue even if retrieval fails
    
    # Stream the response
    try:
        # Set up timeout using threading (works on Windows)
        
        class TimeoutHandler:
            def __init__(self, timeout=45):
                self.timeout = timeout
                self.timed_out = False
                self._timer = None
                
            def start(self):
                self._timer = threading.Timer(self.timeout, self._handle_timeout)
                self._timer.daemon = True
                self._timer.start()
                
            def cancel(self):
                if self._timer:
                    self._timer.cancel()
                    
            def _handle_timeout(self):
                self.timed_out = True
                
        # Create timeout handler with 45 second timeout
        timeout_handler = TimeoutHandler(timeout=45)
        timeout_handler.start()
        
        try:
            events = executer.stream(
                {"messages": ("user", query)}, config, stream_mode="values"
            )
            
            response_text = []
            
            for event in events:
                # Check if we've timed out
                if timeout_handler.timed_out:
                    break
                    
                messages = event.get("messages", [])
                for msg in messages:
                    # Handle different types of message objects
                    if hasattr(msg, 'type') and msg.type == 'assistant':
                        response_text.append(msg.content)
                    elif hasattr(msg, 'content'):
                        # Fallback for any message type with content
                        response_text.append(msg.content)
                    elif isinstance(msg, tuple) and len(msg) > 1 and msg[0] == "assistant":
                        # Handle old format (tuple)
                        response_text.append(msg[1])
                        
        finally:
            # Cancel the timeout
            timeout_handler.cancel()
            
        # Handle timeout
        if timeout_handler.timed_out:
            logger.warning("Response generation timed out")
            response_text = ["I apologize, but it's taking too long to generate a response. Please try again with a simpler query or try a different model."]
        
        # Concatenate all assistant responses
        full_response = "".join(response_text)
        
        # Handle empty responses
        if not full_response:
            logger.warning("Received empty response from LLM")
            full_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question or try with a different model."
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        full_response = "I apologize, but I encountered an error while processing your request. Please try again later or try with a different model."
    
    # Store the response for evaluation
    session["evaluation_data"]["answers"].append(full_response)
    
    # Log to LangSmith if available
    if langsmith_client:
        try:
            # Safe way to get run_id
            run_id = None
            try:
                # Check if the tracer attribute exists
                if hasattr(query_llm, '__langsmith_tracer__') and query_llm.__langsmith_tracer__:
                    if hasattr(query_llm.__langsmith_tracer__, 'latest_run') and query_llm.__langsmith_tracer__.latest_run:
                        run_id = query_llm.__langsmith_tracer__.latest_run.id
            except Exception:
                pass
                
            if run_id:
                # Add metadata about the retrieval
                langsmith_client.update_run(
                    run_id=run_id,
                    inputs={
                        "query": query,
                        "contexts": context
                    },
                    outputs={"response": full_response}
                )
            else:
                # Create a new run if tracing didn't work
                langsmith_client.run_tracking(
                    project_name=os.environ.get("LANGCHAIN_PROJECT", "customer-support-chatbot"),
                    name="customer_support_query",
                    inputs={
                        "query": query,
                        "model": model_name,
                        "contexts": context
                    },
                    outputs={"response": full_response}
                )
        except Exception as e:
            logger.warning(f"Failed to log to LangSmith: {str(e)}")
    
    return full_response

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

    try:
        # Call agent Model
        response = query_llm(user_message, selected_model)
        
        # Generate audio response
        # audio_data = generate_audio(response)
        audio_url = None
        
        # if audio_data is not None:
        #     sample_rate = 22050
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     unique_id = uuid.uuid4().hex[:8]
        #     audio_filename = f"{unique_id}_{timestamp}.wav"
        #     audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            
        #     # Save audio file
        #     sf.write(audio_path, audio_data, sample_rate, format="WAV")
        #     audio_url = f"/static/audio/{audio_filename}"
        
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
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        error_message = "I apologize, but there was an error processing your request. Please try again."
        
        # Add error message to chat history
        context = session.get("chat_history", [])
        context.append({"text": user_message, "sender": 'user'})
        context.append({"text": error_message, 'sender': 'assistant'})
        session['chat_history'] = context
        
        return jsonify({
            "response": error_message,
            "model": selected_model,
            "error": str(e)
        }), 500

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = [{'sender': 'assistant', 'text': "How can I help you today?"}]
    session["thread_id"] = str(uuid.uuid4())
    # Clear evaluation data
    session["evaluation_data"] = {"questions": [], "contexts": [], "answers": []}
    return jsonify({"status": "success"})

@app.route("/evaluate_rag", methods=["POST"])
def evaluate_rag():
    """Endpoint to evaluate the RAG system using LangSmith"""
    if "evaluation_data" not in session or not session["evaluation_data"]["questions"]:
        return jsonify({"error": "No evaluation data available. Please have some conversations first."}), 400
    
    try:
        # Prepare evaluation data
        eval_data = session["evaluation_data"]
        
        # Ensure there's at least one example with context
        valid_examples = [i for i, ctx in enumerate(eval_data["contexts"]) if ctx]
        if not valid_examples:
            return jsonify({"error": "No valid examples with context found. Please have some conversations that retrieve documents."}), 400
            
        # Filter out examples without context
        if len(valid_examples) < len(eval_data["questions"]):
            logger.warning(f"Filtering out {len(eval_data['questions']) - len(valid_examples)} examples without context")
            filtered_data = {
                "questions": [eval_data["questions"][i] for i in valid_examples],
                "contexts": [eval_data["contexts"][i] for i in valid_examples],
                "answers": [eval_data["answers"][i] for i in valid_examples]
            }
        else:
            filtered_data = eval_data
        
        # Prepare dataset
        dataset = rag_evaluator.prepare_evaluation_data(
            questions=filtered_data["questions"],
            contexts=filtered_data["contexts"],
            answers=filtered_data["answers"]
        )
        
        # Run evaluation
        model_name = session.get("current_model", "default")
        results = rag_evaluator.evaluate(dataset)
        
        # Check if evaluation returned an error
        if "error" in results:
            logger.error(f"Evaluation returned an error: {results['error']}")
            return jsonify({
                "error": f"Evaluation failed: {results['error']}",
                "partial_results": results
            }), 500
        
        # Save results
        try:
            rag_evaluator.save_results(results, model_name)
        except Exception as save_error:
            logger.error(f"Error saving evaluation results: {str(save_error)}")
            # Continue anyway - not critical
        
        # Create a more detailed response for the UI
        response_data = {
            "status": "success",
            "results": results,
            "model": model_name,
            "num_examples": len(filtered_data["questions"]),
            "evaluation_method": "LangSmith",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error during RAG evaluation: {str(e)}", exc_info=True)
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

@app.route("/compare_models", methods=["GET"])
def compare_models():
    """Endpoint to compare different models"""
    try:
        # Get all evaluation results
        results_files = [f for f in os.listdir(EVAL_RESULTS_FOLDER) if f.endswith('.json')]
        
        if not results_files:
            return jsonify({"error": "No evaluation results available. Please run evaluations for at least one model first."}), 404
        
        # Load results
        model_results = {}
        for file in results_files:
            try:
                with open(os.path.join(EVAL_RESULTS_FOLDER, file), 'r') as f:
                    data = json.load(f)
                    if "error" in data and len(data) == 1:
                        # Skip files that only contain error information
                        continue
                    # Extract model name from filename
                    model_name = file.split('_')[2] if len(file.split('_')) > 2 else "unknown"
                    model_results[model_name] = data
            except Exception as file_error:
                logger.error(f"Error loading result file {file}: {str(file_error)}")
                # Skip this file
                continue
        
        if not model_results:
            return jsonify({"error": "No valid evaluation results found."}), 404
        
        # Compare models
        comparison_df = rag_evaluator.compare_models(model_results)
        
        # Generate chart
        chart_path = None
        try:
            chart_path = os.path.join(EVAL_RESULTS_FOLDER, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            rag_evaluator.generate_comparison_chart(comparison_df, chart_path)
        except Exception as chart_error:
            logger.error(f"Error generating comparison chart: {str(chart_error)}")
            # Continue without the chart
        
        return jsonify({
            "status": "success",
            "comparison": comparison_df.to_dict(),
            "chart_url": f"/{chart_path.replace(os.path.sep, '/')}" if chart_path else None,
            "models_compared": list(model_results.keys())
        })
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}", exc_info=True)
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500

# run flask app
if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True)