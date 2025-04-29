# Customer Support Chatbot

This is a customer support chatbot built with LangChain, LangGraph and LangSmith for evaluation.

## Features

- Advanced RAG-based customer support
- Multiple LLM support (OpenAI, Anthropic, Mistral, Cohere, etc.)
- LangSmith evaluation of RAG performance
- Speech-to-text input
- Web UI with real-time response streaming

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file:
   ```
   COHERE_API_KEY=your_cohere_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   GOOGLE_API_KEY=your_gemini_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=customer-support-rag-evaluation
   ```
4. Run the application:
   ```
   python app.py
   ```

## Usage

1. Open the web interface at http://localhost:5000
2. Select a language model from the sidebar
3. Type your customer support query
4. View the chatbot's response
5. Evaluate the responses using the LangSmith evaluation

## Evaluation with LangSmith

This project uses LangSmith for comprehensive evaluation of the RAG system. LangSmith provides detailed metrics to assess the quality of responses:

### Evaluation Metrics

- **Faithfulness**: Measures if the generated answer contains only information present in the retrieved contexts
- **Answer Relevancy**: Evaluates if the answer directly addresses the user's question
- **Context Recall**: Assesses if the retrieved contexts contain all the necessary information to answer the question
- **Context Precision**: Evaluates if the retrieved contexts are focused and relevant to the question
- **Overall Score**: A weighted average of all metrics

### Running Evaluations

1. Have several conversations with the chatbot
2. Click "Evaluate with LangSmith" in the sidebar
3. The system will evaluate all conversations using LLM-based evaluators
4. Results will be displayed in a detailed dashboard with visualizations

### Comparing Models

You can compare the performance of different language models:
1. Have conversations with different models
2. Click "Compare Models" to see a side-by-side comparison
3. Review which models perform best for different metrics

## Architecture

The system uses:
- LangChain for LLM interactions and RAG pipeline
- LangGraph for sophisticated agent behavior
- LangSmith for tracing and evaluation
- Flask for the web server
- Chart.js for visualization of evaluation results

## License

This project is licensed under the MIT License - see the LICENSE file for details.
