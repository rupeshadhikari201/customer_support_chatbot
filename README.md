# Enhanced RAG Chatbot with Speech-to-Text

This project implements an enhanced RAG (Retrieval-Augmented Generation) chatbot with speech-to-text capabilities, optimized RAG pipeline, and comprehensive evaluation tools.

## Features

- **Speech-to-Text**: Both client-side and server-side speech recognition for better accuracy and reliability
- **Optimized RAG Pipeline**: Enhanced document retrieval with chunking strategies, ensemble retrieval, and contextual compression
- **Multiple LLM Support**: Integration with various language models (Cohere, Anthropic, Mistral, Gemini, Deepseek, LLaMA)
- **Evaluation Framework**: Comprehensive evaluation of RAG performance using standard metrics
- **Model Comparison**: Tools to compare different LLM models for research purposes
- **Text-to-Speech**: Audio responses for better accessibility

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables in `.env` file:
   ```
   COHERE_API_KEY=your_cohere_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   DB_URL=your_database_url
   PYTHONPATH=.
   RAGAS_CACHE_DIR=./ragas_cache
   ```

## Running the Application

Start the Flask application:

```
python app.py
```

The application will be available at `http://localhost:5000`.

## Usage

1. **Chat Interface**: Type your questions or use the microphone button for speech input
2. **Model Selection**: Choose different LLM models from the sidebar
3. **Evaluation**: Use the "Evaluate" button to assess RAG performance
4. **Model Comparison**: Use the "Compare" button to compare different models

## RAG Pipeline Improvements

The RAG pipeline has been optimized with:

- **Better Chunking**: Improved document splitting with optimized chunk size and overlap
- **Ensemble Retrieval**: Combination of semantic search and BM25 for better retrieval
- **Contextual Compression**: Extraction of relevant parts from retrieved documents
- **Fallback Mechanisms**: Graceful degradation when primary retrieval methods fail

## Evaluation Metrics

The system evaluates RAG performance using:

- **Faithfulness**: Measures if the generated answer is faithful to the retrieved context
- **Answer Relevancy**: Evaluates if the answer is relevant to the question
- **Context Relevancy**: Assesses if the retrieved context is relevant to the question
- **Context Recall**: Measures how much of the relevant information is retrieved
- **Context Precision**: Evaluates the precision of the retrieved context
- **Harmfulness**: Checks for potentially harmful content in responses

## Speech-to-Text Implementation

The system implements two approaches for speech recognition:

1. **Client-side**: Using the Web Speech API for browsers that support it
2. **Server-side**: Using Python's SpeechRecognition library for more reliable transcription

## Project Structure

- `app.py`: Main Flask application
- `agent_customer_support.py`: Agent implementation with tools
- `rag_pipeline.py`: Optimized RAG implementation
- `rag_evaluation.py`: Evaluation framework
- `stt.py`: Server-side speech-to-text implementation
- `tts.py`: Text-to-speech implementation
- `static/js/script.js`: Client-side JavaScript for UI interactions
- `templates/index.html`: Main UI template

## Research Applications

This system is designed for comparative analysis of different LLM models in a RAG context. The evaluation framework provides quantitative metrics for research purposes. 