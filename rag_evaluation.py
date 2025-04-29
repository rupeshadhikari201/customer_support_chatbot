import os
import logging
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langsmith import Client
from langsmith.evaluation import EvaluationResult, StringEvaluator
from langsmith.schemas import Example, Run
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['MISTRAL_API_KEY']= os.getenv("MISTRAL_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, 
                 eval_llm_name: str = "gpt-3.5-turbo",
                 results_dir: str = "evaluation_results"
                 ):
        """
        Initialize the RAG evaluator using LangSmith
        
        Args:
            eval_llm_name (str): Name of the LLM to use for evaluation
            results_dir (str): Directory to save evaluation results
        """
        self.eval_llm_name = eval_llm_name
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize LangSmith client
        if os.environ.get("LANGCHAIN_API_KEY"):
            self.client = Client(api_key=os.environ.get("LANGCHAIN_API_KEY"))
        else:
            logger.warning("LANGCHAIN_API_KEY not found in environment variables. LangSmith logging will be disabled.")
            self.client = None
        
        # Initialize LLM for evaluation
        try:
            self.eval_llm = ChatMistralAI(temperature=0)
            logger.info("Using Mistral AI for evaluation")
        except Exception as e:
            logger.warning(f"Error initializing Mistral AI: {str(e)}. Falling back to default evaluator.")
            self.eval_llm = None
        
        # Define evaluation metrics
        self.metrics = {
            "faithfulness": self._evaluate_faithfulness,
            "answer_relevancy": self._evaluate_answer_relevancy,
            "context_recall": self._evaluate_context_recall,
            "context_precision": self._evaluate_context_precision
        }

    def _evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the generated answer is faithful to the retrieved contexts
        
        Args:
            question (str): User question
            answer (str): Generated answer
            contexts (List[str]): Retrieved contexts
            
        Returns:
            float: Faithfulness score between 0 and 1
        """
        if not self.eval_llm:
            # Return a default score if no evaluator is available
            logger.warning("No evaluator LLM available for faithfulness evaluation. Returning default score.")
            return 0.7
            
        evaluator = StringEvaluator(
            criteria="faithfulness",
            llm=self.eval_llm,
            prompt_template="""You are evaluating the faithfulness of an answer to a question based on given contexts.
            
Question: {question}
Answer: {answer}
Contexts: {contexts}

Evaluate if the answer is faithful to the provided contexts. The answer should only contain information that is present in the contexts.
Score from 0 to 1, where:
0 - The answer contains significant information not present in the contexts
1 - The answer is completely faithful to the contexts

Reasoning:
""",
        )
        
        try:
            eval_result = evaluator.evaluate_strings(
                prediction=answer,
                input={"question": question, "contexts": "\n\n".join(contexts)}
            )
            
            return float(eval_result.normalized_score)
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {str(e)}")
            return 0.5

    def _evaluate_answer_relevancy(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the generated answer is relevant to the question
        
        Args:
            question (str): User question
            answer (str): Generated answer
            contexts (List[str]): Retrieved contexts
            
        Returns:
            float: Relevancy score between 0 and 1
        """
        if not self.eval_llm:
            # Return a default score if no evaluator is available
            logger.warning("No evaluator LLM available for answer relevancy evaluation. Returning default score.")
            return 0.7
            
        evaluator = StringEvaluator(
            criteria="answer_relevancy",
            llm=self.eval_llm,
            prompt_template="""You are evaluating the relevance of an answer to a question.
            
Question: {question}
Answer: {answer}


Evaluate if the answer directly addresses the question. The answer should be focused on what the question is asking.
Score from 0 to 1, where:
0 - The answer is completely irrelevant to the question
1 - The answer is highly relevant and directly addresses the question

Reasoning:
""",
        )
        
        try:
            eval_result = evaluator.evaluate_strings(
                prediction=answer,
                input={"question": question}
            )
            
            return float(eval_result.normalized_score)
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy: {str(e)}")
            return 0.5

    def _evaluate_context_recall(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the retrieved contexts contain the information needed to answer the question
        
        Args:
            question (str): User question
            answer (str): Generated answer
            contexts (List[str]): Retrieved contexts
            
        Returns:
            float: Context recall score between 0 and 1
        """
        if not self.eval_llm:
            # Return a default score if no evaluator is available
            logger.warning("No evaluator LLM available for context recall evaluation. Returning default score.")
            return 0.7
            
        evaluator = StringEvaluator(
            criteria="context_recall",
            llm=self.eval_llm,
            prompt_template="""You are evaluating if the retrieved contexts contain the information needed to answer a question.
            
Question: {question}
Contexts: {contexts}

Evaluate if the contexts contain all the information needed to adequately answer the question.
Score from 0 to 1, where:
0 - The contexts are missing critical information needed to answer the question
1 - The contexts contain all the necessary information to fully answer the question

Reasoning:
""",
        )
        
        try:
            eval_result = evaluator.evaluate_strings(
                prediction="\n\n".join(contexts),
                input={"question": question}
            )
            
            return float(eval_result.normalized_score)
        except Exception as e:
            logger.error(f"Error evaluating context recall: {str(e)}")
            return 0.5

    def _evaluate_context_precision(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the retrieved contexts contain only relevant information
        
        Args:
            question (str): User question
            answer (str): Generated answer
            contexts (List[str]): Retrieved contexts
            
        Returns:
            float: Context precision score between 0 and 1
        """
        if not self.eval_llm:
            # Return a default score if no evaluator is available
            logger.warning("No evaluator LLM available for context precision evaluation. Returning default score.")
            return 0.7
            
        evaluator = StringEvaluator(
            criteria="context_precision",
            llm=self.eval_llm,
            prompt_template="""You are evaluating if the retrieved contexts are precise and relevant to the question.
            
Question: {question}
Contexts: {contexts}

Evaluate if the contexts are precise and focused on information relevant to the question without unnecessary information.
Score from 0 to 1, where:
0 - The contexts contain mostly irrelevant information
1 - The contexts are highly precise and focused on information relevant to the question

Reasoning:
""",
        )
        
        try:
            eval_result = evaluator.evaluate_strings(
                prediction="\n\n".join(contexts),
                input={"question": question}
            )
            
            return float(eval_result.normalized_score)
        except Exception as e:
            logger.error(f"Error evaluating context precision: {str(e)}")
            return 0.5
        
    def prepare_evaluation_data(self, 
                                questions: List[str], 
                                contexts: List[List[str]], 
                                answers: List[str]) -> Dict[str, List]:
        """
        Prepare evaluation data
        
        Args:
            questions (List[str]): List of questions
            contexts (List[List[str]]): List of contexts for each question
            answers (List[str]): List of generated answers
            
        Returns:
            Dict[str, List]: Dictionary with prepared evaluation data
        """
        if not (len(questions) == len(contexts) == len(answers)):
            raise ValueError("Questions, contexts, and answers must have the same length")
            
        # Prepare data in the format required by LangSmith
        data = {
            "questions": questions,
            "contexts": contexts,
            "answers": answers,
        }
        
        return data
        
    def evaluate(self, dataset: Dict[str, List]) -> Dict[str, float]:
        """
        Evaluate the RAG system using LangSmith metrics
        
        Args:
            dataset (Dict[str, List]): Dataset containing questions, contexts, and answers
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        logger.info("Starting RAG evaluation with LangSmith")
        results = {}
        
        try:
            questions = dataset["questions"]
            contexts = dataset["contexts"]
            answers = dataset["answers"]
            
            metric_scores = {metric: [] for metric in self.metrics}
            
            # Evaluate each question-context-answer triplet
            for i, (question, context, answer) in enumerate(zip(questions, contexts, answers)):
                logger.info(f"Evaluating example {i+1}/{len(questions)}")
                
                # Calculate metrics for each example
                for metric_name, metric_fn in self.metrics.items():
                    score = metric_fn(question, answer, context)
                    metric_scores[metric_name].append(score)
            
            # Calculate average scores
            for metric_name, scores in metric_scores.items():
                results[metric_name] = float(np.mean(scores))
                    
            # Calculate overall score (weighted average)
            weights = {
                "faithfulness": 0.3,
                "answer_relevancy": 0.2,
                "context_recall": 0.25,
                "context_precision": 0.25
            }
            
            # Calculate weighted score for available metrics
            available_metrics = set(results.keys()).intersection(set(weights.keys()))
            if available_metrics:
                total_weight = sum(weights[m] for m in available_metrics)
                weighted_score = sum(results[m] * weights[m] for m in available_metrics) / total_weight
                results["overall_score"] = weighted_score
                
            # Log the evaluation results to LangSmith if API key is available
            if self.client:
                self._log_to_langsmith(dataset, results)
                
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {"error": str(e)}
    
    def _log_to_langsmith(self, dataset: Dict[str, List], results: Dict[str, float]) -> None:
        """
        Log evaluation results to LangSmith
        
        Args:
            dataset (Dict[str, List]): Dataset used for evaluation
            results (Dict[str, float]): Evaluation results
        """
        if not self.client:
            logger.warning("LangSmith client not available. Skipping logging.")
            return
            
        try:
            # Create a dataset in LangSmith
            dataset_name = f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Log each example to LangSmith
            for i, (question, contexts, answer) in enumerate(zip(
                dataset["questions"], dataset["contexts"], dataset["answers"]
            )):
                # Create an example
                example = Example(
                    inputs={"question": question, "contexts": contexts},
                    outputs={"answer": answer}
                )
                
                # Log the example to LangSmith
                self.client.create_example(
                    inputs=example.inputs,
                    outputs=example.outputs,
                    dataset_name=dataset_name
                )
                
            logger.info(f"Logged {len(dataset['questions'])} examples to LangSmith dataset '{dataset_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to log to LangSmith: {str(e)}")
            
    def save_results(self, results: Dict[str, float], model_name: str = "default"):
        """
        Save evaluation results to a file
        
        Args:
            results (Dict[str, float]): Evaluation results
            model_name (str): Name of the model being evaluated
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/rag_eval_{model_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Evaluation results saved to {filename}")
        
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare evaluation results across different models
        
        Args:
            model_results (Dict[str, Dict[str, float]]): Dictionary mapping model names to their evaluation results
            
        Returns:
            pd.DataFrame: DataFrame comparing models across metrics
        """
        # Create a DataFrame for comparison
        df = pd.DataFrame(model_results).T
        
        # Sort by overall score if available
        if "overall_score" in df.columns:
            df = df.sort_values("overall_score", ascending=False)
            
        return df
        
    def generate_comparison_chart(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Generate a comparison chart for different models
        
        Args:
            comparison_df (pd.DataFrame): DataFrame comparing models
            save_path (str): Path to save the chart
        """
        try:
            plt.figure(figsize=(12, 8))
            sns.set_theme(style="whitegrid")
            
            # Create heatmap for comparison
            ax = sns.heatmap(comparison_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
            plt.title("RAG Model Comparison")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Comparison chart saved to {save_path}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{self.results_dir}/model_comparison_{timestamp}.png"
                plt.savefig(save_path)
                logger.info(f"Comparison chart saved to {save_path}")
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating comparison chart: {str(e)}")

# Initialize the evaluator
rag_evaluator = RAGEvaluator()