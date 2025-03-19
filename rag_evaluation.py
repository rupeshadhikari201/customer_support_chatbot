import os
import logging
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, 
                 eval_llm_name: str = "gpt-3.5-turbo",
                 results_dir: str = "evaluation_results"
                 ):
        """
        Initialize the RAG evaluator
        
        Args:
            eval_llm_name (str): Name of the LLM to use for evaluation
            results_dir (str): Directory to save evaluation results
        """
        self.eval_llm_name = eval_llm_name
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision
        }
        
    def prepare_evaluation_data(self, 
                                questions: List[str], 
                                contexts: List[List[str]], 
                                answers: List[str]) -> Dataset:
        """
        Prepare evaluation data in the format required by RAGAS
        
        Args:
            questions (List[str]): List of questions
            contexts (List[List[str]]): List of contexts for each question
            answers (List[str]): List of generated answers
            
        Returns:
            Dataset: HuggingFace dataset for evaluation
        """
        if not (len(questions) == len(contexts) == len(answers)):
            raise ValueError("Questions, contexts, and answers must have the same length")
            
        # Prepare data in the format required by RAGAS
        data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
        }
        
        return Dataset.from_dict(data)
        
    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the RAG system using standard metrics
        
        Args:
            dataset (Dataset): Dataset containing questions, contexts, and answers
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        logger.info("Starting RAG evaluation")
        results = {}
        
        try:
            # Run each metric evaluation
            for metric_name, metric_fn in self.metrics.items():
                logger.info(f"Evaluating {metric_name}")
                # Create a new instance of the metric
                metric = metric_fn()
                # Compute the metric
                result = metric(dataset)
                
                # Extract the score
                if isinstance(result, dict) and metric_name in result:
                    results[metric_name] = float(result[metric_name].mean())
                else:
                    # Some metrics return the score directly
                    results[metric_name] = float(np.mean(result))
                    
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
                
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {"error": str(e)}
            
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
            import matplotlib.pyplot as plt
            import seaborn as sns
            
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