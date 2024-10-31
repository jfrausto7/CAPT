import csv
import re
from typing import Dict, Tuple
from langchain_together import Together
from langchain.chains import RetrievalQA
from collections import defaultdict

import numpy as np

from agents.retrieval.MultiVectorstoreRetriever import MultiVectorstoreRetriever

class AnswerEvaluator:
    def __init__(self):
        self.matches = 0
        self.total_questions = 0
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.answer_scores = []

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def contains_numbers(self, text: str) -> bool:
        """Check if text contains numbers."""
        return bool(re.search(r'\d', text))

    def extract_numbers(self, text: str) -> list:
        """Extract all numbers from text."""
        return [float(num) for num in re.findall(r'\d+\.?\d*', text)]

    def calculate_number_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity between numbers in the texts."""
        expected_nums = self.extract_numbers(expected)
        actual_nums = self.extract_numbers(actual)
        
        if not expected_nums or not actual_nums:
            return 0.0
            
        similarities = []
        for exp_num in expected_nums:
            for act_num in actual_nums:
                similarity = 1 - min(abs(exp_num - act_num) / max(exp_num, act_num), 1.0)
                similarities.append(similarity)
                
        return max(similarities) if similarities else 0.0

    def evaluate_answer(self, expected: str, actual: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the similarity between expected and actual answers.
        Returns a tuple of (overall_score, detailed_scores).
        """
        expected_clean = self.clean_text(expected)
        actual_clean = self.clean_text(actual)
        
        # Check if expected answer is contained in actual answer or vice versa
        text_match = float(expected_clean in actual_clean or actual_clean in expected_clean)
        
        # Special handling for numerical comparisons
        if self.contains_numbers(expected) and self.contains_numbers(actual):
            number_similarity = self.calculate_number_similarity(expected, actual)
            # Take the maximum of text match and number similarity
            overall_score = max(text_match, number_similarity)
        else:
            overall_score = text_match
        
        scores = {
            'text_match': text_match,
            'number_similarity': number_similarity if self.contains_numbers(expected) and self.contains_numbers(actual) else 0.0
        }
        
        # Update metrics for precision/recall calculation
        if overall_score >= 0.8:  # Consider it a true positive if score is high enough
            self.true_positives[expected_clean] += 1
        else:
            self.false_positives[actual_clean] += 1
            self.false_negatives[expected_clean] += 1
        
        return overall_score, scores

def create_retrieval_chain(vector_dir: str, model):
    retriever = MultiVectorstoreRetriever(
        vector_dir=vector_dir,
        k=4,
        score_threshold=0.7,
        show_progress=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def calculate_metrics(evaluator):
    """Calculate precision, recall, and F1 score."""
    total_tp = sum(evaluator.true_positives.values())
    total_fp = sum(evaluator.false_positives.values())
    total_fn = sum(evaluator.false_negatives.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }

def run_evaluation():
    # Initialize the model
    model = Together(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        temperature=0.3,
        max_tokens=128
    )

    # Create the retrieval chain
    qa_chain = create_retrieval_chain("app/vectorstore/store", model)

    # Initialize evaluator
    evaluator = AnswerEvaluator()
    results = []

    # Read and evaluate
    with open("prompts-and-answers.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, 1):
            prompt = row["Prompt"]
            expected_answer = row["Answer"]

            response = qa_chain.invoke(prompt, collections=['articles'])
            actual_answer = response['result']

            # Evaluate the answer
            score, detailed_scores = evaluator.evaluate_answer(expected_answer, actual_answer)
            
            result = {
                'question_id': idx,
                'prompt': prompt,
                'expected': expected_answer,
                'actual': actual_answer,
                'overall_score': score,
                'detailed_scores': detailed_scores
            }
            results.append(result)

            # Print detailed results for each question
            print(f"\nQuestion {idx}:")
            print(f"Prompt: {prompt}")
            print(f"Expected answer: {expected_answer}")
            print(f"Actual answer: {actual_answer}")
            print(f"Score: {score:.2f}")

    # Calculate metrics
    metrics = calculate_metrics(evaluator)
    scores = [r['overall_score'] for r in results]
    matches = sum(1 for r in results if r['overall_score'] >= 0.8)

    # Print summary metrics
    print("\nSUMMARY METRICS")
    print("=" * 50)
    print(f"Total questions: {len(results)}")
    print(f"Matches (score >= 0.8): {matches}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Median score: {np.median(scores):.2f}")
    
    # Save detailed results to CSV
    output_file = "evaluation_results.csv"
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['question_id', 'prompt', 'expected', 'actual', 'overall_score', 
                     'text_match', 'number_similarity']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'question_id': result['question_id'],
                'prompt': result['prompt'],
                'expected': result['expected'],
                'actual': result['actual'],
                'overall_score': result['overall_score'],
                **result['detailed_scores']
            }
            writer.writerow(row)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    run_evaluation()