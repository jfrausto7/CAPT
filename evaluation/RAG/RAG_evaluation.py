import csv
import re
import time
import requests
from typing import Dict, Tuple
from langchain_together import Together
from word2number import w2n
from langchain.chains import RetrievalQA
from collections import defaultdict
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np

from agents.retrieval.MultiVectorstoreRetriever import MultiVectorstoreRetriever

class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.window_size = 60  # seconds
        self.requests = []
        self.min_pause = 12.0  # Minimum 20 seconds between requests (5 per minute to be safe)
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        now = time.time()
        
        # Always wait at least min_pause seconds between requests
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_pause:
            sleep_time = self.min_pause - time_since_last
            time.sleep(sleep_time)
        
        # Remove requests older than our window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.window_size]
        
        # If we've hit our limit, wait until enough time has passed
        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request falls out of our window
            sleep_time = max(self.window_size - (now - self.requests[0]), self.min_pause)
            time.sleep(sleep_time)
        
        # Update last request time and add to requests list
        self.last_request_time = time.time()
        self.requests.append(self.last_request_time)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException)
)
def make_api_request(qa_chain, prompt, rate_limiter, collections):
    """Make API request with retry logic"""
    rate_limiter.wait_if_needed()
    try:
        qa_chain.retriever.with_collections(collections)    # collections is ['articles']
        return qa_chain.invoke({"query": prompt})
    except Exception as e:
        print(f"\nAPI request failed: {str(e)}")
        raise

def run_evaluation():
    # Initialize rate limiter (3 requests per minute to be safe)
    rate_limiter = RateLimiter(max_requests_per_minute=5)

    # Initialize the model with retries
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

    # Read CSV file first to get total count
    with open("evaluation/datasets/rag_eval_qa.csv", "r") as csv_file:
        total_questions = sum(1 for _ in csv.DictReader(csv_file))

    # Read and evaluate with progress tracking
    with open("evaluation/datasets/rag_eval_qa.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, 1):
            prompt = row["Prompt"]
            expected_answer = row["Answer"]

            print(f"\nProcessing question {idx}/{total_questions}")
            
            try:
                # Make API call with retries and rate limiting
                response = make_api_request(qa_chain, prompt, rate_limiter, ['articles'])
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
                print(f"Prompt: {prompt}")
                print(f"Expected answer: {expected_answer}")
                print(f"Actual answer: {actual_answer}")
                print(f"Score: {score:.2f}")

                # Save intermediate results after each successful evaluation
                save_results(results, "evaluation/results/evaluation_results_intermediate.csv")

            except Exception as e:
                print(f"\nError processing question {idx}: {str(e)}")
                # Save error information
                result = {
                    'question_id': idx,
                    'prompt': prompt,
                    'expected': expected_answer,
                    'actual': f"ERROR: {str(e)}",
                    'overall_score': 0.0,
                    'detailed_scores': {'text_match': 0.0, 'number_similarity': 0.0}
                }
                results.append(result)
                save_results(results, "evaluation/results/evaluation_results_intermediate.csv")
                continue

    # Calculate and save final metrics
    save_final_results(results, evaluator)

def save_results(results, filename):
    """Save results to CSV file"""
    with open(filename, 'w', newline='') as f:
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

def save_final_results(results, evaluator):
    """Calculate and save final metrics"""
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
    
    # Save final results
    save_results(results, "evaluation/results/evaluation_results_final.csv")
    print(f"\nFinal results saved to evaluation_results_final.csv")

class AnswerEvaluator:
    def __init__(self):
        self.matches = 0
        self.total_questions = 0
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.answer_scores = []
        
        # Dictionary for common number words and their variations
        self.number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12
        }
        
        # Add variations
        self.number_variations = {
            '1st': 1, 'first': 1,
            '2nd': 2, 'second': 2,
            '3rd': 3, 'third': 3,
            '4th': 4, 'fourth': 4,
            # Add more as needed
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens for number ranges
        text = ' '.join(text.split())
        return text

    def normalize_number_expression(self, text: str) -> str:
        """Convert number words to digits and standardize number expressions."""
        text = text.lower()
        
        # Handle hyphenated ranges first (e.g., "1-2" or "one-two")
        range_pattern = r'(\d+|\b[a-z]+\b)-(\d+|\b[a-z]+\b)'
        
        def convert_range(match):
            start, end = match.groups()
            try:
                # Try converting word to number if it's a word
                start_num = w2n.word_to_num(start) if start.isalpha() else int(start)
                end_num = w2n.word_to_num(end) if end.isalpha() else int(end)
                return f"{start_num}-{end_num}"
            except ValueError:
                return match.group(0)
        
        text = re.sub(range_pattern, convert_range, text)
        
        # Handle "or" between numbers (e.g., "one or two")
        or_pattern = r'(\d+|\b[a-z]+\b)\s+or\s+(\d+|\b[a-z]+\b)'
        
        def convert_or(match):
            first, second = match.groups()
            try:
                first_num = w2n.word_to_num(first) if first.isalpha() else int(first)
                second_num = w2n.word_to_num(second) if second.isalpha() else int(second)
                return f"{first_num}-{second_num}"
            except ValueError:
                return match.group(0)
        
        text = re.sub(or_pattern, convert_or, text)
        
        return text

    def extract_numbers(self, text: str) -> list:
        """Extract all numbers from text, including written numbers."""
        normalized_text = self.normalize_number_expression(text)
        
        # Handle ranges (e.g., "1-2" or "1 to 2")
        ranges = re.findall(r'(\d+)[-to]+(\d+)', normalized_text)
        numbers = []
        
        for start, end in ranges:
            numbers.extend([float(start), float(end)])
        
        # Extract individual numbers
        individual_numbers = [float(num) for num in re.findall(r'\d+\.?\d*', normalized_text) 
                            if not any(num in range_match for range_match in ranges)]
        numbers.extend(individual_numbers)
        
        return sorted(set(numbers))  # Remove duplicates and sort

    def calculate_number_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity between numbers in the texts."""
        expected_nums = self.extract_numbers(expected)
        actual_nums = self.extract_numbers(actual)
        
        if not expected_nums or not actual_nums:
            return 0.0
        
        # For number ranges, check if the ranges overlap
        if len(expected_nums) == 2 and len(actual_nums) == 2:
            expected_range = range(int(expected_nums[0]), int(expected_nums[1]) + 1)
            actual_range = range(int(actual_nums[0]), int(actual_nums[1]) + 1)
            overlap = set(expected_range) & set(actual_range)
            if overlap:
                return len(overlap) / max(len(expected_range), len(actual_range))
        
        # For individual numbers, find best matches
        similarities = []
        for exp_num in expected_nums:
            for act_num in actual_nums:
                similarity = 1 - min(abs(exp_num - act_num) / max(exp_num, act_num), 1.0)
                similarities.append(similarity)
        
        return max(similarities) if similarities else 0.0

    def evaluate_answer(self, expected: str, actual: str) -> Tuple[float, Dict[str, float]]:
        """Evaluate the similarity between expected and actual answers."""
        expected_clean = self.clean_text(expected)
        actual_clean = self.clean_text(actual)
        
        # Normalize number expressions in both texts
        expected_normalized = self.normalize_number_expression(expected_clean)
        actual_normalized = self.normalize_number_expression(actual_clean)
        
        # Check for exact matches after normalization
        text_match = float(expected_normalized in actual_normalized or 
                         actual_normalized in expected_normalized)
        
        # Calculate number similarity
        number_similarity = self.calculate_number_similarity(expected_normalized, actual_normalized)
        
        # Take the maximum of text match and number similarity
        overall_score = max(text_match, number_similarity)
        
        scores = {
            'text_match': text_match,
            'number_similarity': number_similarity
        }
        
        # Update metrics for precision/recall calculation
        if overall_score >= 0.8:
            self.true_positives[expected_clean] += 1
        else:
            self.false_positives[actual_clean] += 1
            self.false_negatives[expected_clean] += 1
        
        return overall_score, scores

def create_retrieval_chain(vector_dir: str, model):
    retriever = MultiVectorstoreRetriever(
        vector_dir=vector_dir,
        k=10,
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

if __name__ == "__main__":
    run_evaluation()