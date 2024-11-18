mport time
import pandas as pd
from agents.intent_classifier import IntentClassifier

class IntentClassifierEvaluator:
    def __init__(self, intent_classifier: IntentClassifier):
        # Store the classifier for later use
        self.classifier = intent_classifier

    def evaluate(self, test_data: pd.DataFrame) -> int:
        """
        Evaluates the intent classifier on the provided test data.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing 'Prompt' and 'Answer' columns.

        Returns:
            int: The number of correct predictions made by the classifier.
        """
        correct_count = 0
        for _, row in test_data.iterrows():
            prompt = row['Prompt']
            expected_answer = row['Answer'].strip().strip('"')

            # Call the intent classifier to classify the intent
            intent = self.classifier.classify_intent(prompt).strip()
            print(f"Prompt: {prompt}, Classified Intent: {intent}, Expected: {expected_answer}")

            # Compare the result
            if intent == expected_answer:
                correct_count += 1

            # Throttle the requests to avoid rate limit (adjust sleep as necessary)
            time.sleep(10)  # Adjust time to stay within 6 requests per minute

        return correct_count

def main():
    # Initialize the intent classifier
    intent_classifier = IntentClassifier(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        temperature=0.3,
        max_tokens=256
    )

    evaluator = IntentClassifierEvaluator(intent_classifier)

    # Load test data from a CSV file
    file_path = '/Users/ianlasic/CAPT-new/evaluation/datasets/intent_classifier_eval.csv'
    test_data = pd.read_csv(file_path)

    # Get the total number of questions
    number_of_questions = test_data.shape[0]

    # Evaluate and print results
    num_correct = evaluator.evaluate(test_data)
    percentage_correct = num_correct / number_of_questions * 100
    print(f"Percentage correct: {percentage_correct:.2f}%")

if __name__ == "__main__":
    main()
