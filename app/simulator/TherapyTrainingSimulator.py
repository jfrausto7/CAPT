import pdb
from typing import List, Dict
import csv
import numpy as np
from difflib import SequenceMatcher

class TherapyTrainingSimulator:
    def __init__(self, transcript_path: str):
        self.scenarios = self.load_scenarios(transcript_path)
        self.current_scenario = None
        
    def load_scenarios(self, transcript_path: str) -> List[Dict]:
        """Load training scenarios from therapy transcripts."""
        scenarios = []
        current_scenario = []
        
        with open(transcript_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)  # Load all rows at once for easier indexing
            
            i = 0
            while i < len(rows):
                row = rows[i]
                
                # Start new scenario if therapist speaks first in sequence
                if row['speaker'] == 'therapist' and not current_scenario:
                    current_scenario = [row]
                
                # Add to current scenario
                elif current_scenario:
                    current_scenario.append(row)
                    
                    # If client speaks last, find the therapist's follow-up message
                    if row['speaker'] == 'client':
                        # Check if there's a follow-up therapist message (next row)
                        if i + 1 < len(rows) and rows[i + 1]['speaker'] == 'therapist':
                            ideal_response = rows[i + 1]['message']
                        else:
                            ideal_response = None  # No follow-up therapist message
                        
                        # Add the full scenario to the list
                        scenarios.append({
                            'context': current_scenario[0]['message'],
                            'client_response': current_scenario[1]['message'],
                            'ideal_response': ideal_response,
                            'full_exchange': current_scenario
                        })
                        # Reset for the next scenario
                        current_scenario = []
                
                # Increment the index
                i += 1
        
        return scenarios
    
    def get_random_scenario(self) -> Dict:
        """Get a random training scenario."""
        self.current_scenario = np.random.choice(self.scenarios)
        return {
            'context': self.current_scenario['context'],
            'client_response': self.current_scenario['client_response']
        }
    
    def evaluate_trainee_response(self, trainee_response: str) -> Dict:
        """Evaluate a trainee's response against the ideal response."""
        if not self.current_scenario:
            raise ValueError("No active scenario. Call get_random_scenario() first.")
            
        ideal_response = self.current_scenario['ideal_response']
        
        # Basic similarity score
        similarity = SequenceMatcher(None, trainee_response.lower(), ideal_response.lower()).ratio()
        
        # Check for key therapeutic elements
        key_elements = {
            'validation': any(phrase in trainee_response.lower() for phrase in 
                            ['understand', 'hear you', 'valid', 'natural']),
            'empathy': any(phrase in trainee_response.lower() for phrase in 
                          ['feel', 'sense', 'imagine', 'must be']),
            'question': '?' in trainee_response,
            'forward_looking': any(phrase in trainee_response.lower() for phrase in 
                                 ['would you', 'could we', 'lets', 'moving forward'])
        }
        
        feedback = {
            'similarity_score': round(similarity * 100, 2),
            'elements_present': key_elements,
            'strengths': [],
            'areas_for_improvement': []
        }
        
        # Generate specific feedback
        if similarity > 0.7:
            feedback['strengths'].append("Your response aligns well with the reference response")
        
        for element, present in key_elements.items():
            if present:
                feedback['strengths'].append(f"Good use of {element}")
            else:
                feedback['areas_for_improvement'].append(f"Consider incorporating {element}")
                
        return feedback

if __name__ == "__main__":
    transcripts = "data/transcripts/therapy_transcripts.csv"
    
    # Initialize simulator
    simulator = TherapyTrainingSimulator(transcripts)
    
    # Get a random scenario
    scenario = simulator.get_random_scenario()
    print("Training Scenario:")
    print(f"Context: {scenario['context']}\n")
    print(f"Client: {scenario['client_response']}")
    
    # Example trainee response
    trainee_response = input("\nEnter your response: ")
    feedback = simulator.evaluate_trainee_response(trainee_response)
    print("\nFeedback on trainee response:")
    print(feedback)