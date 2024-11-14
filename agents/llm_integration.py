import time
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import re
import json

from app.clinical_trial_search import search_clinical_trials
from config import RATE_LIMIT_BREAK

COLLECTIONS = []
KEY_WORDS = ["ketamine", "mdma", "lsd", "psilocybin"]

class IntentClassifier:
    # Define intentions
    INTENTIONS = ["Other", "General_knowledge", "Before_experience",
    "During_experience", "After_experience", "Clinical_trial_recruitment"]

    # Step 2: Map key words to specific collection data
    KEY_WORDS_DATA_MAPPING = {"ketamine": ['articles', 'guidelines_ketamine', 'guidelines_general', 'general_knowledge'],
                           "mdma": ['articles', 'guidelines_mdma', 'guidelines_general', 'general_knowledge'],
                           "lsd": ['articles', 'guidelines_LSD', 'guidelines_general', 'general_knowledge'],
                           "psilocybin": ['articles', 'guidelines_psilocybin', 'guidelines_general', 'general_knowledge']}

    def handle_clinical_trial_search(self, user_input: str) -> str:
        """
        Process clinical trial recruitment queries by extracting relevant information
        and searching for matching trials.
        """
        # Initialize a basic LLM to help extract location and condition information
        extraction_prompt = f"""
        Extract the drug name, location, and medical condition (if mentioned) from the following query.
        Return the information in JSON format with keys: drug, location, condition.
        If any field is not mentioned, set it to null.
        
        Query: {user_input}
        
        Example output (JSON):
        ---
        {{"drug": "MDMA", "location": "California", "condition": "PTSD"}}
        ---

        ONLY RETURN THE OUTPUT:
        """
        
        # Use the existing LLM to extract information
        response = self.intent_model.invoke(extraction_prompt).strip()
        response = response.split('"""')[0]
        
        try:
            # Parse the JSON response
            search_params = json.loads(response)
            
            # Ensure we have at least a drug name
            if not search_params.get("drug"):
                return {'result': "I couldn't determine which psychedelic medication you're interested in. Could you please specify?"}
            
            # Set defaults if location or condition are missing
            location = search_params.get("location", "United States")
            condition = search_params.get("condition")
            
            # Search for clinical trials
            trials = search_clinical_trials(
                drug=search_params["drug"],
                location=location,
                condition=condition
            )
            
            # Format the results into a readable response
            if not trials:
                return {'result': f"No active clinical trials found for {search_params['drug']} in {location}."}
            
            response = f"Found {len(trials)} clinical trial(s) for {search_params['drug']}:\n\n"
            
            for trial in trials:
                response += f"Title: {trial.title}\n"
                response += f"Status: {trial.status}\n"
                response += f"Phase: {', '.join(trial.phase) if trial.phase else 'Not specified'}\n"
                response += f"Conditions: {', '.join(trial.conditions)}\n"
                response += f"Location(s):\n"
                for loc in trial.locations[:3]:  # Limit to first 3 locations
                    response += f"- {loc.facility} ({loc.city}, {loc.state})\n"
                response += f"More information: {trial.study_url}\n\n"
            
            return {'result': response}
            
        except json.JSONDecodeError:
            return {'result': "I had trouble processing the clinical trial search. Could you please rephrase your request?"}
        except Exception as e:
            return f"An error occurred while searching for clinical trials: {str(e)}"

    async def complete(self, user_input: str) -> str:
        global COLLECTIONS
        # Set up prompt with user input
        self.create_prompt_template(user_input)

        # Step 1: search for keywords and update global variable array
        # first make the user input all lowercase, then search for key words using regex
        user_input = user_input.lower()
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in KEY_WORDS) + r')\b'
        COLLECTIONS = re.findall(pattern, user_input)

        # Inside the retrieval chain the 3 workflows are accounted for
        return self.retrieval_chain(user_input)

    def __init__(self, model: str, temperature: int, max_tokens: int, qa_chain: RetrievalQA):
        # Set up LLM agent and retrieval agent
        self.intent_model = Together(model=model, temperature=temperature, max_tokens=max_tokens)
        self.formatter = Together(model=model, temperature=temperature, max_tokens=max_tokens*4)
        self.qa_chain = qa_chain
        self.max_tokens = max_tokens

    def create_prompt_template(self, user_input: str) -> PromptTemplate:
        template = """
        You are an intention classifier. You will be provided with a user input inquiring about psychedelic drugs and clinical trials.
        
        Your task is to classify the user's input into EXACTLY one of the following intents: {intentions}. 
        
        DO NOT classify input as 'Clinical_trial_recruitment' unless the input specifically mentions a drug and a location.

        Here are some examples:
        -----------------------
        Input: "What dosage of psilocybin should I take for my first time?"
        CLASSIFIED INTENTION: Before_experience

        Input: "I'm having intense anxiety during my session, what should I do?"
        CLASSIFIED INTENTION: During_experience

        Input: "How long have psychedelics been used in traditional medicine?"
        CLASSIFIED INTENTION: General_knowledge

        Input: "Can you help me integrate my recent ayahuasca experience?"
        CLASSIFIED INTENTION: After_experience

        Input: "Where can I find MDMA clinical studies in California?"
        CLASSIFIED INTENTION: Clinical_trial_recruitment

        Input: "What's the weather like today?"
        CLASSIFIED INTENTION: Other
        -----------------------

        Input to classify: {user_input}

        DO NOT include any other text, punctuation, or explanation. ONLY OUTPUT ONE INTENTION.

        CLASSIFIED INTENTION:"""

        self.prompt = PromptTemplate.from_template(template)
        self.prompt.format(user_input=user_input, intentions=self.INTENTIONS)

        return self.prompt

    def classify_intent(self, user_input: str) -> str:
        self.intent_chain = self.prompt | self.intent_model

        classified_intent = self.intent_chain.invoke({"user_input": user_input, "intentions": self.INTENTIONS}).strip().split(' ', 1)[0]

        for intent in self.INTENTIONS:
            # checking if the intent is in the LLM's response (which could be 64 tokens long)
            if intent in classified_intent:
                return intent
        # If no intent is found default to Other
        return "Other"
    
    def format_therapeutic_response(self, text: str) -> str:
        """
        Format raw information into a therapeutic response maintaining CAPT's persona.
        
        Args:
            text (str): Raw response text
            
        Returns:
            str: Formatted therapeutic response
        """
        prompt = f"""As CAPT, a compassionate psychedelic-assisted therapist, rephrase this information 
        in an empathetic and supportive way while maintaining accuracy. Keep the same information but 
        make it sound like it's coming from a therapist in conversation:

        Information to rephrase: {text}

        Respond in CAPT's voice. ONLY PROVIDE THE REPHRASED RESPONSE, NO PREAMBLE OR EXPLANATIONS:"""
        
        formatted_response = self.formatter.invoke(prompt).strip()
        return formatted_response

    def clean_and_format_rag_response(self, response: dict) -> dict:
        """
        Clean up RAG response and format it in therapeutic voice.
        Only removes quotes and backticks from start/end while preserving them within text.
        
        Args:
            response (dict): Raw response from the QA chain
            
        Returns:
            dict: Cleaned and therapeutically formatted response
        """
        if not response or not isinstance(response, dict):
            return {"result": ""}
            
        # Extract the main response text
        result = response.get("result", "")

        # Format the cleaned response in therapeutic voice
        time.sleep(RATE_LIMIT_BREAK)  # Respect rate limiting
        result = self.format_therapeutic_response(result)

        print("ORIGINAL RESPONSE:")
        print(result)
        
        if isinstance(result, str):
            # Remove common artifacts
            artifacts = [
                "the text",
                "The text",
                "According to the text",
                "Based on the text",
                "From the text",
                "In the text",
                "corrupted file",
                "characters and symbols"
            ]
            
            # Clean up the beginning of the response
            for artifact in artifacts:
                if result.startswith(artifact):
                    result = result.replace(artifact, "", 1).lstrip(" ,:;.-")
                result = result.replace(f"\n{artifact}", "\n")

            # Function to remove specific characters from start and end only
            def strip_chars(text, chars):
                while text and text[0] in chars:
                    text = text[1:]
                while text and text[-1] in chars:
                    text = text[:-1]
                return text
                
            # Remove quotes and backticks only from start and end
            chars_to_strip = {'"', "'", '`', '"', '"'}  # Added smart quotes
            result = strip_chars(result, chars_to_strip)
            
            # Clean up triple quotes at start/end
            while result.startswith('"""'):
                result = result[3:]
            while result.endswith('"""'):
                result = result[:-3]
                
            # Clean up any double spaces or excessive newlines
            result = " ".join(result.split())
            result = "\n".join(line.strip() for line in result.split("\n") if line.strip())
            
            print("FORMATTED RESPONSE:")
            print(result)
            
        return {"result": result}

    def retrieval_chain(self, user_input: str) -> str:
        # classify intent
        classified_intent = self.classify_intent(user_input)

        # Handle the 3 possible workflows after classifying intent
        if classified_intent != "Other" and classified_intent != "Clinical_trial_recruitment":
            # Call RAG with the appropriate vector databases
            time.sleep(RATE_LIMIT_BREAK)
            relevant_collections = set()
            for collection in COLLECTIONS:
                relevant_collections.update(self.KEY_WORDS_DATA_MAPPING[collection])
            if classified_intent == "After_experience":
                relevant_collections.add("guidelines_integration")
            
            # Enhance the query to maintain therapeutic context
            therapeutic_query = f"""As a psychedelic-assisted therapist, I need to answer: {user_input}
            Please provide information that would be helpful in a therapeutic context. Use no more than {self.max_tokens} tokens."""
            
            self.qa_chain.retriever.with_collections(relevant_collections)
            raw_response = self.qa_chain.invoke({"query": therapeutic_query})
            return self.clean_and_format_rag_response(raw_response)
        elif classified_intent == "Clinical_trial_recruitment":
            time.sleep(RATE_LIMIT_BREAK)
            return self.handle_clinical_trial_search(user_input)
        else:
            return None