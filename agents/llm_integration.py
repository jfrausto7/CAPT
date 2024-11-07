from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import re
import json

from agents.retrieval.MultiVectorstoreRetriever import MultiVectorstoreRetriever
from agents.TherapyAgent import TherapyAgent
from app.clinical_trial_search import search_clinical_trials

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
        
        Example output:
        {{"drug": "MDMA", "location": "California", "condition": "PTSD"}}
        """
        
        # Use the existing LLM to extract information
        response = self.intent_model.invoke(extraction_prompt).strip()
        
        try:
            # Parse the JSON response
            search_params = json.loads(response)
            
            # Ensure we have at least a drug name
            if not search_params.get("drug"):
                return "I couldn't determine which psychedelic medication you're interested in. Could you please specify?"
            
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
                return f"No active clinical trials found for {search_params['drug']} in {location}."
            
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
            
            return response
            
        except json.JSONDecodeError:
            return "I had trouble processing the clinical trial search. Could you please rephrase your request?"
        except Exception as e:
            return f"An error occurred while searching for clinical trials: {str(e)}"

    async def complete(self, user_input: str) -> str:
        global COLLECTIONS
        # Set up prompt with user input
        self.create_prompt_template(user_input)
        print("PROMPT:")
        print(self.prompt)

        # Step 1: search for keywords and update global variable array
        # first make the user input all lowercase, then search for key words using regex
        user_input = user_input.lower()
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in KEY_WORDS) + r')\b'
        COLLECTIONS = re.findall(pattern, user_input)

        # Inside the retrieval chain the 3 workflows are accounted for
        return self.retrieval_chain(user_input)

    def __init__(self, model: str, temperature: int, max_tokens: int):
        # Set up LLM agent
        self.intent_model = Together(model=model, temperature=temperature, max_tokens=max_tokens)

    def create_prompt_template(self, user_input: str) -> PromptTemplate:
        intent_classifier_template = f"""
        You are an intention classifier. You will be provided with a user input inquiring about 
        psychedelic drugs and clinical trials. A user's input can fall into the following intentions {self.INTENTIONS}.
        Your task is to classify the user's input into ONLY one of the following intents: {self.INTENTIONS}.
        
        User input: {user_input}
    
        intention: 
        """
        self.prompt = PromptTemplate(
            input_variables=["user_input"],
            template=intent_classifier_template
        )
        return self.prompt

    def classify_intent(self, user_input: str) -> str:
        self.intent_chain = self.prompt | self.intent_model

        classified_intent = self.intent_model.invoke(user_input).strip()
        print("INTENT IS:")
        print(classified_intent)

        for intent in self.INTENTIONS:
            # checking if the intent is in the LLM's response (which could be 64 tokens long)
            if intent in classified_intent:
                return intent
        # If no intent is found default to Other
        return "Other"

    def create_RAG_retrieval_chain(
        self,
        vector_dir: str,
        model,
        show_progress: bool = True
    ):
        # Initialize retriever
        retriever = MultiVectorstoreRetriever(
            vector_dir=vector_dir,
            k=4,  # Number of documents to retrieve per query
            score_threshold=0.7,  # Minimum similarity score
            show_progress=show_progress
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",  # or "map_reduce" for longer contexts
            retriever=retriever,
            return_source_documents=True  # Include source docs in response
        )

        return qa_chain

    def retrieval_chain(self, user_input: str) -> str:
        # classify intent
        classified_intent = self.classify_intent(user_input)

        # initilaize a new llm model for the RAG
        model = Together(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            temperature=0.3,
            max_tokens=256
        )

        # Initialize RAG qa chain
        qa_chain = self.create_RAG_retrieval_chain(
            vector_dir="app/vectorstore/store",
            model=model
            )
        # Handle the 3 possible workflows after classifying intent
        # If COLLECTIONS is populated, then we have key words listed in user input so call RAG
        if classified_intent != "Other" and classified_intent != "Clinical_trial_recruitment":
            # Call RAG with the all the appropriate vector databases based on user's key words and key word mapping
            relevant_collections = set()
            for collection in COLLECTIONS:
                relevant_collections.update(self.KEY_WORDS_DATA_MAPPING[collection])
            qa_chain.retriever.with_collections(relevant_collections)
            return qa_chain.invoke({"user_input": user_input})
        elif classified_intent == "Clinical_trial_recruitment":
            return self.handle_clinical_trial_search(user_input)
        else:
            # Call LLM Therapy Agent if classified intent is "Other"
            return TherapyAgent