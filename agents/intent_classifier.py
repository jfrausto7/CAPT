from langchain_together import Together
from langchain.prompts import PromptTemplate

class IntentClassifier:
    # Define intentions
    INTENTIONS = ["Other", "General_knowledge", "Before_experience",
                  "During_experience", "After_experience", "Clinical_trial_recruitment"]

    def __init__(self, model: str, temperature: int, max_tokens: int):
        # Set up LLM agent and retrieval agent
        self.intent_model = Together(model=model, temperature=temperature, max_tokens=max_tokens)
        self.prompt = None  # Initialize the prompt attribute

    def create_prompt_template(self, user_input: str) -> PromptTemplate:
        template = """
               You are an intention classifier. You will be provided with a user input inquiring about psychedelic drugs and clinical trials.

               Your task is to classify the user's input into EXACTLY one of the following intents: {intentions}. 

               Intent Definitions:
                - General_knowledge: Questions about basic facts, history, effects, or general information about psychedelics
                - Before_experience: Questions about preparation, dosing, safety measures, or what to expect before using psychedelics
                - During_experience: Questions about managing, navigating, or understanding experiences while under the influence
                - After_experience: Questions about integration, reflection, or dealing with after-effects
                - Clinical_trial_recruitment: Questions about participating in, finding, or qualifying for clinical trials
                - Other: ONLY use if the input absolutely cannot fit into any other category
            
                Example Classifications:
                "What is psilocybin?" → General_knowledge
                "How should I prepare for my first mushroom journey?" → Before_experience
                "I'm feeling overwhelmed right now, what should I do?" → During_experience
                "How do I integrate my insights from yesterday's session?" → After_experience
                "Where can I sign up for MDMA therapy trials?" → Clinical_trial_recruitment
                "Can you recommend a good pizza place?" → Other
            
                Guidelines:
                1. Always prefer a specific category over "Other" if there's any reasonable connection
                2. Consider the temporal context (before/during/after) when classifying
                3. If a question touches multiple categories, choose the most prominent one
                4. "Other" should only be used for completely unrelated topics 
                
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
        # Create a prompt and assign it to the instance attribute
        self.prompt = PromptTemplate.from_template(template)
        self.prompt.format(user_input=user_input, intentions=self.INTENTIONS)

        return self.prompt

    def classify_intent(self, user_input: str) -> str:
        # Ensure the prompt is created before using it
        self.create_prompt_template(user_input)

        # Process classification using the prompt and intent model
        self.intent_chain = self.prompt | self.intent_model
        classified_intent = \
            self.intent_chain.invoke({"user_input": user_input, "intentions": self.INTENTIONS}).strip().split(' ', 1)[0]

        for intent in self.INTENTIONS:
            # checking if the intent is in the LLM's response (which could be 64 tokens long)
            if intent in classified_intent:
                return intent
        # If no intent is found, default to Other
        return "Other"
