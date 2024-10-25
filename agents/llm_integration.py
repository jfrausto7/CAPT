# import libraries
import os
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Access API key
API_KEY = os.getenv("TOGETHER_API_KEY")

# Write an intention classifier here
def intention_classifier(user_input) -> str:
    # Initialize separate llm model to handle intent classification
    intent_model = Together(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key=API_KEY,
        temperature = 0.7,
        max_tokens=64 # adjust this value for length of response
    )
    # Define possible intentions
    intentions = ["other", "general knowledge", "before experience", "during experience", "after experience"]

    # Specific prompt template for the classifier
    intent_classifier_template = f"""
    You are an intention classifier. You will be provided with a user input inquiring about 
    psychedelic drugs and clinical trials. Your task is to classify the user's input into one
    of the following intents: {intentions}.
    Please only respond with the intent label, don't add any more information.
    
    User input: {user_input}
    
    intention: 
    """
    # Create prompt
    prompt = PromptTemplate(
        input_variables = ["user_input"],
        template=intent_classifier_template
    )
    # Create LLM chain for classifier (DO I even need a chain here?)
    intent_chain = LLMChain(llm=intent_model, prompt=prompt)
    # Run the LLM on user input
    classified_intent = intent_chain.run({"user_input": user_input})
    # Return intent of user
    return classified_intent.strip()

# initialize LLM model
model = Together(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key=API_KEY,
    temperature = 0.7,
    max_tokens=512
)

# Create a general prompt
prompt = PromptTemplate(
    input_variables = ["user_input"],
    template = "Respond to this prompt: {user_input}",
)

#Create an LLM chain
model_chain = LLMChain(
    llm=model,
    prompt=prompt
)

# Create a loop to continuously answer prompts from the user
while True:
    # Get user's input
    user_input = input("Enter your prompt (or 'exit' to quit): ")
    # If user enters 'exit' we break
    if user_input.lower() == "exit":
        print("Exiting")
        break

    # Get an intitial response to user input
    intent = intention_classifier(user_input)
    print(f"Classified intent: {intent}")

    # Workflow after intent is classified. Call RAG, LLM, or fall back response based on intent
    """if intent == "other": -> fall back response 
        if intent == "general knowledge" -> retrieve with RAG
        if intent == "before experience" -> retrieve with RAG
        if intent == "during experience" -> stretch goal
        if intent == "after experience" -> retrieve with RAG
        """
    # combine response with intent retrieval and model's response
    #response = model_chain.run(user_input)
    # Run the initial response through the intention classifier to identify intent

    # Based on intent, use the RAG model to retrieve data from specific vector data base relevant to the intent

    # Print final response which includes initial response and data retrieved from RAG
    #print(f"Model response: {response}")
