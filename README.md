# CAPT: Conversational Agent for Psychedelic-assisted Therapy
An LLM-powered conversational agent designed to enhance psychedelic therapy through preparation, real-time harm reduction, and post-session integration support in a simulated environment.

![CAPT Demo](CAPTDemo-ezgif.com-video-to-gif-converter.gif)

## Features
- Conversational AI interface for therapy support
- Integration with LangChain for advanced language model capabilities
- FastAPI backend for efficient API handling
- Evaluation framework for all components

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see `config.py`) and run `app/vectorstore/ingestion/ingest.py` OR download vectorstore data from [here](https://drive.google.com/drive/folders/1tr58rSSOrSb2ByypsnJ3wJ_YApLiRUGb?usp=sharing). Ensure that the directory structure looks like:
```
|--app
|----vectorstore
|------store
|--------articles
|--------guidelines_mdma
|--------guidelines_LSD
|--------guidelines_general
|--------...

```

## Usage

### Running CAPT locally
Run the application with: `python main.py`

### Running evalutions
Run any evaluation script in the evaluation directory like so: `python -m evaluation.RAG.RAG_evaluation`

