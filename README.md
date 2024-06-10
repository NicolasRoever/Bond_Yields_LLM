# LLM Program for Sentiment Detection

This repository contains code for the analysis of sentiment towards country solvency in earnings conference call transcripts of financial service companies. This analysis is based on a chain of LLM calls using DSPy as framwork

## Folder and file description

- Signatures: includes the LLm calls
- Config.py: sets up the LLM instances and select the model to be used for the calls
- Module.py: executes the LLM calls
- Data: contains hand-coded excerpt from the transcripts and preproccesses them
- Optimize.py: DSPy optimizer to select input/putput examples for prompts from hand-coded sample (NOT DONE YET) 