NL2SQL app. This is meant to convert natural language to sql using llms that run locally with llama.cpp.
There are multiple LLMs that generate, verify, and compare with the database schema to make sure the generated SQL is valid and would work on the target database.
Each LLM runs in its own docker container and interacts with a flask API to send and receive prompts and responses.

CUDA 11.8
