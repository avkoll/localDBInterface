The goal of this project is to create a NL2SQL LLM that can talk to a sql database. It will elminate the need to learn to write in SQL. 
This will of course be more complicated than I think, but I want this to follow the architecture below but replace the 
GPT stuff with local models. 
![Architecture](static/Architecture%202.png)

CUDA 11.8

# Models
### tscholak/cxmefzzi
This model is the T5-3B trained on the spider database and is fin tuned to be used something called PICARD \
I want to use this model and fine tune it on whatever database I will be 'talking' to. 

### meta-llama/Meta-Llama-3.1-8B-Instruct
This is llama, a bunch of things I read said this was one of the best to run locally and can run on a consumer pc \
I want to use the larger models but I only have a 2070 and don't think I would be able to run them. I need to save for a new pc \
das