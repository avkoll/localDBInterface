from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a custom PAD token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to account for the new token

def natural_language_to_sql(query, model, tokenizer, max_length=128):
    """
    Convert a natural language query to SQL using a pre-trained model.

    Args:
    - query (str): The natural language query to be converted to SQL.
    - model: The pre-trained text-to-SQL model.
    - tokenizer: The tokenizer corresponding to the pre-trained model.
    - max_length (int): Maximum length for the generated SQL query. Default is 128.

    Returns:
    - sql_query (str): The generated SQL query.
    """
    try:
        # Tokenize the input query
        inputs = tokenizer.encode_plus(query, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        print(f"Tokenized input: {input_ids}")
        print(f"Attention mask: {attention_mask}")

        # Generate the SQL query using the model, passing the attention_mask as well
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=6, early_stopping=True)
        print(f"Model raw output: {outputs}")

        # Decode the generated SQL query
        sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded SQL Query: {sql_query}")

        return sql_query
    except Exception as e:
        print(f"Error during SQL generation: {e}")
        return None


# Example usage
nl_query = "Generate SQL Query for: " + input("Enter a natural language query: ")
print(f"Input Query: {nl_query}")
sql_query = natural_language_to_sql(nl_query, model, tokenizer)

if sql_query:
    print("Generated SQL Query:", sql_query)
else:
    print("Failed to generate SQL query.")
