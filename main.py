from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Use the valid model from Hugging Face
model_name = "cssupport/t5-small-awesome-text-to-sql"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


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
        inputs = tokenizer.encode(query, return_tensors="pt", truncation=True, max_length=max_length)

        # Generate the SQL query using the model
        outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)

        # Decode the generated SQL query
        sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return sql_query
    except Exception as e:
        print(f"Error during SQL generation: {e}")
        return None


# Example usage
nl_query = "Show me the names of employees who earn more than 5000"

# Call the function and print the SQL query
sql_query = natural_language_to_sql(nl_query, model, tokenizer)

if sql_query:
    print("Generated SQL Query:", sql_query)
else:
    print("Failed to generate SQL query.")
