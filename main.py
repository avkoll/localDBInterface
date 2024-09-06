from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Salesforce/wikisql"
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def natural_language_to_sql(query):
    inputs = tokenizer.encode(query, return_tensors="pt")
    outputs = model.generate(inputs)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query


nl_query = "Show me the names of employees who earn more than 5000"
sql_query = natural_language_to_sql(nl_query)
print("Generated SQL Query:", sql_query)
