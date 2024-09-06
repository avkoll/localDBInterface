from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sqlite3

# Load the text-to-SQL model
model_name = "Salesforce/wikisql"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Convert natural language to SQL
query = "Show me the names of employees who earn more than 5000"
inputs = tokenizer.encode(query, return_tensors="pt")
outputs = model.generate(inputs)
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated SQL:", sql_query)

# Set up a database connection
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Execute the generated SQL
cursor.execute(sql_query)
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
