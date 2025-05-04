# 📋 Input Formatting

## 📖 Introduction

**Input Formatting** involves structuring inputs for large language models (LLMs) to ensure consistent and accurate outputs. Formats like JSON, text, or key-value pairs help LLMs interpret inputs correctly, especially for retail tasks like query answering or product categorization. This guide explores how to format inputs effectively, with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Structured Formats**: Use JSON, CSV, or key-value pairs for clarity.
- **Consistency**: Standardized inputs reduce LLM confusion.
- **Retail-Specific Formatting**: Structure inputs for retail data (e.g., product details, queries).
- **Error Handling**: Validate input formats to avoid misinterpretation.

## 🛠️ Practical Example

Below is a Python example using LangChain to format retail query inputs as JSON.

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import json
import numpy as np
import nltk

def run_input_formatting_demo():
    # Synthetic Retail Query Data
    queries = [
        {"product": "TechCorp Laptop", "question": "What are its features?"},
        {"product": "TechCorp Smartphone", "question": "How long is the battery life?"},
        {"product": "TechCorp Tablet", "question": "Is it good for students?"}
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # JSON-Formatted Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["input_json"],
        template="You are a retail assistant. Answer the query in the JSON input: {input_json}"
    )
    chain = llm | prompt
    
    responses = [chain.invoke({"input_json": json.dumps(query)}) for query in queries]
    print("Input Formatting: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {json.dumps(query)}")
        print(f"Response: {response.strip()}")

    # Visualization
    json_lengths = [len(json.dumps(query)) for query in queries]
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, json_lengths, 0.4, label='JSON Input Length', color='blue')
    plt.bar(x + 0.2, response_lengths, 0.4, label='Response Length', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("Input JSON vs Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Length")
    plt.legend()
    plt.savefig("input_formatting_output.png")
    print("Visualization: Input and response lengths saved as input_formatting_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_input_formatting_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`input_formatting_output.png`) comparing the character length of JSON inputs to the word count of responses, showing the impact of input formatting.

## 💡 Retail Applications

- **Query Answering**: JSON inputs structure queries (e.g., `{"product": "TechCorp Laptop", "question": "Features"}`).
- **Product Categorization**: Key-value pairs categorize products (e.g., `{"product": "TechCorp Tablet", "category": "Education"}`).
- **Customer Support**: Structured inputs ensure consistent support responses (e.g., `{"query": "Return policy", "product": "TechCorp Smartphone"}`).

## 🏆 Practical Tasks

1. Format retail queries as JSON inputs for an LLM.
2. Compare responses from JSON vs. plain text inputs.
3. Visualize the impact of input formatting on response length.

## 💡 Interview Scenarios

**Question**: Why is input formatting important for LLMs?  
**Answer**: Input formatting ensures LLMs interpret data consistently, reducing errors and improving output relevance, especially for structured tasks.  
**Key**: Formats like JSON provide clarity and standardization.  
**Example**: `json.dumps({"product": "TechCorp Laptop", "question": "Features"})`

**Coding Task**: Format a retail query as a JSON input and design a prompt to process it.  
**Tip**: Use `PromptTemplate` with `json.dumps` and test with retail queries.

## 📚 Resources

- [LangChain Prompt Documentation](https://python.langchain.com/docs/modules/prompts/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [JSON Documentation](https://www.json.org/json-en.html)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/input-formatting`).
3. Commit changes (`git commit -m 'Add input formatting content'`).
4. Push to the branch (`git push origin feature/input-formatting`).
5. Open a Pull Request.