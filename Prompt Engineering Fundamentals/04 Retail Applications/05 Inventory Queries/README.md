# 📦 Inventory Queries

## 📖 Introduction

**Inventory Queries** use prompts to answer retail stock and pricing questions, streamlining operations. This guide explores inventory query prompts with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Stock Queries**: Check product availability.
- **Pricing Queries**: Provide accurate price information.
- **Retail Use Cases**: Inventory management, customer support.
- **Accuracy**: Ensure reliable stock and price answers.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_inventory_queries_demo():
    # Synthetic Retail Query Data
    queries = [
        "Is the TechCorp laptop in stock?",
        "What’s the price of the TechCorp smartphone?",
        "How many TechCorp tablets are available?"
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # Inventory Query Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["query"],
        template="You are a retail inventory manager. Answer the query concisely using this context: TechCorp Laptop (10 in stock, $999), Smartphone (5 in stock, $699), Tablet (20 in stock, $499): {query}"
    )
    chain = llm | prompt
    
    responses = [chain.invoke({"query": query}) for query in queries]
    print("Inventory Queries: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response.strip()}")

    # Visualization
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(queries) + 1), response_lengths, color='red')
    plt.title("Inventory Query Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.savefig("inventory_queries_output.png")
    print("Visualization: Response lengths saved as inventory_queries_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_inventory_queries_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`inventory_queries_output.png`) showing response word counts, illustrating prompt efficiency.

## 💡 Retail Applications

- **Stock Checks**: Confirm product availability for customers.
- **Pricing Queries**: Provide accurate price details.
- **Inventory Management**: Support stock tracking and updates.

## 🏆 Practical Tasks

1. Design a prompt for retail inventory queries.
2. Answer stock and pricing questions with prompts.
3. Visualize response lengths for inventory queries.

## 💡 Interview Scenarios

**Question**: How do you handle inventory queries with prompts?  
**Answer**: Prompts use context like stock and price data to provide accurate, concise answers.  
**Key**: Ensures reliable retail inventory responses.  
**Example**: `PromptTemplate(template="Answer using context: {context}: {query}")`

**Coding Task**: Create a prompt for retail inventory queries.  
**Tip**: Use `PromptTemplate` with stock and price context.

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/inventory-queries`).
3. Commit changes (`git commit -m 'Add inventory queries content'`).
4. Push to the branch (`git push origin feature/inventory-queries`).
5. Open a Pull Request.