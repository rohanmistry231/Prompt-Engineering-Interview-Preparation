# 🔄 Dynamic Prompting

## 📖 Introduction

**Dynamic Prompting** adapts prompts based on input variables, enabling flexible LLM responses for varying retail tasks. This guide explores dynamic prompting with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Variable Adaptation**: Prompts change based on inputs (e.g., user role, task type).
- **Flexibility**: Handles diverse retail queries dynamically.
- **Retail Use Cases**: Customer support, product queries, reviews.
- **Efficiency**: Reduces need for multiple static prompts.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_dynamic_prompting_demo():
    # Synthetic Retail Query Data
    queries = [
        {"task": "describe", "product": "TechCorp Laptop"},
        {"task": "recommend", "product": "TechCorp Smartphone"},
        {"task": "summarize", "product": "TechCorp Tablet"}
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # Dynamic Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["task", "product"],
        template="You are a retail assistant. {task} the {product} based on this context: Laptop (16GB RAM, Intel i7), Smartphone (long battery), Tablet (lightweight)."
    )
    chain = llm | prompt
    
    responses = [chain.invoke({"task": q["task"], "product": q["product"]}) for q in queries]
    print("Dynamic Prompting: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: Task={query['task']}, Product={query['product']}")
        print(f"Response: {response.strip()}")

    # Visualization
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar([q["task"] for q in queries], response_lengths, color='green')
    plt.title("Dynamic Prompt Response Lengths")
    plt.xlabel("Task")
    plt.ylabel("Word Count")
    plt.savefig("dynamic_prompting_output.png")
    print("Visualization: Response lengths saved as dynamic_prompting_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_dynamic_prompting_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`dynamic_prompting_output.png`) showing response word counts by task, illustrating dynamic prompting’s flexibility.

## 💡 Retail Applications

- **Customer Support**: Adapt prompts for query types (e.g., describe, recommend).
- **Product Queries**: Dynamic prompts handle varying product details.
- **Review Analysis**: Summarize or classify based on input variables.

## 🏆 Practical Tasks

1. Create a dynamic prompt for retail tasks (e.g., describe, recommend).
2. Test prompt adaptability across query types.
3. Visualize response lengths for different tasks.

## 💡 Interview Scenarios

**Question**: What’s dynamic prompting, and when is it used?  
**Answer**: Dynamic prompting adapts prompts based on input variables, enabling flexible responses for diverse tasks.  
**Key**: Useful for varying retail queries without multiple prompts.  
**Example**: `PromptTemplate(template="{task} the {product}...")`

**Coding Task**: Design a dynamic prompt for retail query handling.  
**Tip**: Use `PromptTemplate` with variables like task and product.

## 📚 Resources

- [LangChain Prompt Documentation](https://python.langchain.com/docs/modules/prompts/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/dynamic-prompting`).
3. Commit changes (`git commit -m 'Add dynamic prompting content'`).
4. Push to the branch (`git push origin feature/dynamic-prompting`).
5. Open a Pull Request.