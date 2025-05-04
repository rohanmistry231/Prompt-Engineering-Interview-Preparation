# 🧠 Contextual Prompting

## 📖 Introduction

**Contextual Prompting** enhances large language model (LLM) outputs by adding relevant context to prompts, ensuring task-specific responses. In retail, contextual prompts improve accuracy for tasks like customer query answering or product recommendations. This guide explores contextual prompting with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Context Addition**: Include background info (e.g., product details, user profile).
- **Task Specificity**: Context aligns outputs with retail goals.
- **Retail Use Cases**: Query answering, recommendations, support.
- **Balancing Context**: Too much context increases latency.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_contextual_prompting_demo():
    # Synthetic Retail Query Data
    queries = [
        "What’s the best TechCorp product for a student?",
        "Recommend a TechCorp device for gaming.",
        "Suggest a TechCorp gadget for travel."
    ]
    context = "TechCorp products: Laptop (16GB RAM, Intel i7, 512GB SSD), Smartphone (long battery, vibrant display), Tablet (lightweight, 10-hour battery)."
    print("Synthetic Data: Retail queries and context created")
    print(f"Queries: {queries}")
    print(f"Context: {context}")

    # Contextual Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="You are a retail assistant. Use this context: {context}\nAnswer: {query}"
    )
    chain = llm | prompt
    
    responses = [chain.invoke({"context": context, "query": query}) for query in queries]
    print("Contextual Prompting: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response.strip()}")

    # Visualization
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(queries) + 1), response_lengths, color='blue')
    plt.title("Contextual Prompt Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.savefig("contextual_prompting_output.png")
    print("Visualization: Response lengths saved as contextual_prompting_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_contextual_prompting_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`contextual_prompting_output.png`) showing response word counts, illustrating the impact of contextual prompting.

## 💡 Retail Applications

- **Query Answering**: Context improves accuracy (e.g., "Use product specs to answer").
- **Recommendations**: Context ensures relevant suggestions (e.g., "Based on user profile").
- **Customer Support**: Context clarifies policies (e.g., "Use return policy details").

## 🏆 Practical Tasks

1. Add context to a retail query prompt.
2. Compare contextual vs. non-contextual prompt responses.
3. Visualize response lengths with and without context.

## 💡 Interview Scenarios

**Question**: How does contextual prompting improve LLM outputs?  
**Answer**: Contextual prompting provides background info, aligning outputs with task-specific goals, improving accuracy.  
**Key**: Context reduces ambiguity in retail tasks.  
**Example**: `PromptTemplate(template="Use this context: {context}\nAnswer: {query}")`

**Coding Task**: Design a contextual prompt for retail product recommendations.  
**Tip**: Use `PromptTemplate` with product details as context.

## 📚 Resources

- [LangChain Prompt Documentation](https://python.langchain.com/docs/modules/prompts/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/contextual-prompting`).
3. Commit changes (`git commit -m 'Add contextual prompting content'`).
4. Push to the branch (`git push origin feature/contextual-prompting`).
5. Open a Pull Request.