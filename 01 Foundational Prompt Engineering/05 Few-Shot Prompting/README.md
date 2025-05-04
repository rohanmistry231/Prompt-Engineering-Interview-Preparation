# 📚 Few-Shot Prompting

## 📖 Introduction

**Few-Shot Prompting** enhances large language model (LLM) performance by providing a small number of examples within the prompt, guiding the model to produce more accurate outputs. In retail, few-shot prompting is ideal for tasks like product recommendation or review summarization, where examples improve context. This guide explores how to design effective few-shot prompts, with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Example Inclusion**: Provide 1-5 examples to demonstrate the task.
- **Example Quality**: Examples must be relevant and representative.
- **Retail Use Cases**: Few-shot prompts excel in structured tasks (e.g., recommendations, summarization).
- **Balancing Examples**: Too many examples increase latency; too few reduce accuracy.

## 🛠️ Practical Example

Below is a Python example using LangChain for few-shot prompting to generate retail product recommendations.

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_few_shot_prompting_demo():
    # Synthetic Retail Query Data
    queries = [
        "Recommend a product for a gamer.",
        "Suggest a device for a student.",
        "Propose a gadget for a professional."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # Few-Shot Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are a retail assistant. Recommend a product based on the user query. Examples:
        - Query: Recommend a product for a photographer.
          Recommendation: TechCorp Camera with high-resolution lens.
        - Query: Suggest a device for a traveler.
          Recommendation: TechCorp Smartphone with long battery life.
        Answer the query: {query}"""
    )
    chain = llm | prompt
    
    responses = [chain.invoke({"query": query}) for query in queries]
    print("Few-Shot Prompting: Recommendations generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Recommendation: {response.strip()}")

    # Visualization
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(queries) + 1), response_lengths, color='orange')
    plt.title("Few-Shot Recommendation Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.savefig("few_shot_prompting_output.png")
    print("Visualization: Recommendation lengths saved as few_shot_prompting_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_few_shot_prompting_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`few_shot_prompting_output.png`) showing the word count of recommendation responses, illustrating the impact of few-shot prompting.

## 💡 Retail Applications

- **Product Recommendations**: Few-shot prompts guide LLMs to suggest relevant products (e.g., "Recommend a device for a student").
- **Review Summarization**: Examples help summarize reviews accurately (e.g., "Summarize this review in 3 points").
- **Customer Support**: Examples ensure consistent query responses (e.g., "Answer this query like the examples").

## 🏆 Practical Tasks

1. Design a few-shot prompt for retail product recommendations.
2. Summarize retail reviews using few-shot prompting.
3. Visualize response lengths to compare few-shot vs. zero-shot prompting.

## 💡 Interview Scenarios

**Question**: How does few-shot prompting differ from zero-shot prompting?  
**Answer**: Few-shot prompting includes examples to guide LLMs, improving accuracy for specific tasks, while zero-shot relies on instructions alone.  
**Key**: Few-shot is better for tasks needing context; zero-shot is faster but less precise.  
**Example**: `PromptTemplate(template="Examples: Query: ... Recommendation: ... Answer: {query}")`

**Coding Task**: Create a few-shot prompt for summarizing retail reviews in 3 bullet points.  
**Tip**: Use `PromptTemplate` with 2-3 example summaries and test with retail reviews.

## 📚 Resources

- [LangChain Prompt Documentation](https://python.langchain.com/docs/modules/prompts/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/techniques/fewshot)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/few-shot-prompting`).
3. Commit changes (`git commit -m 'Add few-shot prompting content'`).
4. Push to the branch (`git push origin feature/few-shot-prompting`).
5. Open a Pull Request.