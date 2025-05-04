# 🔗 Prompt Chaining

## 📖 Introduction

**Prompt Chaining** uses sequential prompts to handle complex retail tasks, breaking them into manageable steps. This guide explores prompt chaining with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Sequential Prompts**: Chain prompts to process tasks step-by-step.
- **Complex Tasks**: Handle multi-step retail workflows (e.g., query analysis, recommendation).
- **Retail Use Cases**: Customer support, product analysis.
- **Modularity**: Each prompt focuses on a specific sub-task.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_prompt_chaining_demo():
    # Synthetic Retail Query Data
    queries = [
        "Analyze and recommend a TechCorp product for a student.",
        "Evaluate and suggest a TechCorp device for gaming.",
        "Assess and propose a TechCorp gadget for travel."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # Prompt Chaining
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    # Chain 1: Analyze
    analyze_prompt = PromptTemplate(
        input_variables=["query"],
        template="You are a retail analyst. Analyze the query and list key requirements: {query}"
    )
    analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt, output_key="analysis")
    
    # Chain 2: Recommend
    recommend_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="Based on the analysis: {analysis}, recommend a TechCorp product (Laptop, Smartphone, Tablet)."
    )
    recommend_chain = LLMChain(llm=llm, prompt=recommend_prompt, output_key="recommendation")
    
    chain = SequentialChain(
        chains=[analyze_chain, recommend_chain],
        input_variables=["query"],
        output_variables=["analysis", "recommendation"]
    )
    
    responses = [chain({"query": query}) for query in queries]
    print("Prompt Chaining: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Analysis: {response['analysis'].strip()}")
        print(f"Recommendation: {response['recommendation'].strip()}")

    # Visualization
    analysis_lengths = [len(nltk.word_tokenize(resp["analysis"])) for resp in responses]
    recommendation_lengths = [len(nltk.word_tokenize(resp["recommendation"])) for resp in responses]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, analysis_lengths, 0.4, label='Analysis Length', color='blue')
    plt.bar(x + 0.2, recommendation_lengths, 0.4, label='Recommendation Length', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("Prompt Chaining Output Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("prompt_chaining_output.png")
    print("Visualization: Output lengths saved as prompt_chaining_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_prompt_chaining_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`prompt_chaining_output.png`) comparing analysis and recommendation word counts, illustrating prompt chaining’s workflow.

## 💡 Retail Applications

- **Customer Support**: Chain prompts for query analysis and response.
- **Product Recommendations**: Analyze needs, then recommend products.
- **Review Analysis**: Extract insights, then summarize.

## 🏆 Practical Tasks

1. Build a prompt chain for a retail customer support workflow.
2. Compare chained vs. single-prompt responses.
3. Visualize output lengths for chained prompts.

## 💡 Interview Scenarios

**Question**: How do you implement prompt chaining for complex tasks?  
**Answer**: Prompt chaining uses sequential prompts to break complex tasks into steps, improving modularity and accuracy.  
**Key**: Ideal for multi-step retail workflows.  
**Example**: `SequentialChain(chains=[LLMChain(...), LLMChain(...)], ...)`

**Coding Task**: Create a prompt chain for retail query analysis and recommendation.  
**Tip**: Use `SequentialChain` with multiple `LLMChain` instances.

## 📚 Resources

- [LangChain Prompt Documentation](https://python.langchain.com/docs/modules/prompts/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/prompt-chaining`).
3. Commit changes (`git commit -m 'Add prompt chaining content'`).
4. Push to the branch (`git push origin feature/prompt-chaining`).
5. Open a Pull Request.