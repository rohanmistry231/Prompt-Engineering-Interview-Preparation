# 🆚 A/B Testing

## 📖 Introduction

**A/B Testing** compares prompt variations to identify the most effective for retail tasks like customer support or product descriptions. This guide explores A/B testing with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Prompt Variations**: Test different prompt structures or instructions.
- **Performance Comparison**: Evaluate using metrics like accuracy or relevance.
- **Retail Use Cases**: Optimize prompts for query answering, recommendations.
- **Iterative Improvement**: Select the best prompt based on results.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk rouge-score
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from rouge_score import rouge_scorer
import numpy as np

def run_ab_testing_demo():
    # Synthetic Retail Query Data
    queries = [
        "Describe the TechCorp laptop.",
        "Explain the TechCorp smartphone features.",
        "Detail the TechCorp tablet use case."
    ]
    references = [
        "The TechCorp laptop has 16GB RAM, Intel i7, 512GB SSD.",
        "The TechCorp smartphone offers a long-lasting battery with vibrant display.",
        "The TechCorp tablet is lightweight, ideal for students."
    ]
    print("Synthetic Data: Retail queries and references created")
    print(f"Queries: {queries}")

    # A/B Prompt Variations
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt_a = PromptTemplate(
        input_variables=["query"],
        template="You are a retail assistant. Answer: {query}"
    )
    prompt_b = PromptTemplate(
        input_variables=["query"],
        template="You are a retail assistant. Provide a concise bullet-point answer to: {query}"
    )
    chain_a = llm | prompt_a
    chain_b = llm | prompt_b
    
    responses_a = [chain_a.invoke({"query": query}) for query in queries]
    responses_b = [chain_b.invoke({"query": query}) for query in queries]
    
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_a = [scorer.score(ref, resp)['rouge1'].fmeasure for ref, resp in zip(references, responses_a)]
    rouge_b = [scorer.score(ref, resp)['rouge1'].fmeasure for ref, resp in zip(references, responses_b)]
    
    print("A/B Testing: ROUGE scores calculated")
    for i, (query, ra, rb) in enumerate(zip(queries, rouge_a, rouge_b)):
        print(f"Query {i+1}: {query}")
        print(f"Prompt A ROUGE-1: {ra:.2f}, Prompt B ROUGE-1: {rb:.2f}")

    # Visualization
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, rouge_a, 0.4, label='Prompt A', color='blue')
    plt.bar(x + 0.2, rouge_b, 0.4, label='Prompt B', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("A/B Testing: ROUGE-1 Scores")
    plt.xlabel("Query")
    plt.ylabel("ROUGE-1 Score")
    plt.legend()
    plt.savefig("ab_testing_output.png")
    print("Visualization: ROUGE scores saved as ab_testing_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_ab_testing_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`ab_testing_output.png`) comparing ROUGE-1 scores for two prompt variations, illustrating A/B testing results.

## 💡 Retail Applications

- **Customer Support**: Test prompts for query response quality.
- **Product Descriptions**: Compare prompts for description clarity.
- **Review Analysis**: Evaluate prompts for sentiment accuracy.

## 🏆 Practical Tasks

1. Compare two prompt variations for a retail task using ROUGE.
2. Test prompt variations for response relevance.
3. Visualize performance differences between prompts.

## 💡 Interview Scenarios

**Question**: How do you perform A/B testing for prompts?  
**Answer**: A/B testing compares prompt variations using metrics like ROUGE to identify the most effective for a task.  
**Key**: Iterative testing improves retail prompt performance.  
**Example**: `scorer.score(reference, response)['rouge1'].fmeasure`

**Coding Task**: Conduct A/B testing for retail prompt variations.  
**Tip**: Use `PromptTemplate` for two prompts and compare with ROUGE.

## 📚 Resources

- [ROUGE Documentation](https://github.com/pltrdy/rouge)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/ab-testing`).
3. Commit changes (`git commit -m 'Add A/B testing content'`).
4. Push to the branch (`git push origin feature/ab-testing`).
5. Open a Pull Request.