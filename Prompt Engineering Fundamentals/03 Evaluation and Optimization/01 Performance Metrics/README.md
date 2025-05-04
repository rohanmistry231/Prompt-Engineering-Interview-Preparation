# 📈 Performance Metrics

## 📖 Introduction

**Performance Metrics** like BLEU, ROUGE, and custom metrics evaluate prompt quality for retail tasks such as query answering and review summarization. This guide explores how to measure prompt performance with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **BLEU**: Measures n-gram overlap for text similarity.
- **ROUGE**: Evaluates recall and precision for text generation.
- **Custom Metrics**: Retail-specific metrics (e.g., response relevance).
- **Retail Use Cases**: Assessing prompt accuracy for customer support.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk rouge-score
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

def run_performance_metrics_demo():
    # Synthetic Retail Query and Reference Data
    queries = [
        "Describe the TechCorp laptop features.",
        "Explain the TechCorp smartphone battery.",
        "Detail the TechCorp tablet use case."
    ]
    references = [
        "The TechCorp laptop has 16GB RAM, Intel i7, and 512GB SSD.",
        "The TechCorp smartphone offers a long-lasting battery with vibrant display.",
        "The TechCorp tablet is lightweight, ideal for students and professionals."
    ]
    print("Synthetic Data: Retail queries and references created")
    print(f"Queries: {queries}")

    # Generate Responses
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    responses = [llm(query) for query in queries]
    
    bleu_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for ref, resp in zip(references, responses):
        # BLEU Score
        ref_tokens = [ref.split()]
        resp_tokens = resp.split()
        bleu = sentence_bleu(ref_tokens, resp_tokens)
        bleu_scores.append(bleu)
        
        # ROUGE Score
        rouge = scorer.score(ref, resp)
        rouge_scores.append(rouge['rouge1'].fmeasure)
    
    print("Performance Metrics: Scores calculated")
    for i, (query, resp, bleu, rouge) in enumerate(zip(queries, responses, bleu_scores, rouge_scores)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {resp.strip()}")
        print(f"BLEU: {bleu:.2f}, ROUGE-1: {rouge:.2f}")

    # Visualization
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, bleu_scores, 0.4, label='BLEU', color='blue')
    plt.bar(x + 0.2, rouge_scores, 0.4, label='ROUGE-1', color='red')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("BLEU and ROUGE Scores")
    plt.xlabel("Query")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("performance_metrics_output.png")
    print("Visualization: Scores saved as performance_metrics_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_performance_metrics_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`performance_metrics_output.png`) comparing BLEU and ROUGE-1 scores for each query, illustrating prompt performance.

## 💡 Retail Applications

- **Query Answering**: Evaluate response accuracy for customer queries.
- **Review Summarization**: Assess summary quality against reference texts.
- **Product Descriptions**: Measure description relevance and completeness.

## 🏆 Practical Tasks

1. Calculate BLEU and ROUGE for a retail prompt’s responses.
2. Define a custom metric for retail response relevance.
3. Visualize metric scores for multiple prompts.

## 💡 Interview Scenarios

**Question**: What are BLEU and ROUGE metrics used for?  
**Answer**: BLEU measures n-gram overlap, and ROUGE evaluates recall/precision, both assessing prompt response quality.  
**Key**: Useful for retail response evaluation.  
**Example**: `sentence_bleu([ref.split()], resp.split())`

**Coding Task**: Calculate ROUGE for a retail prompt’s output.  
**Tip**: Use `rouge_scorer.RougeScorer` with retail references.

## 📚 Resources

- [ROUGE Documentation](https://github.com/pltrdy/rouge)
- [NLTK BLEU Documentation](https://www.nltk.org/api/nltk.translate.bleu_score.html)
- [LangChain Documentation](https://python.langchain.com/docs/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/performance-metrics`).
3. Commit changes (`git commit -m 'Add performance metrics content'`).
4. Push to the branch (`git push origin feature/performance-metrics`).
5. Open a Pull Request.