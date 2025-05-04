# 🔁 Self-Consistency

## 📖 Introduction

**Self-Consistency** generates multiple outputs for a prompt and selects the most consistent response, improving reliability for retail tasks like sentiment analysis or query answering. This guide explores self-consistency with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Multiple Outputs**: Generate several responses for the same prompt.
- **Consistency Check**: Select the most frequent or reliable output.
- **Retail Use Cases**: Sentiment analysis, query validation.
- **Reliability**: Reduces variability in LLM outputs.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from collections import Counter
import numpy as np
import nltk

def run_self_consistency_demo():
    # Synthetic Retail Review Data
    reviews = [
        "TechCorp laptop is fast and reliable!",
        "TechCorp smartphone has poor battery life.",
        "TechCorp tablet is okay, app selection limited."
    ]
    print("Synthetic Data: Retail reviews created")
    print(f"Reviews: {reviews}")

    # Self-Consistency Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["review"],
        template="You are a retail analyst. Classify the sentiment of this review as Positive, Negative, or Neutral: {review}"
    )
    chain = llm | prompt
    
    # Generate multiple outputs (3 runs per review)
    responses = [[chain.invoke({"review": review}).strip() for _ in range(3)] for review in reviews]
    final_sentiments = [Counter(resp).most_common(1)[0][0] for resp in responses]
    print("Self-Consistency: Sentiments classified")
    for i, (review, sentiment) in enumerate(zip(reviews, final_sentiments)):
        print(f"Review {i+1}: {review}")
        print(f"Final Sentiment: {sentiment}")

    # Visualization
    consistency_scores = [Counter(resp).most_common(1)[0][1] / 3 for resp in responses]  # Consistency ratio
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(reviews) + 1), consistency_scores, color='red')
    plt.title("Self-Consistency Scores")
    plt.xlabel("Review")
    plt.ylabel("Consistency Ratio")
    plt.savefig("self_consistency_output.png")
    print("Visualization: Consistency scores saved as self_consistency_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_self_consistency_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`self_consistency_output.png`) showing consistency ratios for sentiment classifications, illustrating self-consistency’s reliability.

## 💡 Retail Applications

- **Sentiment Analysis**: Ensure consistent review classifications.
- **Query Answering**: Validate responses for accuracy.
- **Product Categorization**: Confirm consistent category assignments.

## 🏆 Practical Tasks

1. Generate multiple outputs for a retail review sentiment task.
2. Select the most consistent response using self-consistency.
3. Visualize consistency scores across reviews.

## 💡 Interview Scenarios

**Question**: What is self-consistency in prompt engineering?  
**Answer**: Self-consistency generates multiple outputs for a prompt and selects the most frequent, improving reliability.  
**Key**: Useful for retail tasks needing stable outputs.  
**Example**: `Counter(responses).most_common(1)[0][0]`

**Coding Task**: Implement self-consistency for retail sentiment analysis.  
**Tip**: Use `PromptTemplate` and generate multiple outputs.

## 📚 Resources

- [LangChain Prompt Documentation](https://python.langchain.com/docs/modules/prompts/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/self-consistency`).
3. Commit changes (`git commit -m 'Add self-consistency content'`).
4. Push to the branch (`git push origin feature/self-consistency`).
5. Open a Pull Request.