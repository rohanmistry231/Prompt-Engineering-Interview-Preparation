# 📊 Review Analysis

## 📖 Introduction

**Review Analysis** uses prompts to extract sentiment and insights from retail reviews, informing business decisions. This guide explores review analysis prompts with hands-on examples and visualizations, tailored to retail scenarios (April 26, 2025).

## 🌟 Key Concepts

- **Sentiment Extraction**: Classify reviews as positive, negative, or neutral.
- **Insight Generation**: Identify key themes or issues.
- **Retail Use Cases**: Product feedback, customer satisfaction.
- **Accuracy**: Ensure reliable sentiment and insight extraction.

## 🛠️ Practical Example

```python
# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from collections import Counter
import numpy as np
import nltk

def run_review_analysis_demo():
    # Synthetic Retail Review Data
    reviews = [
        "TechCorp laptop is fast and reliable!",
        "TechCorp smartphone has poor battery life.",
        "TechCorp tablet is okay, app selection limited."
    ]
    print("Synthetic Data: Retail reviews created")
    print(f"Reviews: {reviews}")

    # Review Analysis Prompt
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["review"],
        template="You are a retail analyst. Classify the sentiment of this review as Positive, Negative, or Neutral and extract one key insight: {review}"
    )
    chain = llm | prompt
    
    responses = [chain.invoke({"review": review}) for review in reviews]
    sentiments = [resp.split("Insight")[0].strip().split(": ")[-1] for resp in responses]
    insights = [resp.split("Insight: ")[-1].strip() for resp in responses]
    
    print("Review Analysis: Sentiments and insights extracted")
    for i, (review, sentiment, insight) in enumerate(zip(reviews, sentiments, insights)):
        print(f"Review {i+1}: {review}")
        print(f"Sentiment: {sentiment}")
        print(f"Insight: {insight}")

    # Visualization
    sentiment_counts = Counter(sentiments)
    plt.figure(figsize=(8, 4))
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color='purple')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("review_analysis_output.png")
    print("Visualization: Sentiment distribution saved as review_analysis_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_review_analysis_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`review_analysis_output.png`) showing sentiment distribution, illustrating review analysis results.

## 💡 Retail Applications

- **Product Feedback**: Identify strengths and weaknesses.
- **Customer Satisfaction**: Monitor sentiment trends.
- **Quality Control**: Highlight recurring issues.

## 🏆 Practical Tasks

1. Design a prompt to extract sentiment from retail reviews.
2. Extract insights from multiple reviews.
3. Visualize sentiment distribution for reviews.

## 💡 Interview Scenarios

**Question**: How do you extract sentiment from reviews using prompts?  
**Answer**: Use prompts to classify sentiment and extract insights, specifying role and output format.  
**Key**: Ensures accurate retail feedback analysis.  
**Example**: `PromptTemplate(template="Classify sentiment and extract insight: {review}")`

**Coding Task**: Create a prompt for retail review sentiment analysis.  
**Tip**: Use `PromptTemplate` with sentiment and insight extraction.

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [“Prompt Engineering Guide” by DAIR.AI](https://www.promptingguide.ai/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/review-analysis`).
3. Commit changes (`git commit -m 'Add review analysis content'`).
4. Push to the branch (`git push origin feature/review-analysis`).
5. Open a Pull Request.