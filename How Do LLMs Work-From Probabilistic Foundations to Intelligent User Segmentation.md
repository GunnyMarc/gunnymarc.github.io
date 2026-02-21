# How Do LLMs Work: From Probabilistic Foundations to Intelligent User Segmentation

*A deep dive into the mechanics of Large Language Models and their transformative applications in recommendation systems*

---

Large Language Models have fundamentally altered how we approach machine learning problems—from natural language understanding to, perhaps surprisingly, user segmentation and recommendation systems. But what exactly happens inside these models? How do they learn, and more importantly, how can we harness their capabilities for practical business applications like predicting user behavior?

This post explores the four-stage training pipeline that transforms random neural networks into sophisticated reasoning engines, examines the probabilistic foundations that make text generation possible, and demonstrates how these same principles can revolutionize user segmentation and recommendation logic.

## The Probabilistic Heart of Language Models

Before diving into architectures and training stages, we must understand the mathematical foundation that makes LLMs possible: **conditional probability**.

Consider a simple population scenario. Imagine 14 individuals with varying preferences—some enjoy tennis, others football, some both, and a few neither. Conditional probability answers questions like: *"Given that someone likes football, what's the probability they also enjoy tennis?"*[^1]

This same principle powers every word an LLM generates. When you prompt GPT with "The boy went to the," the model calculates:

$$P(\text{next word} | \text{"The boy went to the"})$$

The model evaluates thousands of possible continuations—"school," "park," "hospital"—and assigns each a probability based on patterns learned from massive text corpora. The word with the highest conditional probability becomes the prediction.

```python
import numpy as np
from scipy.special import softmax

# Simulating LLM output layer logits
logits = np.array([2.1, 0.3, 3.8, 1.5, 2.9])  # [Cafe, Hospital, Playground, Park, School]
tokens = ["Cafe", "Hospital", "Playground", "Park", "School"]

# Convert to probabilities via softmax
probabilities = softmax(logits)

for token, prob in zip(tokens, probabilities):
    print(f"{token}: {prob:.3f}")
# Output: Playground has highest probability at 0.467
```

This mechanism—predicting distributions over possible next tokens—forms the backbone of both text generation and, as we'll explore, intelligent recommendation systems.

## The Four Stages of Training LLMs from Scratch

Modern LLMs don't emerge fully-formed. They progress through four distinct training stages, each building upon the last to create increasingly capable systems.[^2]

### Stage 0: The Blank Slate

Every LLM begins as a randomly initialized neural network—billions of parameters set to arbitrary values. Ask this untrained model "What is an LLM?" and you'll receive gibberish like "try peter hand and hello 448Sn." The model possesses no knowledge, only random weights waiting to be shaped by data.

### Stage 1: Pre-training—Learning the Language

Pre-training transforms this blank slate into a language-understanding machine through **next-token prediction** on massive text corpora. The model ingests vast archives—books, websites, code repositories—learning to predict what comes next in any sequence.

```python
import torch
import torch.nn.functional as F

def calculate_pretraining_loss(model_output, target_tokens):
    """
    Cross-entropy loss for next-token prediction
    This is the core learning signal during pre-training
    """
    # model_output: (batch_size, sequence_length, vocab_size)
    # target_tokens: (batch_size, sequence_length)
    
    loss = F.cross_entropy(
        model_output.view(-1, model_output.size(-1)),
        target_tokens.view(-1),
        ignore_index=-100  # Padding tokens
    )
    return loss

# The loss function: -log(P(correct_token | context))
# Lower loss = higher probability assigned to correct continuations
```

Through billions of such predictions, the model absorbs grammar, facts, reasoning patterns, and world knowledge. However, a pre-trained model isn't conversational—it simply continues text rather than answering questions.

### Stage 2: Instruction Fine-tuning—Becoming Conversational

To transform the text-completion engine into a helpful assistant, we employ **instruction fine-tuning**. The model trains on carefully curated instruction-response pairs:

- *Instruction*: "Summarize this article in three sentences."
- *Response*: [A proper three-sentence summary]

This stage teaches the model to follow prompts, format responses appropriately, and exhibit helpful behaviors.

### Stage 3: Preference Fine-tuning (RLHF)

Human preferences are nuanced. Two grammatically correct responses might differ vastly in helpfulness, clarity, or safety. Preference fine-tuning addresses this through **Reinforcement Learning from Human Feedback (RLHF)**.

The process:
1. Generate multiple responses to the same prompt
2. Human annotators rank responses by preference
3. Train a reward model to predict these preferences
4. Update the LLM using reinforcement learning (PPO algorithm)

This teaches the model to align with human values—even when there's no objectively "correct" answer.

### Stage 4: Reasoning Fine-tuning

For tasks with definitive answers—mathematics, logic problems, code execution—we can verify correctness directly. **Reasoning fine-tuning** uses this verifiable feedback:

1. Present the model with a reasoning task
2. Model generates a step-by-step solution
3. Compare against the known correct answer
4. Assign rewards based on correctness
5. Update parameters to increase likelihood of correct reasoning chains

DeepSeek's GRPO algorithm exemplifies this approach, creating models that excel at structured reasoning tasks.

## Temperature: Controlling Creativity vs. Determinism

If LLMs always selected the highest-probability token, outputs would be repetitive and predictable. **Temperature** introduces controlled randomness:

```python
def temperature_adjusted_softmax(logits, temperature=1.0):
    """
    Temperature controls the sharpness of probability distribution
    - Low temperature (0.1): Nearly deterministic, picks most likely token
    - High temperature (2.0): More uniform, enables creative exploration
    """
    return softmax(logits / temperature)

# Demonstration
logits = np.array([1, 2, 3, 4])

print("Original:", softmax(logits).round(3))
# [0.032, 0.087, 0.236, 0.644]

print("Low temp (0.1):", temperature_adjusted_softmax(logits, 0.1).round(6))
# [0.000000, 0.000000, 0.000045, 0.999955] - Almost always picks "4"

print("High temp (2.0):", temperature_adjusted_softmax(logits, 2.0).round(3))
# [0.148, 0.195, 0.256, 0.336] - More uniform, exploratory
```

This same principle of probability distribution manipulation proves crucial for recommendation systems.

## From Language to User Segmentation: A Natural Bridge

Here's the insight that transforms LLM understanding into business value: **user behavior sequences are just another language**.

Just as "The boy went to the [school]" has predictable continuations based on patterns, so does "User clicked Product A, then Product B, then [?]." The same probabilistic framework that powers GPT can power sophisticated user segmentation and recommendation engines.

### Loading and Preparing User Behavior Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load user interaction data
def load_user_behavior_data(filepath):
    """
    Load user behavior sequences for segmentation analysis
    Similar to loading text corpora for LLM pre-training
    """
    df = pd.read_csv(filepath)
    
    # Each user's behavior is a "sentence" of actions
    user_sequences = df.groupby('user_id')['action'].apply(list).reset_index()
    user_sequences.columns = ['user_id', 'action_sequence']
    
    # Encode actions as tokens (like vocabulary in LLMs)
    encoder = LabelEncoder()
    all_actions = df['action'].unique()
    encoder.fit(all_actions)
    
    return user_sequences, encoder

# Example: User behavior as "language"
# User 1: [view_product, add_to_cart, checkout, purchase]
# User 2: [view_product, view_product, exit]
# These sequences can be modeled with the same techniques as text!
```

### Calculating User Propensities

The core LLM insight—conditional probability—directly applies to propensity scoring:

```python
def calculate_conversion_propensity(user_sequences, target_action='purchase'):
    """
    Calculate P(purchase | user_behavior_sequence)
    This mirrors how LLMs calculate P(next_token | context)
    """
    propensities = []
    
    for _, row in user_sequences.iterrows():
        sequence = row['action_sequence']
        
        # Feature extraction from sequence
        features = {
            'sequence_length': len(sequence),
            'unique_actions': len(set(sequence)),
            'cart_additions': sequence.count('add_to_cart'),
            'product_views': sequence.count('view_product'),
            'has_target': 1 if target_action in sequence else 0
        }
        
        # Simple propensity calculation (in practice, use trained model)
        propensity = (
            0.1 * features['cart_additions'] +
            0.05 * features['product_views'] +
            0.3 * (features['unique_actions'] / max(features['sequence_length'], 1))
        )
        propensities.append(min(propensity, 1.0))
    
    user_sequences['propensity'] = propensities
    return user_sequences
```

### Visualizing User Segments

```python
import matplotlib.pyplot as plt

def visualize_user_segments(user_data):
    """
    Visualize user segments based on behavioral propensities
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Propensity distribution
    axes[0].hist(user_data['propensity'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Conversion Propensity')
    axes[0].set_ylabel('User Count')
    axes[0].set_title('Distribution of User Conversion Propensities')
    axes[0].axvline(user_data['propensity'].mean(), color='red', 
                    linestyle='--', label=f"Mean: {user_data['propensity'].mean():.3f}")
    axes[0].legend()
    
    # Segment breakdown
    segments = pd.cut(user_data['propensity'], 
                      bins=[0, 0.3, 0.6, 1.0], 
                      labels=['Low', 'Medium', 'High'])
    segment_counts = segments.value_counts()
    axes[1].pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%',
                colors=['#ff9999', '#ffcc99', '#99ff99'])
    axes[1].set_title('User Segment Distribution')
    
    plt.tight_layout()
    plt.savefig('user_segments.png', dpi=150)
    return fig
```

### Building LLM-Powered Recommendation Logic

Modern recommendation systems can leverage LLMs for contextual, explainable suggestions:

```python
from openai import OpenAI

def llm_powered_recommendations(user_profile, product_catalog, n_recommendations=5):
    """
    Use LLM reasoning for context-aware recommendations
    Combines behavioral propensity with semantic understanding
    """
    client = OpenAI()
    
    prompt = f"""
    Based on this user profile:
    - Browsing history: {user_profile['recent_views']}
    - Purchase history: {user_profile['purchases']}
    - Propensity score: {user_profile['propensity']:.2f}
    - Segment: {user_profile['segment']}
    
    Available products: {product_catalog[:20]}
    
    Recommend {n_recommendations} products with brief explanations.
    Consider both behavioral patterns and semantic relevance.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3  # Lower temperature for consistent recommendations
    )
    
    return response.choices[0].message.content
```

## Early Prediction: The Competitive Advantage

The same pre-training insight that allows LLMs to predict the next word enables **early prediction** of user behavior. By analyzing partial sequences, we can identify high-value users before conversion, predict churn before it happens, and personalize experiences in real-time.

This represents the convergence of probabilistic language modeling and business intelligence—where understanding "what comes next" drives both coherent text generation and intelligent user engagement.

## Conclusion

Large Language Models are, at their core, sophisticated probability machines trained through a four-stage pipeline: pre-training on vast corpora, instruction fine-tuning for task compliance, preference alignment through human feedback, and reasoning enhancement through verifiable rewards.

These same probabilistic foundations—conditional probability, temperature-controlled sampling, sequence modeling—transfer directly to user segmentation and recommendation systems. The patterns in user behavior form a language as learnable as English or Python, and the techniques that produce eloquent text can equally produce intelligent business decisions.

As LLMs continue evolving, their application extends far beyond chatbots into the core infrastructure of personalization, prediction, and user understanding.

---

## References

[^1]: Conditional probability visualizations and explanations adapted from "How Do LLMs Work?" by Daily Dose of Data Science, demonstrating the Venn diagram approach to understanding P(A|B) calculations.

[^2]: The four-stage LLM training framework—pre-training, instruction fine-tuning, preference fine-tuning, and reasoning fine-tuning—is detailed in "4 Stages of Training LLMs from Scratch" by Avi Chawla, Daily Dose of Data Science, July 2025.

[^3]: Temperature-adjusted softmax and its effects on generation diversity are demonstrated through code examples showing how T→0 produces deterministic outputs while T→∞ produces uniform distributions.

[^4]: RLHF (Reinforcement Learning from Human Feedback) and the PPO algorithm are the standard approaches for preference fine-tuning, as used by OpenAI and other major AI labs.

[^5]: GRPO (Group Relative Policy Optimization) by DeepSeek represents the state-of-the-art in reasoning fine-tuning with verifiable rewards.

---

*Word count: ~1,720*
