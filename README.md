# Cross-Partisan Interactions on Twitter Dataset

## Overview

This repository contains the dataset for the paper "Cross-Partisan Interactions on Twitter" which analyzes 3 million tweets from 2020 related to the U.S. political context. The dataset provides insights about cross-partisan interactions (CPIs), including tweet metadata, author information, conversation chains, toxicity scores, and topic classifications.

## Abstract

Many social media studies argue that social media creates echo chambers where some users only interact with peers of the same political orientation. However, recent studies suggest that a substantial amount of Cross-Partisan Interactions (CPIs) do exist — even within echo chambers, but they may be toxic. There is no consensus about how such interactions occur and when they lead to healthy or toxic dialogue. In this paper, we study a comprehensive Twitter dataset that consists of 3 million tweets from 2020 related to the U.S. context to understand the dynamics behind CPIs. We investigate factors that are more associated with such interactions, including how users engage in CPIs, which topics are more contentious, and what are the stances associated with healthy interactions. We find that CPIs are significantly influenced by the nature of the topics being discussed, with politically charged events acting as strong catalysts. The political discourse and pre-established political views sway how users participate in CPIs, but the direction in which users go is nuanced. While Democrats engage in cross-partisan interactions slightly more frequently, these interactions often involve more negative and nonconstructive stances compared to their intra-party interactions. In contrast, Republicans tend to maintain a more consistent tone across interactions. Although users are more likely to engage in CPIs with popular accounts in general, this is less common among Republicans who often engage in CPIs with accounts with a low number of followers for personal matters. Our study has implications beyond Twitter as identifying topics with low toxicity and high CPI can help highlight potential opportunities for reducing polarization while topics with high toxicity and low CPI may action targeted interventions when moderating harm.


## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{ccetinkaya2025cross,
  title={Cross-Partisan Interactions on Twitter},
  author={{\c{C}}etinkaya, Yusuf M{\"u}cahit and Ghafouri, Vahid and Suarez-Tangil, Guillermo and Such, Jose and Elmas, Tu{\u{g}}rulcan},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={19},
  pages={324--340},
  year={2025}
}
```

## Dataset Structure

The dataset consists of several compressed CSV files (`.csv.bz2`) containing different aspects of the Twitter data:

### Files Description

| File | Shape | Description |
|------|--------|-------------|
| `conversation_chain.csv.bz2` | (3,029,231 × 6) | Core conversation linking data |
| `tweet_meta.csv.bz2` | (4,859,929 × 7) | Tweet metadata (timestamps, counts, language) |
| `author_meta.csv.bz2` | (1,034,902 × 10) | Author profile information |
| `tweet_texts_part_1.csv.bz2` to `tweet_texts_part_10.csv.bz2` | (485,992 × 2) each | Tweet text content split into 10 parts |
| `author_ideal_point.csv.bz2` | (1,039,108 × 3) | Author political ideal point scores |
| `llm_stance_inferences.csv.bz2` | (400,000 × 7) | LLM-based stance inference results |
| `perspective_toxicity.csv.bz2` | (2,736,355 × 3) | Perspective API toxicity scores |
| `tweet_topics.csv.bz2` | (1,418,079 × 2) | Tweet topic classifications |

## Data Schema

### Core Files

#### `conversation_chain.csv.bz2`
- `tweet_id`: Unique identifier for the tweet
- `author_id`: Unique identifier for the tweet author
- `in_reply_to_tweet_id`: ID of the tweet being replied to
- `in_reply_to_author_id`: ID of the author being replied to
- `conversation_tweet_id`: ID of the root tweet in the conversation
- `conversation_author_id`: ID of the author who started the conversation

#### `tweet_meta.csv.bz2`
- `tweet_id`: Unique identifier for the tweet
- `created_at`: Tweet creation timestamp
- `lang`: Language of the tweet
- `reply_count`: Number of replies to the tweet
- `retweet_count`: Number of retweets
- `quote_count`: Number of quote tweets
- `like_count`: Number of likes

#### `author_meta.csv.bz2`
- `author_id`: Unique identifier for the author
- `author_username`: Twitter username
- `author_name`: Display name
- `author_description`: Profile description
- `author_location`: Profile location
- `author_followers_count`: Number of followers
- `author_following_count`: Number of accounts following
- `author_listed_count`: Number of lists the author appears in
- `author_tweet_count`: Total number of tweets by the author
- `author_verified`: Verification status

#### `tweet_texts_part_X.csv.bz2` (Can be provided upon request)
- `tweet_id`: Unique identifier for the tweet
- `text`: Full text content of the tweet

#### `author_ideal_point.csv.bz2`
- `author_id`: Unique identifier for the author
- `author_ideal_point_theta`: Political ideal point score
- `author_ideal_point_followed`: Number of political accounts followed

#### `llm_stance_inferences.csv.bz2`
- `tweet_id`: Unique identifier for the tweet
- `conversation_id`: ID of the conversation thread
- `partisan_relationship`: Relationship between participants
- `sentiment_root`: Sentiment of the root tweet
- `sentiment_root_normalized`: Normalized sentiment score
- `stance_reply`: Stance of the reply
- `stance_reply_normalized`: Normalized stance score

#### `perspective_toxicity.csv.bz2`
- `tweet_id`: Unique identifier for the tweet
- `perspective_toxicity_score`: Toxicity score from Perspective API
- `perspective_detected_languages`: Detected languages

#### `tweet_topics.csv.bz2`
- `tweet_id`: Unique identifier for the tweet
- `topic_category`: Topic classification

## Data Merging Instructions

To create a merged dataset, follow these steps:

### Prerequisites

```python
import pandas as pd
import numpy as np
```

### Step 1: Load the Main Conversation Chain

```python
# Load the main conversation chain file
print("Loading conversation chain...")
merged_df = pd.read_csv('data/conversation_chain.csv.bz2')
print(f"Conversation chain shape: {merged_df.shape}")
print(f"Columns: {list(merged_df.columns)}")
```

### Step 2: Merge Tweet Metadata

```python
# Load and merge tweet metadata
print("Loading tweet metadata...")
tweet_meta = pd.read_csv('data/tweet_meta.csv.bz2')
print(f"Tweet meta shape: {tweet_meta.shape}")

# Merge tweet metadata
merged_df = merged_df.merge(tweet_meta, on='tweet_id', how='left')
print(f"After tweet meta merge: {merged_df.shape}")
```

### Step 3: Merge Tweet Texts

```python
# Load and concatenate all tweet text files
print("Loading tweet texts...")
tweet_texts_dfs = []
for i in range(1, 11):  # Files part_1 to part_10
    filename = f'data/tweet_texts_part_{i}.csv.bz2'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        tweet_texts_dfs.append(df)
        print(f"Loaded {filename}: {df.shape}")

# Concatenate all tweet text files
all_tweet_texts = pd.concat(tweet_texts_dfs, ignore_index=True)
print(f"Combined tweet texts shape: {all_tweet_texts.shape}")

# Merge tweet texts
merged_df = merged_df.merge(all_tweet_texts, on='tweet_id', how='left')
print(f"After tweet texts merge: {merged_df.shape}")
```

### Step 4: Merge Author Metadata

```python
# Load author metadata
print("Loading author metadata...")
author_meta = pd.read_csv('data/author_meta.csv.bz2')
print(f"Author meta shape: {author_meta.shape}")

# Merge author metadata for the main author
merged_df = merged_df.merge(author_meta, on='author_id', how='left')
print(f"After author meta merge: {merged_df.shape}")
```

### Step 5: Merge Author Ideal Points

```python
# Load author ideal points
print("Loading author ideal points...")
author_ideal_points = pd.read_csv('data/author_ideal_point.csv.bz2')
print(f"Author ideal points shape: {author_ideal_points.shape}")

# Merge author ideal points for the main author
merged_df = merged_df.merge(author_ideal_points, on='author_id', how='left')

# Merge ideal points for the replied-to author
author_ideal_points_reply = author_ideal_points.copy()
author_ideal_points_reply.columns = ['in_reply_to_author_id', 'in_reply_to_author_ideal_point_theta', 'in_reply_to_author_ideal_point_followed']
merged_df = merged_df.merge(author_ideal_points_reply, on='in_reply_to_author_id', how='left')

print(f"After author ideal points merge: {merged_df.shape}")
```

### Step 6: Merge Conversation Tweet Information

```python
# Create conversation tweet metadata by merging with tweet_meta
print("Creating conversation tweet metadata...")
conversation_tweet_meta = tweet_meta.copy()
conversation_tweet_meta.columns = ['conversation_tweet_id', 'conversation_created_at', 'conversation_lang', 
                                   'conversation_reply_count', 'conversation_retweet_count', 
                                   'conversation_quote_count', 'conversation_like_count']

merged_df = merged_df.merge(conversation_tweet_meta, on='conversation_tweet_id', how='left')

# Merge conversation tweet text
conversation_tweet_texts = all_tweet_texts.copy()
conversation_tweet_texts.columns = ['conversation_tweet_id', 'conversation_text']
merged_df = merged_df.merge(conversation_tweet_texts, on='conversation_tweet_id', how='left')

print(f"After conversation tweet merge: {merged_df.shape}")
```

### Step 7: Merge Conversation Author Information

```python
# Create conversation author metadata
print("Creating conversation author metadata...")
conversation_author_meta = author_meta.copy()
conversation_author_meta.columns = ['conversation_author_id', 'conversation_author_username', 
                                    'conversation_author_name', 'conversation_author_description',
                                    'conversation_author_location', 'conversation_author_followers_count',
                                    'conversation_author_following_count', 'conversation_author_listed_count',
                                    'conversation_author_tweet_count', 'conversation_author_verified']

merged_df = merged_df.merge(conversation_author_meta, on='conversation_author_id', how='left')

# Merge conversation author ideal points
conversation_author_ideal_points = author_ideal_points.copy()
conversation_author_ideal_points.columns = ['conversation_author_id', 'conversation_author_ideal_point_theta', 
                                           'conversation_author_ideal_point_followed']
merged_df = merged_df.merge(conversation_author_ideal_points, on='conversation_author_id', how='left')

print(f"After conversation author merge: {merged_df.shape}")
```

### Step 8: Merge Additional Features

```python
# Load and merge perspective toxicity scores
print("Loading perspective toxicity...")
perspective_toxicity = pd.read_csv('data/perspective_toxicity.csv.bz2')
print(f"Perspective toxicity shape: {perspective_toxicity.shape}")
merged_df = merged_df.merge(perspective_toxicity, on='tweet_id', how='left')

# Load and merge tweet topics
print("Loading tweet topics...")
tweet_topics = pd.read_csv('data/tweet_topics.csv.bz2')
print(f"Tweet topics shape: {tweet_topics.shape}")
merged_df = merged_df.merge(tweet_topics, on='tweet_id', how='left')

# Load and merge LLM stance inferences (if available)
print("Loading LLM stance inferences...")
llm_stance = pd.read_csv('data/llm_stance_inferences.csv.bz2')
print(f"LLM stance shape: {llm_stance.shape}")
merged_df = merged_df.merge(llm_stance, on='tweet_id', how='left')

print(f"Final merged dataframe shape: {merged_df.shape}")
```

### Step 9: Create Additional Derived Features

```python
# Create CPI (Cross-Partisan Interaction) indicators
# Note: This requires political ideal point thresholds - adjust as needed
print("Creating CPI indicators...")

# Define threshold for cross-partisan interaction (example: opposite sides of 0)
def is_cross_partisan(author_theta, target_theta, threshold=0):
    """
    Determine if interaction is cross-partisan based on ideal point scores
    """
    if pd.isna(author_theta) or pd.isna(target_theta):
        return False
    return (author_theta > threshold) != (target_theta > threshold)

# Create CPI indicators
merged_df['is_cpi_reply'] = merged_df.apply(
    lambda row: is_cross_partisan(row['author_ideal_point_theta'], 
                                 row['in_reply_to_author_ideal_point_theta']), 
    axis=1
)

merged_df['is_cpi_conversation'] = merged_df.apply(
    lambda row: is_cross_partisan(row['author_ideal_point_theta'], 
                                 row['conversation_author_ideal_point_theta']), 
    axis=1
)


# Create topic numeric encoding (example)
merged_df['topic'] = merged_df['topic_category'].astype('category').cat.codes.astype(float)
merged_df.loc[merged_df['topic_category'].isna(), 'topic'] = -1.0

print("Created additional features:")
print(f"- is_cpi_reply: {merged_df['is_cpi_reply'].sum()} cross-partisan replies")
print(f"- is_cpi_conversation: {merged_df['is_cpi_conversation'].sum()} cross-partisan conversations")
```

### Step 10: Final Dataset Information

```python
# Display final dataset information
print("=== FINAL MERGED DATASET ===")
print(f"Shape: {merged_df.shape}")
print(f"Memory usage: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nColumn overview:")
print(merged_df.dtypes)
print("\nSample row:")
print(merged_df.iloc[0])
```

### Complete Merge Function

For convenience, here's a complete function to perform the entire merge:

```python
def create_merged_dataset(data_dir='data/'):
    """
    Create the complete merged dataset from all component files
    
    Args:
        data_dir (str): Directory containing the data files
    
    Returns:
        pd.DataFrame: Complete merged dataset
    """
    # Step 1: Load conversation chain
    print("Step 1: Loading conversation chain...")
    merged_df = pd.read_csv(f'{data_dir}/conversation_chain.csv.bz2')
    
    # Step 2: Merge tweet metadata
    print("Step 2: Merging tweet metadata...")
    tweet_meta = pd.read_csv(f'{data_dir}/tweet_meta.csv.bz2')
    merged_df = merged_df.merge(tweet_meta, on='tweet_id', how='left')
    
    # Step 3: Merge tweet texts
    print("Step 3: Merging tweet texts...")
    tweet_texts_dfs = []
    for i in range(1, 11):
        filename = f'{data_dir}/tweet_texts_part_{i}.csv.bz2'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            tweet_texts_dfs.append(df)
    
    all_tweet_texts = pd.concat(tweet_texts_dfs, ignore_index=True)
    merged_df = merged_df.merge(all_tweet_texts, on='tweet_id', how='left')
    
    # Step 4: Merge author metadata
    print("Step 4: Merging author metadata...")
    author_meta = pd.read_csv(f'{data_dir}/author_meta.csv.bz2')
    merged_df = merged_df.merge(author_meta, on='author_id', how='left')
    
    # Step 5: Merge author ideal points
    print("Step 5: Merging author ideal points...")
    author_ideal_points = pd.read_csv(f'{data_dir}/author_ideal_point.csv.bz2')
    merged_df = merged_df.merge(author_ideal_points, on='author_id', how='left')
    
    # Reply-to author ideal points
    author_ideal_points_reply = author_ideal_points.copy()
    author_ideal_points_reply.columns = ['in_reply_to_author_id', 'in_reply_to_author_ideal_point_theta', 
                                        'in_reply_to_author_ideal_point_followed']
    merged_df = merged_df.merge(author_ideal_points_reply, on='in_reply_to_author_id', how='left')
    
    # Step 6: Merge conversation tweet information
    print("Step 6: Merging conversation tweet information...")
    conversation_tweet_meta = tweet_meta.copy()
    conversation_tweet_meta.columns = ['conversation_tweet_id', 'conversation_created_at', 'conversation_lang', 
                                       'conversation_reply_count', 'conversation_retweet_count', 
                                       'conversation_quote_count', 'conversation_like_count']
    merged_df = merged_df.merge(conversation_tweet_meta, on='conversation_tweet_id', how='left')
    
    conversation_tweet_texts = all_tweet_texts.copy()
    conversation_tweet_texts.columns = ['conversation_tweet_id', 'conversation_text']
    merged_df = merged_df.merge(conversation_tweet_texts, on='conversation_tweet_id', how='left')
    
    # Step 7: Merge conversation author information
    print("Step 7: Merging conversation author information...")
    conversation_author_meta = author_meta.copy()
    conversation_author_meta.columns = ['conversation_author_id', 'conversation_author_username', 
                                        'conversation_author_name', 'conversation_author_description',
                                        'conversation_author_location', 'conversation_author_followers_count',
                                        'conversation_author_following_count', 'conversation_author_listed_count',
                                        'conversation_author_tweet_count', 'conversation_author_verified']
    merged_df = merged_df.merge(conversation_author_meta, on='conversation_author_id', how='left')
    
    conversation_author_ideal_points = author_ideal_points.copy()
    conversation_author_ideal_points.columns = ['conversation_author_id', 'conversation_author_ideal_point_theta', 
                                               'conversation_author_ideal_point_followed']
    merged_df = merged_df.merge(conversation_author_ideal_points, on='conversation_author_id', how='left')
    
    # Step 8: Merge additional features
    print("Step 8: Merging additional features...")
    perspective_toxicity = pd.read_csv(f'{data_dir}/perspective_toxicity.csv.bz2')
    merged_df = merged_df.merge(perspective_toxicity, on='tweet_id', how='left')
    
    tweet_topics = pd.read_csv(f'{data_dir}/tweet_topics.csv.bz2')
    merged_df = merged_df.merge(tweet_topics, on='tweet_id', how='left')
    
    # Step 9: Create derived features
    print("Step 9: Creating derived features...")
    def is_cross_partisan(author_theta, target_theta, threshold=0):
        if pd.isna(author_theta) or pd.isna(target_theta):
            return False
        return (author_theta > threshold) != (target_theta > threshold)
    
    merged_df['is_cpi_reply'] = merged_df.apply(
        lambda row: is_cross_partisan(row['author_ideal_point_theta'], 
                                     row['in_reply_to_author_ideal_point_theta']), 
        axis=1
    )
    
    merged_df['is_cpi_conversation'] = merged_df.apply(
        lambda row: is_cross_partisan(row['author_ideal_point_theta'], 
                                     row['conversation_author_ideal_point_theta']), 
        axis=1
    )
    
    merged_df['topic'] = merged_df['topic_category'].astype('category').cat.codes.astype(float)
    merged_df.loc[merged_df['topic_category'].isna(), 'topic'] = -1.0
    
    print(f"Final dataset shape: {merged_df.shape}")
    return merged_df

# Usage
merged_df = create_merged_dataset()
```

## Notes

1. **Missing Data**: Some files may not contain all tweets due to Twitter API limitations or processing constraints. Tweet texts are not shared due to copyright policies.

2. **Memory Usage**: The complete merged dataset will be quite large (>1GB in memory). Consider processing in chunks if memory is limited.

3. **CPI Calculation**: The cross-partisan interaction logic uses ideal point thresholds. Adjust the threshold in the `is_cross_partisan` function based on your analysis needs. Used threshold is shared in the publication.

4. **Data Quality**: Always verify data quality and handle missing values appropriately for your specific analysis.
