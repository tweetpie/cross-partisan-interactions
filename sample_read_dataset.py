#!/usr/bin/env python3
"""
Cross-Partisan Interactions on Twitter Dataset - Data Merger

This script reads and merges all the component CSV files to create a comprehensive
dataset for analyzing cross-partisan interactions on Twitter.

Usage:
    python sample_read_dataset.py

Output:
    merged_df: Complete merged dataset with all features
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Optional
import warnings
import time
from datetime import datetime

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def print_progress(step: str, message: str) -> None:
    """Print progress message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {step}: {message}")

def check_file_exists(filepath: str) -> bool:
    """Check if file exists and print warning if not"""
    if not os.path.exists(filepath):
        print(f"WARNING: File {filepath} not found. Skipping...")
        return False
    return True

def load_tweet_texts(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all tweet text files
    
    Args:
        data_dir (str): Directory containing the data files
    
    Returns:
        pd.DataFrame: Combined tweet texts dataframe
    """
    print_progress("STEP 3", "Loading tweet texts...")
    tweet_texts_dfs = []
    
    for i in range(1, 11):
        filename = f'{data_dir}/tweet_texts_part_{i}.csv.bz2'
        if check_file_exists(filename):
            try:
                df = pd.read_csv(filename)
                tweet_texts_dfs.append(df)
                print_progress("STEP 3", f"Loaded {filename}: {df.shape}")
            except Exception as e:
                print(f"ERROR loading {filename}: {str(e)}")
                continue
    
    if not tweet_texts_dfs:
        print("ERROR: No tweet text files found!")
        return pd.DataFrame()
    
    # Concatenate all tweet text files
    all_tweet_texts = pd.concat(tweet_texts_dfs, ignore_index=True)
    
    # Remove duplicates if any
    initial_shape = all_tweet_texts.shape
    all_tweet_texts = all_tweet_texts.drop_duplicates(subset=['tweet_id'])
    final_shape = all_tweet_texts.shape
    
    if initial_shape[0] != final_shape[0]:
        print_progress("STEP 3", f"Removed {initial_shape[0] - final_shape[0]} duplicate tweets")
    
    print_progress("STEP 3", f"Combined tweet texts shape: {all_tweet_texts.shape}")
    return all_tweet_texts

def is_cross_partisan(author_theta: float, target_theta: float, threshold: float = 0.0) -> bool:
    """
    Determine if interaction is cross-partisan based on ideal point scores
    
    Args:
        author_theta (float): Author's ideal point score
        target_theta (float): Target author's ideal point score
        threshold (float): Threshold for determining partisan sides
    
    Returns:
        bool: True if interaction is cross-partisan
    """
    if pd.isna(author_theta) or pd.isna(target_theta):
        return False
    return (author_theta > threshold) != (target_theta > threshold)

def create_merged_dataset(data_dir: str = 'data/') -> pd.DataFrame:
    """
    Create the complete merged dataset from all component files
    
    Args:
        data_dir (str): Directory containing the data files
    
    Returns:
        pd.DataFrame: Complete merged dataset
    """
    start_time = time.time()
    
    print("=" * 60)
    print("CROSS-PARTISAN INTERACTIONS ON TWITTER - DATA MERGER")
    print("=" * 60)
    
    # Step 1: Load conversation chain
    print_progress("STEP 1", "Loading conversation chain...")
    conversation_chain_file = f'{data_dir}/conversation_chain.csv.bz2'
    if not check_file_exists(conversation_chain_file):
        print("ERROR: conversation_chain.csv.bz2 is required but not found!")
        return pd.DataFrame()
    
    merged_df = pd.read_csv(conversation_chain_file)
    print_progress("STEP 1", f"Conversation chain shape: {merged_df.shape}")
    print_progress("STEP 1", f"Columns: {list(merged_df.columns)}")
    
    # Step 2: Merge tweet metadata
    print_progress("STEP 2", "Merging tweet metadata...")
    tweet_meta_file = f'{data_dir}/tweet_meta.csv.bz2'
    if check_file_exists(tweet_meta_file):
        tweet_meta = pd.read_csv(tweet_meta_file)
        print_progress("STEP 2", f"Tweet meta shape: {tweet_meta.shape}")
        
        merged_df = merged_df.merge(tweet_meta, on='tweet_id', how='left')
        print_progress("STEP 2", f"After tweet meta merge: {merged_df.shape}")
    else:
        print_progress("STEP 2", "Tweet metadata file not found, skipping...")
    
    # Step 3: Merge tweet texts
    all_tweet_texts = load_tweet_texts(data_dir)
    if not all_tweet_texts.empty:
        merged_df = merged_df.merge(all_tweet_texts, on='tweet_id', how='left')
        print_progress("STEP 3", f"After tweet texts merge: {merged_df.shape}")
    else:
        print_progress("STEP 3", "No tweet texts loaded, skipping...")
    
    # Step 4: Merge author metadata
    print_progress("STEP 4", "Merging author metadata...")
    author_meta_file = f'{data_dir}/author_meta.csv.bz2'
    if check_file_exists(author_meta_file):
        author_meta = pd.read_csv(author_meta_file)
        print_progress("STEP 4", f"Author meta shape: {author_meta.shape}")
        
        merged_df = merged_df.merge(author_meta, on='author_id', how='left')
        print_progress("STEP 4", f"After author meta merge: {merged_df.shape}")
    else:
        print_progress("STEP 4", "Author metadata file not found, skipping...")
        author_meta = pd.DataFrame()  # Empty dataframe for later use
    
    # Step 5: Merge author ideal points
    print_progress("STEP 5", "Merging author ideal points...")
    author_ideal_points_file = f'{data_dir}/author_ideal_point.csv.bz2'
    if check_file_exists(author_ideal_points_file):
        author_ideal_points = pd.read_csv(author_ideal_points_file)
        print_progress("STEP 5", f"Author ideal points shape: {author_ideal_points.shape}")
        
        # Merge author ideal points for the main author
        merged_df = merged_df.merge(author_ideal_points, on='author_id', how='left')
        
        # Merge ideal points for the replied-to author
        author_ideal_points_reply = author_ideal_points.copy()
        author_ideal_points_reply.columns = ['in_reply_to_author_id', 'in_reply_to_author_ideal_point_theta', 
                                           'in_reply_to_author_ideal_point_followed']
        merged_df = merged_df.merge(author_ideal_points_reply, on='in_reply_to_author_id', how='left')
        
        print_progress("STEP 5", f"After author ideal points merge: {merged_df.shape}")
    else:
        print_progress("STEP 5", "Author ideal points file not found, skipping...")
        author_ideal_points = pd.DataFrame()  # Empty dataframe for later use
    
    # Step 6: Merge conversation tweet information
    print_progress("STEP 6", "Merging conversation tweet information...")
    if not tweet_meta.empty:
        conversation_tweet_meta = tweet_meta.copy()
        conversation_tweet_meta.columns = ['conversation_tweet_id', 'conversation_created_at', 'conversation_lang', 
                                          'conversation_reply_count', 'conversation_retweet_count', 
                                          'conversation_quote_count', 'conversation_like_count']
        merged_df = merged_df.merge(conversation_tweet_meta, on='conversation_tweet_id', how='left')
        
        # Merge conversation tweet text
        if not all_tweet_texts.empty:
            conversation_tweet_texts = all_tweet_texts.copy()
            conversation_tweet_texts.columns = ['conversation_tweet_id', 'conversation_text']
            merged_df = merged_df.merge(conversation_tweet_texts, on='conversation_tweet_id', how='left')
        
        print_progress("STEP 6", f"After conversation tweet merge: {merged_df.shape}")
    else:
        print_progress("STEP 6", "Tweet metadata not available, skipping conversation tweet merge...")
    
    # Step 7: Merge conversation author information
    print_progress("STEP 7", "Merging conversation author information...")
    if not author_meta.empty:
        conversation_author_meta = author_meta.copy()
        conversation_author_meta.columns = ['conversation_author_id', 'conversation_author_username', 
                                            'conversation_author_name', 'conversation_author_description',
                                            'conversation_author_location', 'conversation_author_followers_count',
                                            'conversation_author_following_count', 'conversation_author_listed_count',
                                            'conversation_author_tweet_count', 'conversation_author_verified']
        merged_df = merged_df.merge(conversation_author_meta, on='conversation_author_id', how='left')
        
        # Merge conversation author ideal points
        if not author_ideal_points.empty:
            conversation_author_ideal_points = author_ideal_points.copy()
            conversation_author_ideal_points.columns = ['conversation_author_id', 'conversation_author_ideal_point_theta', 
                                                       'conversation_author_ideal_point_followed']
            merged_df = merged_df.merge(conversation_author_ideal_points, on='conversation_author_id', how='left')
        
        print_progress("STEP 7", f"After conversation author merge: {merged_df.shape}")
    else:
        print_progress("STEP 7", "Author metadata not available, skipping conversation author merge...")
    
    # Step 8: Merge additional features
    print_progress("STEP 8", "Merging additional features...")
    
    # Perspective toxicity scores
    perspective_toxicity_file = f'{data_dir}/perspective_toxicity.csv.bz2'
    if check_file_exists(perspective_toxicity_file):
        perspective_toxicity = pd.read_csv(perspective_toxicity_file)
        print_progress("STEP 8", f"Perspective toxicity shape: {perspective_toxicity.shape}")
        merged_df = merged_df.merge(perspective_toxicity, on='tweet_id', how='left')
    else:
        print_progress("STEP 8", "Perspective toxicity file not found, skipping...")
    
    # Tweet topics
    tweet_topics_file = f'{data_dir}/tweet_topics.csv.bz2'
    if check_file_exists(tweet_topics_file):
        tweet_topics = pd.read_csv(tweet_topics_file)
        print_progress("STEP 8", f"Tweet topics shape: {tweet_topics.shape}")
        merged_df = merged_df.merge(tweet_topics, on='tweet_id', how='left')
    else:
        print_progress("STEP 8", "Tweet topics file not found, skipping...")
    
    # LLM stance inferences (optional)
    llm_stance_file = f'{data_dir}/llm_stance_inferences.csv.bz2'
    if check_file_exists(llm_stance_file):
        llm_stance = pd.read_csv(llm_stance_file)
        print_progress("STEP 8", f"LLM stance shape: {llm_stance.shape}")
        merged_df = merged_df.merge(llm_stance, on='tweet_id', how='left')
    else:
        print_progress("STEP 8", "LLM stance inferences file not found, skipping...")
    
    print_progress("STEP 8", f"After additional features merge: {merged_df.shape}")
    
    # Step 9: Create derived features
    print_progress("STEP 9", "Creating derived features...")
    
    # Create CPI indicators if ideal points are available
    if 'author_ideal_point_theta' in merged_df.columns and 'in_reply_to_author_ideal_point_theta' in merged_df.columns:
        merged_df['is_cpi_reply'] = merged_df.apply(
            lambda row: is_cross_partisan(row['author_ideal_point_theta'], 
                                         row['in_reply_to_author_ideal_point_theta']), 
            axis=1
        )
        cpi_reply_count = merged_df['is_cpi_reply'].sum()
        print_progress("STEP 9", f"Created is_cpi_reply: {cpi_reply_count} cross-partisan replies")
    else:
        print_progress("STEP 9", "Cannot create is_cpi_reply - missing ideal point data")
    
    if 'author_ideal_point_theta' in merged_df.columns and 'conversation_author_ideal_point_theta' in merged_df.columns:
        merged_df['is_cpi_conversation'] = merged_df.apply(
            lambda row: is_cross_partisan(row['author_ideal_point_theta'], 
                                         row['conversation_author_ideal_point_theta']), 
            axis=1
        )
        cpi_conversation_count = merged_df['is_cpi_conversation'].sum()
        print_progress("STEP 9", f"Created is_cpi_conversation: {cpi_conversation_count} cross-partisan conversations")
    else:
        print_progress("STEP 9", "Cannot create is_cpi_conversation - missing ideal point data")
        
    # Create topic numeric encoding
    if 'topic_category' in merged_df.columns:
        merged_df['topic'] = merged_df['topic_category'].astype('category').cat.codes.astype(float)
        merged_df.loc[merged_df['topic_category'].isna(), 'topic'] = -1.0
        unique_topics = merged_df['topic_category'].nunique()
        print_progress("STEP 9", f"Created topic numeric encoding with {unique_topics} unique topics")
    else:
        print_progress("STEP 9", "Cannot create topic encoding - missing topic_category")
    
    # Final statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    print_progress("FINAL", f"Dataset shape: {merged_df.shape}")
    print_progress("FINAL", f"Processing time: {processing_time:.2f} seconds")
    print_progress("FINAL", f"Memory usage: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return merged_df

def display_dataset_info(df: pd.DataFrame) -> None:
    """Display comprehensive information about the merged dataset"""
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing values:")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages.round(2)
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    print(missing_info.head(10))
    
    # Display sample row
    print("\nSample row (first non-null tweet):")
    sample_idx = df.dropna(subset=['tweet_id']).index[0] if not df.empty else 0
    sample_row = df.iloc[sample_idx]
    
    # Display key fields
    key_fields = ['tweet_id', 'author_id', 'text', 'created_at', 'conversation_tweet_id', 
                  'author_username', 'author_ideal_point_theta', 'perspective_toxicity_score',
                  'topic_category', 'is_cpi_reply', 'is_cpi_conversation']
    
    for field in key_fields:
        if field in sample_row.index:
            value = sample_row[field]
            if pd.isna(value):
                print(f"{field}: NaN")
            elif isinstance(value, str) and len(str(value)) > 100:
                print(f"{field}: {str(value)[:100]}...")
            else:
                print(f"{field}: {value}")

def main():
    """Main function to run the data merger"""
    print("Cross-Partisan Interactions on Twitter - Dataset Merger")
    print("This script will merge all component CSV files into a single dataset.")
    print()
    
    # Check if data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        print("Please ensure the data files are in the 'data/' directory.")
        sys.exit(1)
    
    # Create merged dataset
    try:
        merged_df = create_merged_dataset(data_dir)
        
        if merged_df.empty:
            print("ERROR: Failed to create merged dataset!")
            sys.exit(1)
        
        # Display dataset information
        display_dataset_info(merged_df)
        
        # Ask user if they want to save the dataset
        print("\n" + "=" * 60)
        response = input("Do you want to save the merged dataset to 'merged_dataset.csv'? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("Saving merged dataset...")
            merged_df.to_csv('merged_dataset.csv', index=False)
            print("Dataset saved successfully!")
        
        print("\nMerged dataset is available in the 'merged_df' variable.")
        print("You can now use it for your analysis!")
        
        return merged_df
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    merged_df = main() 