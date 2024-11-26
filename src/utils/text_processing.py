import pandas as pd 
import os 
import re
def normalizer(s: str):
    """
    Normalize a string by converting it to lowercase and removing all non-alphanumeric characters.
    Convert vietnamese to english

    Args:
    - s (str): The input string.

    Returns:
    - str: The normalized string.
    """
    # Convert to lowercase
    s = " ".join(s.split(":")[1:]) if ":" in s else s
    s = s.lower()

    # Convert vietnamese to english
    s = s.replace('đ', 'd')
    s = s.replace('ă', 'a')
    s = s.replace('â', 'a')
    s = s.replace('ê', 'e')
    s = s.replace('ô', 'o')
    s = s.replace('ơ', 'o')
    s = s.replace('ư', 'u')
    s = s.replace('ơ', 'o')
    s = s.replace('ư', 'u')
    
    # Remove non-alphanumeric and whitesapce characters
    s = re.sub(r'[^a-z0-9\s]', '', s)
    
    # remove 2 or more near whitespace 
    s = re.sub(r'\s{2,}', ' ', s)
    s = s.strip()
    
    return s
    
def jaccard_similarity_2_shingle(s1: str, s2: str) -> float:
    """
    Calculate the Jaccard similarity between two strings using 2-shingles.

    Args:
    - s1 (str): The first input string.
    - s2 (str): The second input string.

    Returns:
    - float: The Jaccard similarity between the two strings.
    """
    #normalize input string
    s1 = normalizer(s1)
    s2 = normalizer(s2)
    
    # Extract 2-shingles from each string
    shingles1 = set(s1[i:i+2] for i in range(len(s1) - 1))
    shingles2 = set(s2[i:i+2] for i in range(len(s2) - 1))

    # Calculate Jaccard similarity
    intersection = len(shingles1.intersection(shingles2))
    union = len(shingles1.union(shingles2))
    return intersection / union if union > 0 else 0

def main():
    s1 = "#9883: Rơi dao T4"
    s2 = "rơi dao"
    print(normalizer(s1))
    print(normalizer(s2))
    print(jaccard_similarity_2_shingle(normalizer(s1), normalizer(s2)))
    
if __name__ == "__main__":
    main()