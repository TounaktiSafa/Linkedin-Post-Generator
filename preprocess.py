import json
import re
from langchain_core.prompts import PromptTemplate
from llm_helper import llm
import time
from typing import Dict, Any, List


def process_posts(raw_file_path, processed_file_path="Dataset/Preprocessed_posts.json"):
    enriched_posts = []
    with open(raw_file_path, "r", encoding="utf-8", errors="surrogatepass") as f:
        posts = json.load(f)
        
        for i, post in enumerate(posts):
            try:
                # Clean the text before processing
                clean_text = clean_unicode(post['text'])
                post['text'] = clean_text  # Update the post with cleaned text
                
                # Extract metadata with retry logic
                metadata = extract_metadata_with_retry(clean_text)
                post_metadata = {**post, **metadata}
                enriched_posts.append(post_metadata)
                
                print(f"Processed post {i+1}/{len(posts)}")
                
            except Exception as e:
                print(f"Error processing post {i+1}: {e}")
                # Add fallback metadata for failed posts
                fallback_metadata = {
                    "line_count": len(post.get('text', '').split('\n')),
                    "language": "English",
                    "tags": []
                }
                post_metadata = {**post, **fallback_metadata}
                enriched_posts.append(post_metadata)
                continue

    # Write the processed data
    write_json_safely(enriched_posts, processed_file_path)
    return enriched_posts


def clean_unicode(text: str) -> str:
    """Handle problematic Unicode characters including surrogate pairs"""
    if not isinstance(text, str):
        return str(text)
    
    # Method 1: Replace surrogate pairs with safe characters
    try:
        # First, try to encode/decode to catch issues
        text.encode('utf-8', 'strict')
        return text
    except UnicodeEncodeError:
        # Replace problematic surrogate pairs
        # Surrogate pairs are in the range \ud800-\udfff
        text = re.sub(r'[\ud800-\udfff]', '�', text)
        
        # Alternative: Remove surrogate pairs entirely
        # text = re.sub(r'[\ud800-\udfff]', '', text)
        
        return text
    except Exception as e:
        print(f"Unicode cleaning error: {e}")
        # Fallback: use Python's built-in error handling
        return text.encode('utf-8', 'replace').decode('utf-8')


def extract_json_from_text(text: str) -> [str]:
    """Extract JSON from text that may contain explanations"""
    import re
    
    # Method 1: Look for JSON blocks in markdown
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Method 2: Look for standalone JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        # Take the largest JSON-like string (most complete)
        return max(matches, key=len).strip()
    
    # Method 3: Try to find anything that starts with { and ends with }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return text[start_idx:end_idx + 1].strip()
    
    return None


def extract_metadata_with_retry(post: str, max_retries: int = 3) -> Dict[str, Any]:
    """Extract metadata with retry logic for API failures"""
    
    for attempt in range(max_retries):
        try:
            return extract_metadata(post)
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a service unavailable error or JSON parsing error
            if ('503' in error_msg or 'service unavailable' in error_msg or 
                'invalid json output' in error_msg or 'expecting value' in error_msg or
                'context too big' in error_msg):
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20 seconds
                print(f"API unavailable (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                if attempt < max_retries - 1:  # Don't wait on the last attempt
                    time.sleep(wait_time)
                continue
            else:
                # For other errors, don't retry
                print(f"Non-retryable error: {e}")
                break
    
    # If all retries failed, return fallback metadata
    print("All API attempts failed, using fallback metadata")
    return get_fallback_metadata(post)


def extract_metadata(post: str) -> Dict[str, Any]:
    """Extract metadata using LLM"""
    # Truncate very long posts to avoid context size issues
    max_post_length = 1000
    if len(post) > max_post_length:
        post = post[:max_post_length] + "..."
    
    template = '''Extract the following information from this LinkedIn post:
    - line_count: number of lines
    - language: either "English" or "French"
    - tags: array of maximum two relevant tags

    CRITICAL: Return ONLY a valid JSON object. No explanations, no markdown, no additional text.

    Post: {post}

    JSON:'''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm

    response = chain.invoke(input={'post': post})
    
    # Clean the response more thoroughly
    text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # Remove any explanatory text before JSON
    json_str = extract_json_from_text(text)
    
    if not json_str:
        raise ValueError("No valid JSON found in response")

    return json.loads(json_str)


def get_fallback_metadata(post: str) -> Dict[str, Any]:
    """Generate fallback metadata when API fails"""
    # Simple heuristics for fallback
    line_count = len(post.split('\n'))
    
    # Simple language detection
    french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'une', 'un', 'ce', 'cette']
    text_lower = post.lower()
    french_count = sum(1 for word in french_words if word in text_lower)
    language = "French" if french_count > 3 else "English"
    
    # Simple tag extraction based on common LinkedIn keywords
    tags = []
    linkedin_keywords = {
        'career': ['career', 'job', 'work', 'employment'],
        'business': ['business', 'company', 'startup', 'entrepreneur'],
        'tech': ['technology', 'tech', 'ai', 'digital', 'software'],
        'leadership': ['leadership', 'management', 'team', 'leader'],
        'marketing': ['marketing', 'brand', 'social media'],
        'networking': ['network', 'connection', 'professional']
    }
    
    post_lower = post.lower()
    for tag, keywords in linkedin_keywords.items():
        if any(keyword in post_lower for keyword in keywords):
            tags.append(tag)
            if len(tags) >= 2:
                break
    
    return {
        "line_count": line_count,
        "language": language,
        "tags": tags
    }


def write_json_safely(data: List[Dict], file_path: str):
    """Write JSON data with proper Unicode handling"""
    
    # Clean all data before writing
    cleaned_data = clean_data_recursively(data)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote {len(cleaned_data)} posts to {file_path}")
    except UnicodeEncodeError as e:
        print(f"Unicode error during writing: {e}")
        # Fallback: write with ASCII encoding
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=True)
        print(f"Wrote with ASCII encoding to {file_path}")


def clean_data_recursively(obj):
    """Recursively clean Unicode issues in nested data structures"""
    if isinstance(obj, str):
        return clean_unicode(obj)
    elif isinstance(obj, dict):
        return {clean_unicode(str(k)): clean_data_recursively(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_data_recursively(item) for item in obj]
    else:
        return obj


if __name__ == "__main__":
    try:
        processed_data = process_posts("Dataset/RawData.json", "Dataset/Preprocessed_posts.json")
        print(f"\nProcessing complete! Processed {len(processed_data)} posts.")
        
        # Display sample of processed posts
        print("\nSample processed posts:")
        for i, post in enumerate(processed_data[:3]):
            print(f"\nPost {i+1}:")
            print(f"  Text preview: {post.get('text', '')[:100]}...")
            print(f"  Language: {post.get('language', 'Unknown')}")
            print(f"  Line count: {post.get('line_count', 0)}")
            print(f"  Tags: {post.get('tags', [])}")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()