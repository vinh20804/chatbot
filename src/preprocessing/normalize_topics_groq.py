import json
import os
import time
from groq import Groq
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
INPUT_FILE = 'tlu_metadata_clean.jsonl'
OUTPUT_MAPPING_FILE = 'topic_mapping.json'
BATCH_SIZE = 30 
MODEL = "llama-3.1-8b-instant" 
PROMPT_TEMPLATE = """
You are a librarian standardizing Vietnamese keywords.
Your task is to GROUP similar topics and provide ONE canonical (standard) form for each group.

Rules:
1. Fix typos (e.g., "Quản lí" -> "Quản lý", "Kĩ thuật" -> "Kỹ thuật")
2. Standardize case (Title Case or sentence case, be consistent)
3. Merge synonyms (e.g., "Xây dựng DD" -> "Xây dựng dân dụng")
4. Remove redundant punctuation
5. Keep specific topics if they are distinct (e.g., "Project A" != "Project B")
6. Return JSON format ONLY: {{"original_topic": "canonical_topic", ...}}

Input Topics:
{topics}

JSON Output:
"""

def normalize_batch(client, batch_topics):
    """Send a batch of topics to Groq for normalization"""
    topics_str = "\n".join([f"- {t}" for t in batch_topics])
    prompt = PROMPT_TEMPLATE.format(topics=topics_str)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"  ❌ Error processing batch: {e}")
        return {}

def main():
    print("🚀 TOPIC NORMALIZATION WITH GROQ")
    print("="*80)
    
    # 1. Load Topics
    print(f"📂 Loading topics from {INPUT_FILE}...")
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    all_topics = []
    for d in data:
        all_topics.extend(d.get('subjects', []))
        
    topic_counts = Counter(all_topics)
    unique_topics = sorted(list(topic_counts.keys()))
    print(f"✅ Found {len(unique_topics):,} unique topics")
    
    # Filter rare topics (optional optimization)
    # unique_topics = [t for t in unique_topics if topic_counts[t] >= 1] 
    
    # 2. Initialize Groq Client
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("❌ Please update GROQ_API_KEY in the script!")
        return
        
    client = Groq(api_key=GROQ_API_KEY)
    
    # 3. Process Batches
    mapping = {}
    total_batches = (len(unique_topics) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🔄 Processing {total_batches} batches...")
    
    for i in range(0, len(unique_topics), BATCH_SIZE):
        batch = unique_topics[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} topics)...", end='', flush=True)
        
        batch_mapping = normalize_batch(client, batch)
        mapping.update(batch_mapping)
        
        print(f" Done. Mapped {len(batch_mapping)} items.")
        time.sleep(2) 
        
    # 4. Save Mapping
    print(f"\n💾 Saving mapping to {OUTPUT_MAPPING_FILE}...")
    with open(OUTPUT_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
        
    print("✅ Done! Now run 'apply_topic_mapping.py' (to be created)")

if __name__ == "__main__":
    main()
