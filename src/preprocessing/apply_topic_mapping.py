"""
Apply Topic Mapping
===================
Reads topic_mapping.json and updates tlu_metadata_clean.jsonl
"""

import json
import shutil

INPUT_FILE = 'tlu_metadata_clean.jsonl'
MAPPING_FILE = 'topic_mapping.json'
OUTPUT_FILE = 'tlu_metadata_clean_v2.jsonl'

def main():
    print("🚀 APPLYING TOPIC MAPPING")
    print("="*80)
    
    # 1. Load Mapping
    print(f"📂 Loading mapping from {MAPPING_FILE}...")
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"✅ Loaded {len(mapping):,} mapping rules")
    except FileNotFoundError:
        print("❌ Mapping file not found! Run normalize_topics_groq.py first.")
        return

    # 2. Process Data
    print(f"\n🔄 Processing {INPUT_FILE}...")
    processed_count = 0
    changed_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            rec = json.loads(line)
            original_topics = rec.get('subjects', [])
            new_topics = []
            has_change = False
            
            for t in original_topics:
                if t in mapping:
                    normalized = mapping[t]
                    if normalized != t:
                        has_change = True
                    new_topics.append(normalized)
                else:
                    new_topics.append(t) # Keep original if no mapping
            
            # Deduplicate topics after normalization
            unique_new_topics = sorted(list(set(new_topics)))
            if len(unique_new_topics) != len(original_topics):
                has_change = True
                
            rec['subjects'] = unique_new_topics
            f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            
            processed_count += 1
            if has_change:
                changed_count += 1
                
    print(f"\n✅ Processed {processed_count:,} records")
    print(f"📝 Updated topics in {changed_count:,} records")
    print(f"💾 Saved to {OUTPUT_FILE}")
    
    # Optional: Replace original file
    # shutil.move(OUTPUT_FILE, INPUT_FILE)
    # print(f"🔄 Overwrote {INPUT_FILE}")

if __name__ == "__main__":
    main()
