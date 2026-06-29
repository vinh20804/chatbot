import json
import re
from collections import defaultdict

input_file = 'tlu_metadata_dim.jsonl'

type_counts = defaultdict(int)
type_samples = defaultdict(list)
degree_counts = defaultdict(int)
unmatched_titles = []

# Regex cho bậc học
degree_patterns = {
    "Tiến sĩ": r"tiến sĩ|phd|doctor",
    "Thạc sĩ": r"thạc sĩ|master",
    "Kỹ sư": r"kỹ sư|engineer",
    "Cử nhân": r"cử nhân|bachelor",
    "Đồ án": r"đồ án",
    "Khóa luận": r"khóa luận"
}

print(f"Analyzing types and degrees in {input_file}...")

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                
                # --- Analyze Type ---
                doc_types = record.get('t   ype', ['Unknown'])
                if isinstance(doc_types, str):
                    doc_types = [doc_types]
                
                title = record.get('title', ['No Title'])[0]
                
                for t in doc_types:
                    type_counts[t] += 1
                    if len(type_samples[t]) < 3: # Keep 3 samples per type
                        type_samples[t].append(title)
                
                # --- Analyze Degree ---
                title_lower = title.lower()
                matched = False
                for degree, pattern in degree_patterns.items():
                    if re.search(pattern, title_lower):
                        degree_counts[degree] += 1
                        matched = True
                        break # Ưu tiên match đầu tiên
                
                if not matched:
                    degree_counts["Unknown"] += 1
                    if len(unmatched_titles) < 10:
                        unmatched_titles.append(title)

            except json.JSONDecodeError:
                continue

    print("\n=== 1. TYPE MAPPING EVIDENCE ===")
    for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"\nType Code: [{doc_type}] (Count: {count})")
        print("Samples:")
        for sample in type_samples[doc_type]:
            print(f"  - {sample[:80]}...")

    print("\n=== 2. DEGREE EXTRACTION TEST ===")
    for degree, count in sorted(degree_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{degree}: {count}")
    
    print("\n--- Unmatched Titles (Sample) ---")
    for t in unmatched_titles:
        print(f"  - {t[:80]}...")

except FileNotFoundError:
    print(f"File {input_file} not found. Please run the crawler first.")
