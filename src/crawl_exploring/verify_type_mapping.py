import json
from collections import defaultdict

input_file = 'tlu_metadata_dim.jsonl'

# Collect samples for QT and SH
samples = defaultdict(list)

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            rec = json.loads(line)
            type_code = rec.get('type', [''])[0] if isinstance(rec.get('type'), list) else rec.get('type', '')
            title = rec.get('title', [''])[0] if isinstance(rec.get('title'), list) else rec.get('title', '')
            
            # Focus on QT and SH
            if type_code in ['QT', 'SH', 'QP', 'TC', 'DT', 'BC']:
                if len(samples[type_code]) < 15:
                    samples[type_code].append(title)
        except:
            continue

# Print analysis
for code in sorted(samples.keys()):
    titles = samples[code]
    print(f"\n{'='*80}")
    print(f"Type Code: [{code}] (Total samples: {len(titles)})")
    print(f"{'='*80}")
    for i, title in enumerate(titles, 1):
        # Truncate long titles
        display_title = title[:150] if len(title) > 150 else title
        print(f"{i:2d}. {display_title}")
    print()
