import json
import re
from datetime import datetime

input_file = 'tlu_metadata_dim.jsonl'
output_file = 'tlu_metadata_clean.jsonl'

# 1. Type Mapping Dictionary
TYPE_MAPPING = {
    'LA': 'Thesis',       # Luận văn/Luận án
    'BB': 'Article',      # Bài báo
    'LT': 'Lecture',      # Bài giảng
    'SH': 'Publication',  # Sách/Giáo trình/Tài liệu
    'QP': 'Standard',     # Tiêu chuẩn (TCVN/QCVN)
    'TC': 'Journal',      # Tạp chí
    'DT': 'Project',      # Đề tài NCKH
    'BC': 'Report',       # Báo cáo
    'QT': 'Standard'      # Tiêu chuẩn ngành (TCN)
}

# 2. Degree Extraction Regex
DEGREE_PATTERNS = {
    "Tiến sĩ": r"tiến sĩ|phd|doctor",
    "Thạc sĩ": r"thạc sĩ|master",
    "Kỹ sư": r"kỹ sư|engineer",
    "Cử nhân": r"cử nhân|bachelor",
    "Đồ án": r"đồ án",
    "Khóa luận": r"khóa luận"
}

def clean_record(record):
    clean = {}
    
    # --- A. Fix ID ---
    # Extract Handle ID (e.g., DHTL/6051) from URI or OAI ID
    # URI format: http://tailieuso.tlu.edu.vn/handle/DHTL/6051
    uri = record.get('identifier.uri', [''])[0]
    oai_id = record.get('oai_identifier', [''])[0]
    
    short_id = None
    if 'handle/' in uri:
        short_id = uri.split('handle/')[-1] # DHTL/6051
    elif 'DHTL/' in oai_id:
        short_id = 'DHTL/' + oai_id.split('DHTL/')[-1]
    
    # Fallback if extraction fails
    if not short_id:
        short_id = oai_id.replace('oai:localhost:', '').replace('oai:tailieuso.tlu.edu.vn:', '')
        
    clean['id'] = short_id
    clean['uri'] = uri
    clean['oai_id'] = oai_id # Keep original for reference
    
    # --- B. Map Type ---
    raw_types = record.get('type', [])
    # Take first type code if list, else string
    raw_type_code = raw_types[0] if isinstance(raw_types, list) and raw_types else str(raw_types)
    clean['type'] = TYPE_MAPPING.get(raw_type_code, 'Other')
    clean['raw_type'] = raw_type_code # Keep raw for reference
    
    # --- C. Normalize Fields (Flatten vs Array) ---
    clean['title'] = record.get('title', [''])[0]
    clean['abstract'] = record.get('description.abstract', [''])[0]
    clean['language'] = record.get('language', [''])[0]
    clean['publisher'] = record.get('publisher', ['Trường Đại học Thủy Lợi'])[0]
    
    # Keep Arrays
    clean['authors'] = record.get('contributor.author', [])
    clean['advisors'] = record.get('contributor.advisor', [])
    clean['subjects'] = record.get('subject', [])
    
    # --- D. Normalize Names (Họ, Tên -> Họ Tên) ---
    def fix_name(name_list):
        fixed = []
        for name in name_list:
            if ',' in name:
                parts = [p.strip() for p in name.split(',')]
                # "Đinh, Tuấn Hải" -> parts=["Đinh", "Tuấn Hải"] -> "Đinh Tuấn Hải"
                fixed.append(" ".join(parts))
            else:
                fixed.append(name)
        return fixed

    clean['authors'] = fix_name(clean['authors'])
    clean['advisors'] = fix_name(clean['advisors'])
    
    # --- E. Normalize Dates ---
    # Prefer date.issued -> date.available -> date.accessioned
    year = record.get('date.issued', [''])[0]
    clean['year'] = year # Often just '2019'
    
    # Parse full date if needed (from date.available)
    avail_date = record.get('date.available', [''])[0]
    clean['date'] = avail_date # Keep ISO string "2021-01-19T08:57:51Z"
    
    # --- F. Extract Degree (Only for Thesis) ---
    clean['degree'] = None
    if clean['type'] == 'Thesis':
        title_lower = clean['title'].lower()
        for degree, pattern in DEGREE_PATTERNS.items():
            if re.search(pattern, title_lower):
                clean['degree'] = degree
                break
    
    # --- G. Map Major/Department ---
    clean['major'] = record.get('subjectacademic', [''])[0]
    clean['ddc'] = record.get('subject.ddc', [''])[0]

    return clean

print(f"Cleaning data from {input_file} -> {output_file}...")
count = 0
with open(input_file, 'r', encoding='utf-8') as f_in, \
    open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        try:
            raw_rec = json.loads(line)
            clean_rec = clean_record(raw_rec)
            f_out.write(json.dumps(clean_rec, ensure_ascii=False) + '\n')
            count += 1
        except json.JSONDecodeError:
            continue

print(f"Done! Cleaned {count} records.")
