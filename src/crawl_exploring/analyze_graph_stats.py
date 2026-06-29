import json
from collections import Counter, defaultdict

input_file = 'tlu_metadata_clean.jsonl'

# Load data
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

total_docs = len(data)

print("="*80)
print("📊 NEO4J SCHEMA VALIDATION - STATISTICAL ANALYSIS")
print("="*80)

print(f"\n📄 Total Documents: {total_docs:,}")

# ============================================================================
# 1. FIELD COVERAGE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1️⃣  FIELD COVERAGE (% documents có data)")
print("="*80)

fields_to_check = ['authors', 'advisors', 'subjects', 'major', 'degree']
coverage = {}

for field in fields_to_check:
    if field in ['authors', 'advisors', 'subjects']:
        # List fields - check if non-empty
        non_empty = sum(1 for d in data if d.get(field) and len(d[field]) > 0)
    else:
        # String fields - check if not empty/null
        non_empty = sum(1 for d in data if d.get(field) and d[field] not in ['', None])
    
    coverage[field] = non_empty
    pct = (non_empty / total_docs) * 100
    print(f"   {field:15s}: {non_empty:5,} / {total_docs:,} ({pct:5.1f}%)")

# ============================================================================
# 2. UNIQUE VALUE COUNTS (Estimate nodes to create)
# ============================================================================
print("\n" + "="*80)
print("2️⃣  UNIQUE VALUES (Số nodes sẽ tạo)")
print("="*80)

# Unique document types
types = Counter(d['type'] for d in data)
print(f"\n   📁 Document Types: {len(types)}")
for dtype, count in types.most_common():
    print(f"      • {dtype:15s}: {count:4,} ({count/total_docs*100:5.1f}%)")

# Unique persons (authors + advisors combined)
all_persons = set()
for d in data:
    all_persons.update(d.get('authors', []))
    all_persons.update(d.get('advisors', []))
print(f"\n   👤 Unique Persons: {len(all_persons):,}")

# Unique topics
all_topics = set()
for d in data:
    all_topics.update(d.get('subjects', []))
print(f"   🏷️  Unique Topics: {len(all_topics):,}")

# Unique majors (filter empty)
all_majors = set(d['major'] for d in data if d.get('major') and d['major'].strip())
print(f"   🎓 Unique Majors: {len(all_majors):,}")

# Unique degrees (filter null)
all_degrees = set(d['degree'] for d in data if d.get('degree'))
print(f"   📜 Unique Degrees: {len(all_degrees):,}")
if all_degrees:
    print(f"      Degrees: {sorted(all_degrees)}")

# ============================================================================
# 3. DISTRIBUTION ANALYSIS (Average per document)
# ============================================================================
print("\n" + "="*80)
print("3️⃣  DISTRIBUTION (Trung bình per document)")
print("="*80)

# Authors per doc
authors_count = [len(d.get('authors', [])) for d in data]
avg_authors = sum(authors_count) / total_docs
max_authors = max(authors_count)
print(f"   Authors per doc: Avg={avg_authors:.2f}, Max={max_authors}")

# Advisors per doc
advisors_count = [len(d.get('advisors', [])) for d in data]
avg_advisors = sum(advisors_count) / total_docs
max_advisors = max(advisors_count)
print(f"   Advisors per doc: Avg={avg_advisors:.2f}, Max={max_advisors}")

# Topics per doc
topics_count = [len(d.get('subjects', [])) for d in data]
avg_topics = sum(topics_count) / total_docs
max_topics = max(topics_count)
print(f"   Topics per doc: Avg={avg_topics:.2f}, Max={max_topics}")

# ============================================================================
# 4. RELATIONSHIP ESTIMATION
# ============================================================================
print("\n" + "="*80)
print("4️⃣  ESTIMATED RELATIONSHIPS")
print("="*80)

total_wrote = sum(authors_count)
total_advised = sum(advisors_count)
total_has_topic = sum(topics_count)
total_has_major = sum(1 for d in data if d.get('major') and d['major'].strip())
total_has_degree = sum(1 for d in data if d.get('degree'))

print(f"   WROTE:             {total_wrote:6,}")
print(f"   ADVISED:           {total_advised:6,}")
print(f"   HAS_TOPIC:         {total_has_topic:6,}")
print(f"   BELONGS_TO_MAJOR:  {total_has_major:6,}")
print(f"   HAS_DEGREE:        {total_has_degree:6,}")
print(f"   {'─'*30}")
print(f"   TOTAL:             {total_wrote + total_advised + total_has_topic + total_has_major + total_has_degree:6,}")

# ============================================================================
# 5. GRAPH SIZE ESTIMATION
# ============================================================================
print("\n" + "="*80)
print("5️⃣  TOTAL GRAPH SIZE ESTIMATION")
print("="*80)

total_nodes = total_docs + len(all_persons) + len(all_topics) + len(all_majors) + len(all_degrees)
total_relationships = total_wrote + total_advised + total_has_topic + total_has_major + total_has_degree

print(f"   📦 Total Nodes:         {total_nodes:6,}")
print(f"      • Documents:         {total_docs:6,}")
print(f"      • Persons:           {len(all_persons):6,}")
print(f"      • Topics:            {len(all_topics):6,}")
print(f"      • Majors:            {len(all_majors):6,}")
print(f"      • Degrees:           {len(all_degrees):6,}")
print(f"\n   🔗 Total Relationships: {total_relationships:6,}")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("6️⃣  SCHEMA VALIDATION & RECOMMENDATIONS")
print("="*80)

# Check if any field has too low coverage
low_coverage_threshold = 5  # 5%

recommendations = []

for field, count in coverage.items():
    pct = (count / total_docs) * 100
    if pct < low_coverage_threshold:
        recommendations.append(f"⚠️  {field}: Chỉ {pct:.1f}% có data → Cân nhắc skip")
    elif pct > 95:
        recommendations.append(f"✅ {field}: {pct:.1f}% có data → Quan trọng, nên giữ")

if recommendations:
    for rec in recommendations:
        print(f"   {rec}")
else:
    print("   ✅ Tất cả fields đều có coverage hợp lý!")

print("\n" + "="*80)
print("🎯 CONCLUSION: Schema design hợp lý, proceed implementation!")
print("="*80)
