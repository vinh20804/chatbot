import json

input_file = 'tlu_metadata_dim.jsonl'

# Counters
qp_quy_pham = []
qp_quy_trinh = []
qp_tieu_chuan = []
qt_quy_pham = []
qt_quy_trinh = []
qt_tieu_chuan = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            rec = json.loads(line)
            type_code = rec.get('type', [''])[0] if isinstance(rec.get('type'), list) else rec.get('type', '')
            title = rec.get('title', [''])[0] if isinstance(rec.get('title'), list) else rec.get('title', '')
            title_lower = title.lower()
            
            if type_code == 'QP':
                if 'quy phạm' in title_lower or 'quy phap' in title_lower:
                    qp_quy_pham.append(title)
                if 'quy trình' in title_lower or 'quy trinh' in title_lower:
                    qp_quy_trinh.append(title)
                if 'tiêu chuẩn' in title_lower or 'tieu chuan' in title_lower or 'tcvn' in title_lower or 'tcxd' in title_lower:
                    qp_tieu_chuan.append(title)
                    
            elif type_code == 'QT':
                if 'quy phạm' in title_lower or 'quy phap' in title_lower:
                    qt_quy_pham.append(title)
                if 'quy trình' in title_lower or 'quy trinh' in title_lower:
                    qt_quy_trinh.append(title)
                if 'tiêu chuẩn' in title_lower or 'tieu chuan' in title_lower or 'tcvn' in title_lower or 'tcxd' in title_lower:
                    qt_tieu_chuan.append(title)
        except:
            continue

print("="*80)
print("PHÂN TÍCH QP vs QT: Quy phạm / Quy trình / Tiêu chuẩn")
print("="*80)

print(f"\n📊 QP (Quy chuẩn?):")
print(f"   - Chứa 'Quy phạm': {len(qp_quy_pham)} records")
print(f"   - Chứa 'Quy trình': {len(qp_quy_trinh)} records")
print(f"   - Chứa 'Tiêu chuẩn/TCVN': {len(qp_tieu_chuan)} records")

if qp_quy_pham:
    print(f"\n   Ví dụ 'Quy phạm' trong QP:")
    for title in qp_quy_pham[:3]:
        print(f"   - {title[:100]}...")

if qp_quy_trinh:
    print(f"\n   Ví dụ 'Quy trình' trong QP:")
    for title in qp_quy_trinh[:3]:
        print(f"   - {title[:100]}...")

print(f"\n📊 QT (Quy trình?):")
print(f"   - Chứa 'Quy phạm': {len(qt_quy_pham)} records")
print(f"   - Chứa 'Quy trình': {len(qt_quy_trinh)} records")
print(f"   - Chứa 'Tiêu chuẩn/TCVN': {len(qt_tieu_chuan)} records")

if qt_quy_pham:
    print(f"\n   Ví dụ 'Quy phạm' trong QT:")
    for title in qt_quy_pham[:3]:
        print(f"   - {title[:100]}...")

if qt_quy_trinh:
    print(f"\n   Ví dụ 'Quy trình' trong QT:")
    for title in qt_quy_trinh[:3]:
        print(f"   - {title[:100]}...")

print("\n" + "="*80)
print("KẾT LUẬN:")
print("="*80)
if len(qp_tieu_chuan) > len(qp_quy_pham) and len(qt_tieu_chuan) > len(qt_quy_trinh):
    print("✅ Cả QP và QT đều chủ yếu là TIÊU CHUẨN (Standards)")
    print("❌ KHÔNG phân biệt 'Quy phạm' vs 'Quy trình' như lý thuyết thư viện")
else:
    print("✅ Có khả năng TLU phân biệt QP (Quy phạm) và QT (Quy trình)")
