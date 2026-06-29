import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import ssl
import re

# Disable SSL verification for macOS certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Target directory
OUTPUT_DIR = "./inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Categories and how many papers to fetch for each
CATEGORIES = {
    "cs.AI": {"name": "Artificial_Intelligence", "count": 15},
    "cs.LG": {"name": "Machine_Learning", "count": 15},
    "cs.CV": {"name": "Deep_Learning_CV", "count": 15},
    "cs.CY": {"name": "Information_Technology", "count": 14},
    "cs.DB": {"name": "Information_Systems", "count": 14},
    "cs.CR": {"name": "Network_Security", "count": 14},
    "cs.SE": {"name": "Software_Engineering", "count": 14}
}

# XML namespace
NS = {'atom': 'http://www.w3.org/2005/Atom'}

def clean_old_md_files():
    print("Cleaning up old markdown files starting with arxiv_...")
    count = 0
    for file in os.listdir(OUTPUT_DIR):
        if file.startswith("arxiv_") and file.endswith(".md"):
            try:
                os.remove(os.path.join(OUTPUT_DIR, file))
                count += 1
            except Exception as e:
                print(f"Error removing {file}: {e}")
    print(f"Removed {count} old markdown files.")

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def download_pdf(pdf_url, filepath):
    # Set headers to look like a browser to avoid 403 Forbidden from arXiv
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    req = urllib.request.Request(pdf_url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filepath, 'wb') as f:
            f.write(response.read())

def fetch_category_papers(arxiv_cat, info):
    cat_name = info["name"]
    count = info["count"]
    print(f"\n==========================================")
    print(f"Fetching {count} PDFs for {cat_name} ({arxiv_cat})...")
    print(f"==========================================")
    
    # URL encode query
    query = f"cat:{arxiv_cat}"
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&max_results={count}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            entries = root.findall('atom:entry', NS)
            
            print(f"  Found {len(entries)} metadata entries on arXiv.")
            for idx, entry in enumerate(entries, 1):
                title_elem = entry.find('atom:title', NS)
                title = clean_text(title_elem.text) if title_elem is not None else "No Title"
                
                id_elem = entry.find('atom:id', NS)
                paper_url = clean_text(id_elem.text) if id_elem is not None else ""
                
                # Extract PDF ID
                paper_id = paper_url.split('/abs/')[-1].split('v')[0]
                versioned_id = paper_url.split('/abs/')[-1]
                
                # Construct PDF download link
                pdf_url = f"https://arxiv.org/pdf/{versioned_id}.pdf"
                
                filename = f"arxiv_{cat_name}_{idx:02d}_{paper_id}.pdf"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                print(f"  [{idx}/{count}] Downloading: {filename}...")
                print(f"    Title: {title[:80]}...")
                print(f"    URL: {pdf_url}")
                
                # Try downloading PDF
                success = False
                retries = 3
                for attempt in range(retries):
                    try:
                        # Wait 3 seconds before downloading to respect arXiv's rate limits
                        time.sleep(3)
                        download_pdf(pdf_url, filepath)
                        file_size_kb = os.path.getsize(filepath) / 1024
                        print(f"    -> Success ({file_size_kb:.1f} KB)")
                        success = True
                        break
                    except Exception as download_err:
                        print(f"    -> Attempt {attempt+1} failed: {download_err}")
                        if os.path.exists(filepath):
                            try:
                                os.remove(filepath)
                            except:
                                pass
                        time.sleep(5)  # Wait longer before retrying
                
                if not success:
                    print(f"    -> [FAILED] Could not download PDF for {paper_id}")
                    
    except Exception as e:
        print(f"  Error fetching {arxiv_cat}: {e}")

def main():
    total_start = time.time()
    clean_old_md_files()
    
    for cat, info in CATEGORIES.items():
        fetch_category_papers(cat, info)
        print("Waiting 5 seconds before next category...")
        time.sleep(5)
    
    # Verify how many files were saved
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("arxiv_") and f.endswith(".pdf")]
    print(f"\nDone! Successfully saved {len(files)} PDFs in '{OUTPUT_DIR}' directory.")
    print(f"Total time elapsed: {time.time() - total_start:.2f} seconds.")

if __name__ == "__main__":
    main()
