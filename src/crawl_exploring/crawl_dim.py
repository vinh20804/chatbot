from sickle import Sickle
import json
import xml.etree.ElementTree as ET
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
sickle = Sickle('https://tailieuso.tlu.edu.vn/oai/request', verify=False, timeout=60)
output_file = 'tlu_metadata_dim.jsonl'
max_records = 5000 

def parse_dim_record(raw_xml):
    """Parses raw XML of a DIM record into a dictionary."""
    try:
        root = ET.fromstring(raw_xml)
        # DIM Namespace
        ns = {'dim': 'http://www.dspace.org/xmlns/dspace/dim'}
        
        # Find <dim:dim> inside <metadata>
        dim_node = root.find('.//dim:dim', ns)
        if dim_node is None:
            return {}
            
        data = {}
        
        # Iterate over all <dim:field> elements
        for field in dim_node.findall('dim:field', ns):
            element = field.get('element')
            qualifier = field.get('qualifier')
            value = field.text
            
            # Construct key: "element" or "element.qualifier"
            key = element
            if qualifier:
                key = f"{element}.{qualifier}"
                
            # Store in dict, using list for multiple values
            if key not in data:
                data[key] = []
            if value:
                data[key].append(value)
            
        return data
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return {}

print(f"Starting crawl (DIM format) -> {output_file}...")

count = 0
with open(output_file, 'w', encoding='utf-8') as f:
    try:
        # Fetch records with metadataPrefix='dim'
        records = sickle.ListRecords(metadataPrefix='dim', ignore_deleted=True)
        
        for record in records:
            # Parse raw XML
            meta = parse_dim_record(record.raw)
            
            # Add OAI identifier
            if record.header.identifier:
                meta['oai_identifier'] = [record.header.identifier]
            
            # Write to file (JSON Lines)
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
            f.flush() # Force write to disk
            
            count += 1
            if count % 10 == 0: # Print more frequently
                print(f"Fetched {count} records...", flush=True)
            
            if count >= max_records:
                break
                
    except Exception as e:
        print(f"Error or end of stream: {e}")

print(f"Done! Saved {count} records to {output_file}")
