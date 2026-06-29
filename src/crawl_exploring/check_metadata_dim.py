from sickle import Sickle
import json

sickle = Sickle('https://tailieuso.tlu.edu.vn/oai/request', verify=False)

print("Fetching one record with metadataPrefix='dim'...")
try:
    records = sickle.ListRecords(metadataPrefix='dim', ignore_deleted=True)
    
    for record in records:
        print("--- Raw XML ---")
        print(record.raw)
        print("\n--- Parsed Metadata (Sickle default) ---")
        try:
            print(record.metadata)
        except Exception as e:
            print(f"Sickle metadata access error: {e}")
            
        # Stop after 1 record
        break
        
except Exception as e:
    print(f"Error: {e}")
