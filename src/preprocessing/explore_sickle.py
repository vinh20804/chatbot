from sickle import Sickle
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

print("Initializing Sickle...")
# Add timeout to avoid hanging
sickle = Sickle('https://tailieuso.tlu.edu.vn/oai/request', verify=False, timeout=30)

print("--- Identify ---")
try:
    identify = sickle.Identify()
    print(identify.raw)
except Exception as e:
    print(f"Error Identify: {e}")

print("\n--- ListMetadataFormats ---")
try:
    formats = sickle.ListMetadataFormats()
    for f in formats:
        print(f"Prefix: {f.metadataPrefix}, Namespace: {f.metadataNamespace}")
except Exception as e:
    print(f"Error ListMetadataFormats: {e}")

print("\n--- ListSets ---")
try:
    sets = sickle.ListSets()
    count = 0
    for s in sets:
        print(f"SetSpec: {s.setSpec}, SetName: {s.setName}")
        count += 1
        if count >= 10:
            print("... (showing first 10 sets)")
            break
except Exception as e:
    print(f"Error ListSets: {e}")
