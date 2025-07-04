#!/usr/bin/env python3
"""
Check Pinecone index details using built-in libraries.
"""

import os
import json
import urllib.request
import ssl
from pathlib import Path

# Load .env
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

api_key = os.getenv("PINECONE_API_KEY")
index_name = "mighty-walnut"  # The existing index

print(f"Checking Pinecone index: {index_name}")
print("=" * 60)

# Get index stats
try:
    # First, get the index host
    url = f"https://api.pinecone.io/indexes/{index_name}"
    req = urllib.request.Request(url)
    req.add_header("Api-Key", api_key)
    
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, context=context) as response:
        index_info = json.loads(response.read())
        host = index_info['host']
        print(f"Index host: {host}")
        print(f"Dimension: {index_info['dimension']}")
        print(f"Metric: {index_info['metric']}")
        print(f"Status: {index_info['status']['state']}")
    
    # Get index stats from the data plane
    stats_url = f"https://{host}/describe_index_stats"
    stats_req = urllib.request.Request(stats_url)
    stats_req.add_header("Api-Key", api_key)
    stats_req.add_header("Content-Type", "application/json")
    
    with urllib.request.urlopen(stats_req, context=context) as response:
        stats = json.loads(response.read())
        print(f"\nIndex Statistics:")
        print(f"Total vectors: {stats.get('totalVectorCount', 0)}")
        
        if 'namespaces' in stats:
            print(f"Namespaces: {len(stats['namespaces'])}")
            for ns, ns_stats in stats['namespaces'].items():
                print(f"  - {ns}: {ns_stats.get('vectorCount', 0)} vectors")
    
except Exception as e:
    print(f"Error: {str(e)}")

print("\n" + "=" * 60)
print("Note: This index has 3072 dimensions, which suggests it's using")
print("a different embedding model than our configured all-MiniLM-L6-v2 (384D).")
print("This might be from a different project.")
print("\nOur system is configured to use index: ijon-pdfs")
print("=" * 60)