# to curate (organize) the wiki dataset used for training Kai as a personal assistant as of 01/02/2025
# update if needed

import os
import pandas as pd
import random
from wikiextractor import WikiExtractor  # Run WikiExtractor externally first

wiki_dir = "wiki_text"  # Output from WikiExtractor
output_file = "data/datasets/wiki_curated.csv"

# Gather all extracted files
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(wiki_dir) for f in filenames]
sample_files = random.sample(files, min(10000, len(files)))  # 10,000 articles

data = []
for file in sample_files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if len(text) > 50:  # Decent length
            # First line often has title-like text
            lines = text.split('\n')
            title = lines[0].strip()[:100] if lines else "Article"
            desc = text[:500]  # Cap description
            data.append({
                "Author": "Wikipedia",
                "Title": title,
                "Description": desc,
                "Genre": "Info",
                "Banned": ""
            })

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"Created {output_file} with {len(df)} samples")