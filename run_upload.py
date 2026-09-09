import json
from ingest_book import embed_and_upsert

data = json.loads(open("data/enriched/tales_cocktail_2010_enriched.json", encoding="utf-8").read())
uploaded = embed_and_upsert(data)
print(f"\nΟλοκληρώθηκε: {uploaded} vectors ανέβηκαν στο Pinecone.")