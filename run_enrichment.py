from pathlib import Path
from ingest_book import chunk_markdown, enrich_chunks

parsed_path = Path("data/parsed/tales_cocktail_2010.md")
full_markdown = parsed_path.read_text(encoding="utf-8")
chunks = chunk_markdown(full_markdown, source="tales_cocktail_2010")

print(f"Θα γίνει enrich σε {len(chunks)} chunks...")

result = enrich_chunks(
    chunks,
    source="tales_cocktail_2010",
    enriched_path="data/enriched/tales_cocktail_2010_enriched.json",
)

print(f"\nΤέλος. Συνολικές εγγραφές: {len(result)}")