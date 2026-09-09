import re
from unidecode import unidecode


def slugify(text: str) -> str:
    """Μετατρέπει ένα string σε ασφαλές Pinecone ID: πεζά, ascii, underscores."""
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


def make_recipe_id(source: str, name: str) -> str:
    """Φτιάχνει unique, deterministic Pinecone ID από source (βιβλίο) + όνομα συνταγής."""
    source_slug = slugify(source)
    name_slug = slugify(name)
    return f"{source_slug}__{name_slug}"


def chunk_markdown(markdown_text: str, source: str) -> list[dict]:
    """
    Σπάει markdown σε chunks σε ΚΑΘΕ heading (οποιουδήποτε επιπέδου: #, ##, ### ...).
    Καθαρίζει το περιεχόμενο και πετάει chunks χωρίς πραγματικό περιεχόμενο ή γνωστό θόρυβο.
    """
    pattern = re.compile(r"^#+ (.+?)$\n(.*?)(?=^#+ |\Z)", re.MULTILINE | re.DOTALL)

    chunks = []
    for match in pattern.finditer(markdown_text):
        name = match.group(1).strip()
        content = match.group(2).strip()

        if name == "Index":
            continue

        content = content.replace("---PAGE_BREAK---", "").strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"<sup>(\d+)</sup>/<sub>(\d+)</sub>", r"\1/\2", content)

        if not content:
            continue

        chunks.append({
            "name": name,
            "content": content,
            "full_text": f"## {name}\n\n{content}",
            "char_count": len(content),
            "source": source,
        })

    return chunks


if __name__ == "__main__":
    id1 = make_recipe_id("straub_1914", "Old Fashioned")
    id2 = make_recipe_id("new_book_2020", "Old Fashioned")
    print(id1)
    print(id2)
    print("Διαφορετικά IDs:", id1 != id2)

import json
import time
from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic

load_dotenv()
anthropic_client = Anthropic()

ENRICHMENT_PROMPT = """You are analyzing content from a cocktail recipe book.

Heading name in the book: {name}
Content under this heading:
{content}

Extract structured metadata as a JSON ARRAY. Almost always this will contain
exactly ONE item, describing the single drink (or non-recipe content) above.

Occasionally, due to a document formatting quirk, the content above may
actually contain MORE THAN ONE distinct drink recipe merged together (no
heading separated them in the original book). If you notice this, return one
array item PER distinct recipe you can identify.

Each array item must have this shape:

{{
  "name": "the specific name of this drink -- use \\"{name}\\" if there is only one recipe, otherwise your best guess at each individual recipe's name",
  "type": "cocktail | punch | cobbler | cooler | fizz | toddy | frappé | cup | shot | other | info",
  "base_spirit": "primary spirit (e.g., gin, whiskey, rum, brandy, vermouth, wine, none)",
  "all_ingredients": ["list of all ingredient names, normalized"],
  "flavor_profile": ["list of 2-4 flavor descriptors like bitter, sweet, herbal, citrus, fruity, spicy, dry, smoky"],
  "sweetness": "none | low | medium | high",
  "strength": "low | medium | high",
  "method": "shaken | stirred | built | blended | muddled | other",
  "glassware": "cocktail | rocks | highball | wine | punch | other",
  "ingredient_count": <integer>,
  "era_style": "infer briefly, e.g. 'classic pre-prohibition', 'modern craft', 'tiki', 'unknown'"
}}

If the content is NOT a drink recipe (e.g., a restaurant name, an index
listing, a chapter title, general prose), return a single-item array with
"type" set to "info" and all other fields (except "name") set to null.

Return ONLY the JSON array, no markdown, no explanation.
"""



def parse_claude_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n", "", cleaned)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return json.loads(cleaned)


def enrich_one(chunk: dict):
    """Καλεί τον Claude για ΕΝΑ chunk. Επιστρέφει (λίστα από metadata dicts, usage)."""
    prompt = ENRICHMENT_PROMPT.format(name=chunk["name"], content=chunk["content"])
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_claude_json(response.content[0].text)

    if isinstance(result, dict):  # ασφάλεια, σε περίπτωση που επιστρέψει ένα αντικείμενο
        result = [result]

    return result, response.usage

def enrich_chunks(chunks: list[dict], source: str, enriched_path: str) -> dict:
    """
    Εμπλουτίζει όλα τα chunks μέσω Claude. Ένα chunk μπορεί να παράξει 1+ εγγραφές
    (αν κρύβει συγχωνευμένες συνταγές). Auto-save κάθε 50, resume αν ξανατρέξει.
    """
    enriched_path = Path(enriched_path)

    enriched_data = {}
    if enriched_path.exists():
        enriched_data = json.loads(enriched_path.read_text(encoding="utf-8"))
        print(f"Βρέθηκαν {len(enriched_data)} ήδη αποθηκευμένες εγγραφές")

    # Το resume-tracking γίνεται με βάση το ΑΡΧΙΚΟ chunk, όχι το τελικό όνομα συνταγής
    processed_chunk_ids = {r["chunk_id"] for r in enriched_data.values() if "chunk_id" in r}

    total_in = total_out = successful = failed = skipped = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks, 1):
        chunk_id = make_recipe_id(source, chunk["name"])

        if chunk_id in processed_chunk_ids:
            skipped += 1
            continue

        if i % 20 == 0 or i == 1:
            cost = (total_in * 0.80 + total_out * 4.00) / 1_000_000
            print(f"[{i:4d}/{len(chunks)}] {successful} ok | {failed} failed | {skipped} skip | ${cost:.3f}")

        try:
            items, usage = enrich_one(chunk)
            total_in += usage.input_tokens
            total_out += usage.output_tokens

            for item in items:
                recipe_name = item.get("name") or chunk["name"]
                recipe_id = make_recipe_id(source, recipe_name)
                enriched_data[recipe_id] = {
                    "name": recipe_name,
                    "source": source,
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "full_text": chunk["full_text"],
                    "char_count": chunk["char_count"],
                    "metadata": item,
                }
            processed_chunk_ids.add(chunk_id)
            successful += 1
        except Exception as e:
            enriched_data[chunk_id] = {
                "name": chunk["name"],
                "source": source,
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "full_text": chunk["full_text"],
                "char_count": chunk["char_count"],
                "metadata": None,
                "error": str(e)[:200],
            }

            failed += 1

        if i % 50 == 0:
            enriched_path.write_text(json.dumps(enriched_data, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(0.3)

    enriched_path.write_text(json.dumps(enriched_data, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    cost = (total_in * 0.80 + total_out * 4.00) / 1_000_000
    print(f"\nΟλοκληρώθηκε σε {elapsed/60:.1f} λεπτά -- {successful} chunks ok ({len(enriched_data)} εγγραφές συνολικά), {failed} failed, {skipped} skipped -- ${cost:.4f}")

    return enriched_data
def make_pinecone_metadata(recipe: dict) -> dict:
    """Μετατρέπει μια εμπλουτισμένη εγγραφή σε metadata έτοιμα για Pinecone."""
    meta = recipe["metadata"]
    return {
        "name": recipe["name"],
        "source": recipe["source"],
        "type": str(meta.get("type") or "unknown"),
        "base_spirit": str(meta.get("base_spirit") or "unknown"),
        "method": str(meta.get("method") or "unknown"),
        "strength": str(meta.get("strength") or "unknown"),
        "sweetness": str(meta.get("sweetness") or "unknown"),
        "ingredient_count": int(meta.get("ingredient_count") or 0),
        "flavor_profile": meta.get("flavor_profile") or [],
        "text": recipe["full_text"][:1000],
    }


def select_recipes_to_upload(enriched_data: dict) -> list[dict]:
    """Κρατάει μόνο τις πραγματικές συνταγές -- πετάει 'info' και αποτυχίες."""
    return [
        r for r in enriched_data.values()
        if r.get("metadata") and r["metadata"].get("type") not in (None, "info")
    ]
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

openai_client = OpenAI()
pc = Pinecone()

INDEX_NAME = "cocktail-rag"
EMBED_MODEL = "text-embedding-3-small"
MAX_EMBED_TOKENS = 8000


def _get_index():
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Δημιουργία index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model(EMBED_MODEL)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def embed_and_upsert(enriched_data: dict, batch_size: int = 50) -> int:
    """Ανεβάζει στο Pinecone μόνο τις πραγματικές συνταγές. Επιστρέφει πόσα vectors ανέβηκαν."""
    index = _get_index()
    recipes = select_recipes_to_upload(enriched_data)
    skipped = len(enriched_data) - len(recipes)
    print(f"Θα ανέβουν {len(recipes)} συνταγές (παραλείπονται {skipped})")

    total_uploaded = 0
    total_tokens = 0

    for batch_start in range(0, len(recipes), batch_size):
        batch = recipes[batch_start:batch_start + batch_size]
        texts = [_truncate_to_tokens(r["full_text"], MAX_EMBED_TOKENS) for r in batch]

        response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        total_tokens += response.usage.total_tokens

        vectors = []
        for i, recipe in enumerate(batch):
            recipe_id = make_recipe_id(recipe["source"], recipe["name"])
            vectors.append({
                "id": recipe_id,
                "values": response.data[i].embedding,
                "metadata": make_pinecone_metadata(recipe),
            })

        for attempt in range(3):
            try:
                index.upsert(vectors=vectors)
                break
            except Exception as e:
                print(f"  ⚠️  Batch {batch_start // batch_size + 1} timeout (προσπάθεια {attempt + 1}/3)")
                if attempt == 2:
                    raise
                time.sleep(3)

        total_uploaded += len(vectors)
        print(f"  Batch {batch_start // batch_size + 1}: {total_uploaded}/{len(recipes)}")
        time.sleep(0.5)

    cost = (total_tokens / 1_000_000) * 0.02
    print(f"\nΤέλος -- {total_uploaded} vectors, {total_tokens:,} tokens, ${cost:.4f}")
    return total_uploaded