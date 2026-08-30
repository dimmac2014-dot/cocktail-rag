"""
🎩 Vintage Cocktail Consultant
Web app γύρω από το RAG pipeline με τον Jack the Bartender.
"""

import os
import json
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from pinecone import Pinecone
import cohere

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🎩 Jack's Vintage Cocktails",
    page_icon="🍸",
    layout="wide"
)

load_dotenv()


# Cache clients για performance
@st.cache_resource
def init_clients():
    return {
        "anthropic": Anthropic(),
        "openai": OpenAI(),
        "pinecone": Pinecone().Index("cocktail-rag"),
        "cohere": cohere.Client(os.getenv("COHERE_API_KEY")),
    }


clients = init_clients()


# ═══════════════════════════════════════════════════════════════
# PASSWORD GATE
# ═══════════════════════════════════════════════════════════════

def check_password():
    """Returns True αν ο χρήστης έχει βάλει σωστό password."""

    def password_entered():
        if st.session_state["password"] == os.getenv("APP_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First time - show input
        st.text_input(
            "🔒 Enter password to access Jack's Bar",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.info("💡 Contact the owner for the password")
        return False
    elif not st.session_state["password_correct"]:
        # Wrong password
        st.text_input(
            "🔒 Enter password to access Jack's Bar",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("❌ Wrong password. Try again.")
        return False
    else:
        # Correct password
        return True


# Check password before showing app
if not check_password():
    st.stop()


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def parse_claude_json(raw_text):
    """Καθαρίζει markdown code blocks από Claude output."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n", "", cleaned)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return json.loads(cleaned)


def parse_query_filters(question):
    """Ζητάει από τον Claude να βγάλει filters από το query."""
    filter_prompt = f"""Analyze this cocktail question and extract search filters.

Question: "{question}"

Return a JSON object with these fields (use null if not mentioned):
- base_spirit: "gin" | "whiskey" | "rum" | "brandy" | "wine" | "vermouth" | "absinthe" | null
- recipe_type: "cocktail" | "punch" | "cooler" | "fizz" | "toddy" | "cobbler" | "frappé" | null
- max_strength: "low" | "medium" | "high" | null
- max_sweetness: "none" | "low" | "medium" | "high" | null
- search_query: rewrite as short semantic search query (2-6 words)

Return ONLY the JSON."""

    response = clients["anthropic"].messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": filter_prompt}]
    )

    try:
        filters = parse_claude_json(response.content[0].text)
        search_query = filters.pop("search_query", question)
        active_filters = {k: v for k, v in filters.items() if v is not None}
        return search_query, active_filters, response.usage
    except:
        return question, {}, response.usage


def build_pinecone_filter(active_filters):
    """Μετατρέπει filters σε Pinecone format."""
    pinecone_filter = {}
    if active_filters.get("base_spirit"):
        pinecone_filter["base_spirit"] = {"$eq": active_filters["base_spirit"]}
    if active_filters.get("recipe_type"):
        pinecone_filter["type"] = {"$eq": active_filters["recipe_type"]}
    if active_filters.get("max_strength"):
        order = ["low", "medium", "high"]
        allowed = order[:order.index(active_filters["max_strength"]) + 1]
        pinecone_filter["strength"] = {"$in": allowed}
    if active_filters.get("max_sweetness"):
        order = ["none", "low", "medium", "high"]
        allowed = order[:order.index(active_filters["max_sweetness"]) + 1]
        pinecone_filter["sweetness"] = {"$in": allowed}
    return pinecone_filter


def hybrid_search_with_rerank(query, initial_top_k=30, final_top_k=5, filter_dict=None):
    """Wide semantic search + Cohere rerank."""
    query_embedding = clients["openai"].embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    initial_results = clients["pinecone"].query(
        vector=query_embedding,
        top_k=initial_top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    if not initial_results['matches']:
        return []

    documents = []
    for match in initial_results['matches']:
        meta = match['metadata']
        doc = f"{meta['name']}\n{meta['text']}\nType: {meta['type']}, Base: {meta['base_spirit']}, Flavors: {', '.join(meta.get('flavor_profile', []))}"
        documents.append(doc)

    rerank_response = clients["cohere"].rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=min(final_top_k, len(documents))
    )

    reranked = []
    for r in rerank_response.results:
        original = initial_results['matches'][r.index]
        reranked.append({
            'match': original,
            'semantic_score': original['score'],
            'rerank_score': r.relevance_score,
        })

    return reranked


def generate_bartender_response(question, reranked_matches):
    """Ο Jack απαντάει με βάση τα retrieved recipes."""
    context = "\n\n".join([
        f"[Recipe {i}] {item['match']['metadata']['name']}\n"
        f"Type: {item['match']['metadata']['type']} | Base: {item['match']['metadata']['base_spirit']} | "
        f"Method: {item['match']['metadata']['method']} | Strength: {item['match']['metadata']['strength']} | "
        f"Sweet: {item['match']['metadata']['sweetness']}\n"
        f"Flavors: {', '.join(item['match']['metadata'].get('flavor_profile', []))}\n"
        f"{item['match']['metadata']['text']}"
        for i, item in enumerate(reranked_matches, 1)
    ])

    prompt = f"""You are Jack, a charismatic vintage bartender who has been tending bar since 1914. You have deep knowledge of classic cocktails from the golden age of American bartending. You speak with the warmth and wisdom of an old-school bartender.

A patron approaches your bar and asks: "{question}"

You have access to these highly relevant recipes (already filtered and ranked):

{context}

Craft a response that includes:

🍸 **RECOMMENDATION**: Your top pick, with brief reasoning about WHY it fits
📖 **THE RECIPE**: Present it beautifully (ingredients + method)
🥃 **BARTENDER'S NOTES**: 1-2 professional tips (glassware, garnish, temperature, timing, food pairing)
🎭 **ALTERNATIVE**: One backup option briefly

Keep it warm and personal. Respond in the same language as the question. Under 300 words."""

    response = clients["anthropic"].messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text, response.usage


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

# Header
st.title("🎩 Jack's Vintage Cocktail Bar")
st.markdown(
    "*Since 1914 — Ask Jack for expert cocktail recommendations from Jacques Straub's classic guide.*"
)
st.divider()

# Sidebar - Info
with st.sidebar:
    st.header("📚 About")
    st.markdown(
        """
        This app uses a **RAG pipeline** built on:

        - 📖 Jacques Straub's **"Drinks"** (1914)
        - 🧠 **527 recipes** with rich metadata
        - 🎯 Semantic search + reranking
        - 🎩 Jack, your AI bartender

        **Try questions like:**
        - *"Something dry with gin"*
        - *"Sweet dessert cocktail"*
        - *"Aperitif before dinner"*
        - *"Ρομαντικό ποτό για δύο"*
        """
    )
    st.divider()
    st.caption("Powered by Anthropic • OpenAI • Pinecone • Cohere • LlamaCloud")

# Main input
question = st.text_input(
    "💬 Ask Jack anything about cocktails:",
    placeholder="e.g., I want something sophisticated with whiskey for a vintage evening...",
    key="question_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🍸 Ask Jack", type="primary", use_container_width=True)

# Process query
if ask_button and question:
    # Parse filters
    with st.spinner("🧠 Understanding your request..."):
        search_query, active_filters, filter_usage = parse_query_filters(question)

    # Show filters
    if active_filters:
        filter_display = " | ".join([f"**{k}**: `{v}`" for k, v in active_filters.items()])
        st.info(f"🎯 Detected filters: {filter_display}")

    # Search + rerank
    with st.spinner("🔍 Searching Jack's recipe collection..."):
        pinecone_filter = build_pinecone_filter(active_filters)
        reranked = hybrid_search_with_rerank(
            search_query,
            initial_top_k=30,
            final_top_k=5,
            filter_dict=pinecone_filter if pinecone_filter else None
        )

        # Fallback if no results with filters
        if not reranked and pinecone_filter:
            st.warning("No results with filters. Searching without filters...")
            reranked = hybrid_search_with_rerank(search_query, initial_top_k=30, final_top_k=5)

    if not reranked:
        st.error("😔 Sorry, no matching recipes found.")
    else:
        # Generate response
        with st.spinner("🎩 Jack is thinking..."):
            answer, gen_usage = generate_bartender_response(question, reranked)

        # Display answer
        st.markdown("### 🎩 Jack says:")
        st.markdown(answer)

        st.divider()

        # Retrieved recipes (expandable)
        with st.expander(f"📚 View the {len(reranked)} recipes Jack considered"):
            for i, item in enumerate(reranked, 1):
                meta = item['match']['metadata']
                st.markdown(f"**{i}. {meta['name']}**")
                st.caption(
                    f"Type: {meta['type']} | Base: {meta['base_spirit']} | "
                    f"Method: {meta['method']} | Rerank score: {item['rerank_score']:.3f}"
                )
                st.code(meta['text'], language=None)
                st.markdown("---")

        # Cost tracker
        input_tokens = filter_usage.input_tokens + gen_usage.input_tokens
        output_tokens = filter_usage.output_tokens + gen_usage.output_tokens
        total_cost = (input_tokens * 0.80 + output_tokens * 4.00) / 1_000_000

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Cost", f"${total_cost:.4f}")
        col2.metric("📥 Input tokens", input_tokens)
        col3.metric("📤 Output tokens", output_tokens)

elif ask_button and not question:
    st.warning("👆 Please enter a question first!")