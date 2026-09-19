"""
🎩 Vintage Cocktail Consultant
Web app γύρω από το RAG pipeline με τον Jack the Bartender.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

import requests
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
# WEATHER (Open-Meteo — δωρεάν, χωρίς API key)
# ═══════════════════════════════════════════════════════════════

WEATHER_CODES = {
    0: "αίθριος", 1: "κυρίως αίθριος", 2: "μερική συννεφιά", 3: "συννεφιά",
    45: "ομίχλη", 48: "παγωμένη ομίχλη",
    51: "ψιλόβροχο", 53: "ψιλόβροχο", 55: "έντονο ψιλόβροχο",
    56: "παγωμένο ψιλόβροχο", 57: "έντονο παγωμένο ψιλόβροχο",
    61: "ελαφριά βροχή", 63: "βροχή", 65: "δυνατή βροχή",
    66: "παγωμένη βροχή", 67: "δυνατή παγωμένη βροχή",
    71: "ελαφριά χιονόπτωση", 73: "χιονόπτωση", 75: "δυνατή χιονόπτωση",
    77: "χιονόκοκκοι",
    80: "μπόρες", 81: "δυνατές μπόρες", 82: "καταιγιδώδεις μπόρες",
    85: "χιονομπόρες", 86: "δυνατές χιονομπόρες",
    95: "καταιγίδα", 96: "καταιγίδα με χαλάζι", 99: "ισχυρή καταιγίδα με χαλάζι",
}


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(city: str):
    """
    Φέρνει τον τρέχοντα καιρό για μια πόλη μέσω Open-Meteo.
    Επιστρέφει dict {city, temperature, description} ή None αν κάτι πάει στραβά
    (άγνωστη πόλη, timeout, API down) — ποτέ δεν σκάει, απλά αγνοείται το weather context.
    """
    if not city or not city.strip():
        return None

    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city.strip(), "count": 1, "language": "el", "format": "json"},
            timeout=5,
        ).json()

        results = geo.get("results")
        if not results:
            return None

        lat, lon = results[0]["latitude"], results[0]["longitude"]
        resolved_name = results[0]["name"]

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code",
                "timezone": "auto",
            },
            timeout=5,
        ).json()

        current = forecast.get("current")
        if not current or current.get("temperature_2m") is None:
            return None

        return {
            "city": resolved_name,
            "temperature": current["temperature_2m"],
            "description": WEATHER_CODES.get(current.get("weather_code"), "άγνωστος καιρός"),
            "local_time": current.get("time"),  # ISO string, τοπική ώρα της πόλης (timezone=auto)
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# DATETIME CONTEXT (χωρίς API, μηδενικό κόστος)
# ═══════════════════════════════════════════════════════════════

DAYS_EL = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]

SPECIAL_OCCASIONS = {
    (12, 31): "Παραμονή Πρωτοχρονιάς",
    (1, 1): "Πρωτοχρονιά",
    (12, 24): "Παραμονή Χριστουγέννων",
    (12, 25): "Χριστούγεννα",
    (2, 14): "Ημέρα του Αγίου Βαλεντίνου",
    (10, 31): "Halloween",
    (3, 25): "25η Μαρτίου",
}


def get_datetime_context(local_time_str: str | None = None):
    """
    Υπολογίζει context από την ημερομηνία/ώρα -- ημέρα, στιγμή της ημέρας,
    εποχή, αν είναι Σαββατοκύριακο, και τυχόν ειδική γιορτή.
    Αν δοθεί local_time_str (ISO string από το Open-Meteo, τοπική ώρα του
    χρήστη), το χρησιμοποιεί -- αλλιώς πέφτει πίσω στην ώρα του server.
    Καθαρός υπολογισμός, καμία εξωτερική κλήση.
    """
    if local_time_str:
        try:
            now = datetime.fromisoformat(local_time_str)
        except ValueError:
            now = datetime.now()
    else:
        now = datetime.now()

    hour = now.hour
    if 5 <= hour < 12:
        time_of_day = "πρωί"
    elif 12 <= hour < 17:
        time_of_day = "μεσημέρι"
    elif 17 <= hour < 23:
        time_of_day = "βράδυ"
    else:
        time_of_day = "νύχτα"

    month = now.month
    if month in (12, 1, 2):
        season = "χειμώνας"
    elif month in (3, 4, 5):
        season = "άνοιξη"
    elif month in (6, 7, 8):
        season = "καλοκαίρι"
    else:
        season = "φθινόπωρο"

    weekday_idx = now.weekday()  # 0 = Δευτέρα ... 6 = Κυριακή

    return {
        "day_of_week": DAYS_EL[weekday_idx],
        "time_of_day": time_of_day,
        "season": season,
        "is_weekend": weekday_idx >= 5,
        "special_occasion": SPECIAL_OCCASIONS.get((now.month, now.day)),
    }


# ═══════════════════════════════════════════════════════════════
# UNIT CONVERTER (bartending units — χωρίς API, μηδενικό κόστος)
# ═══════════════════════════════════════════════════════════════

# Όλα μετατρέπονται μέσω κοινής βάσης: χιλιοστόλιτρα (ml).
# Οι παλιές μονάδες (jigger, pony, dash, wine glass) είναι ιστορικά bar
# standards -- οι τιμές παρακάτω είναι οι πιο συνηθισμένες σύγχρονες συμβάσεις.
UNIT_TO_ML = {
    "ml": 1.0,
    "cl": 10.0,
    "l": 1000.0,
    "oz": 29.5735,       # fluid ounce (US)
    "jigger": 44.36,     # 1.5 oz -- ο πιο κοινός σύγχρονος ορισμός
    "pony": 29.5735,     # 1 oz
    "dash": 0.92,        # ≈ 1/32 oz -- κατά προσέγγιση, διαφέρει ανά μπάρμαν
    "tsp": 5.0,           # teaspoon
    "tbsp": 15.0,         # tablespoon
    "cup": 236.588,
    "wine_glass": 59.15,  # ≈ 2 oz, ιστορικό bar measure -- κατά προσέγγιση
}

UNIT_LABELS = {
    "ml": "ml", "cl": "cl", "l": "λίτρα", "oz": "oz (fl. ounce)",
    "jigger": "jigger", "pony": "pony", "dash": "dash",
    "tsp": "τσαγιού (tsp)", "tbsp": "σούπας (tbsp)", "cup": "cup",
    "wine_glass": "wine glass (ιστορικό)",
}


def convert_units(amount: float, from_unit: str, to_unit: str):
    """
    Μετατρέπει ποσότητα μεταξύ bartending μονάδων (oz, ml, cl, jigger, pony,
    dash, tsp, tbsp, cup, wine_glass). Επιστρέφει float ή None αν η μονάδα
    είναι άγνωστη. Καθαρός υπολογισμός -- όχι API call, μηδενικό κόστος.
    """
    if from_unit not in UNIT_TO_ML or to_unit not in UNIT_TO_ML:
        return None
    ml = amount * UNIT_TO_ML[from_unit]
    return ml / UNIT_TO_ML[to_unit]


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


def generate_bartender_response(question, reranked_matches, weather=None, dt_context=None):
    """Ο Jack απαντάει με βάση τα retrieved recipes (και προαιρετικά καιρό + ώρα/ημέρα)."""
    context = "\n\n".join([
        f"[Recipe {i}] {item['match']['metadata']['name']}\n"
        f"Type: {item['match']['metadata']['type']} | Base: {item['match']['metadata']['base_spirit']} | "
        f"Method: {item['match']['metadata']['method']} | Strength: {item['match']['metadata']['strength']} | "
        f"Sweet: {item['match']['metadata']['sweetness']}\n"
        f"Flavors: {', '.join(item['match']['metadata'].get('flavor_profile', []))}\n"
        f"{item['match']['metadata']['text']}"
        for i, item in enumerate(reranked_matches, 1)
    ])

    weather_line = ""
    if weather:
        weather_line = (
            f'\nRight now, where the patron is ({weather["city"]}), the weather is '
            f'{weather["temperature"]}°C and {weather["description"]}. '
            f"Let this subtly influence your recommendation (e.g. lean refreshing/citrus/lower-ABV in hot weather, "
            f"warming/spirit-forward in cold weather) — but don't force it if the patron already asked for something specific.\n"
        )

    dt_line = ""
    if dt_context:
        occasion = dt_context.get("special_occasion")
        weekend_note = " It's the weekend." if dt_context["is_weekend"] else ""
        occasion_note = f" Tonight is {occasion} — feel free to suggest something festive if it fits!" if occasion else ""
        dt_line = (
            f"\nIt's currently {dt_context['time_of_day']} on a {dt_context['day_of_week']}, in {dt_context['season']}."
            f"{weekend_note}{occasion_note} "
            f"Let this subtly inform your tone/suggestion (e.g. lighter for a weekday afternoon, more festive for a "
            f"weekend evening) — but don't force it if the patron already asked for something specific.\n"
        )

    prompt = f"""You are Jack, a charismatic vintage bartender who has been tending bar since 1914. You have deep knowledge of classic cocktails from the golden age of American bartending. You speak with the warmth and wisdom of an old-school bartender.

A patron approaches your bar and asks: "{question}"
{weather_line}{dt_line}
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
    "*From 1914 to 2010 — Ask Jack for expert cocktail recommendations from two classic bartending guides*."
)
st.divider()

# Sidebar - Info
with st.sidebar:
    st.header("📚 About")
    st.markdown(
        """
        This app uses a **RAG pipeline** built on:

        - 📖 Jacques Straub's **"Drinks"** (1914) - 527 recipes
        - 🍹 **Tales of the Cocktail** Recipe Book (2010) — 652 recipes
        - 🧠 **1.179 recipes** with rich metadata
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
    st.subheader("📍 Ο καιρός σου")
    weather_city = st.text_input(
        "Πόλη (για context-aware προτάσεις)",
        value="Athens",
        key="weather_city",
    )

    st.divider()
    with st.expander("🥄 Μετατροπέας Μονάδων"):
        st.caption("Χρήσιμο για τις παλιές μονάδες του 1914 (jigger, pony, dash...)")
        unit_keys = list(UNIT_TO_ML.keys())

        conv_amount = st.number_input("Ποσότητα", min_value=0.0, value=1.0, step=0.5, key="conv_amount")
        conv_from = st.selectbox(
            "Από", unit_keys, index=unit_keys.index("jigger"),
            format_func=lambda u: UNIT_LABELS[u], key="conv_from",
        )
        conv_to = st.selectbox(
            "Σε", unit_keys, index=unit_keys.index("ml"),
            format_func=lambda u: UNIT_LABELS[u], key="conv_to",
        )

        result = convert_units(conv_amount, conv_from, conv_to)
        if result is not None:
            st.success(f"{conv_amount:g} {UNIT_LABELS[conv_from]} = **{result:.2f} {UNIT_LABELS[conv_to]}**")

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
        # Weather context (προαιρετικό — αν αποτύχει, απλά αγνοείται)
        weather = get_weather(st.session_state.get("weather_city", "Athens"))
        if weather:
            st.caption(f"🌤️ {weather['city']}: {weather['temperature']}°C, {weather['description']}")

        # Datetime context (χρησιμοποιεί την τοπική ώρα της πόλης αν έχουμε weather, αλλιώς ώρα server)
        dt_context = get_datetime_context(weather.get("local_time") if weather else None)
        occasion_badge = f" • 🎉 {dt_context['special_occasion']}" if dt_context["special_occasion"] else ""
        st.caption(f"🕒 {dt_context['day_of_week']} {dt_context['time_of_day']}, {dt_context['season']}{occasion_badge}")

        # Generate response
        with st.spinner("🎩 Jack is thinking..."):
            answer, gen_usage = generate_bartender_response(question, reranked, weather, dt_context)

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