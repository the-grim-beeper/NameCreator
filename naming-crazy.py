import ollama
import json
import datetime
import re
import os
import time

# --- NLTK Imports ---
import nltk
import random

# --- NLTK Custom Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_NLTK_DATA_PATH = os.path.join(SCRIPT_DIR, 'project_nltk_data')

if CUSTOM_NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, CUSTOM_NLTK_DATA_PATH)

os.makedirs(CUSTOM_NLTK_DATA_PATH, exist_ok=True)
# --- End NLTK Custom Path Configuration ---

# For loading .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ .env file loaded (if it exists).")
except ImportError:
    print("⚠️ 'python-dotenv' library not found. Please install it: pip install python-dotenv")
    print("   Environment variables should be set manually if .env is not used.")

# For Google Custom Search API
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    GOOGLE_API_CLIENT_AVAILABLE = True
except ImportError:
    print(
        "🔴 'google-api-python-client' library not found. Please install it: pip install --upgrade google-api-python-client")
    print("   Web checking via Google API will be disabled.")
    GOOGLE_API_CLIENT_AVAILABLE = False

# --- Configuration ---
GENERATOR_MODEL = 'llama3.2'  # Or your preferred model
CRITIC_BASE_MODEL = 'llama3.2'  # Or your preferred model
SYNTHESIS_CRITIC_MODEL = 'gemma3'  # Or a larger model like llama3:70b for complex reasoning
# CHIEF_RANKING_OFFICER_MODEL is implicitly SYNTHESIS_CRITIC_MODEL for now, can be made separate

MAX_NAMES_TO_CRITIQUE = 15
# API_REQUEST_DELAY = 1.0 # Commented out as Google API calls are bypassed

# Google API Configuration Status
GOOGLE_API_CONFIGURED = False
if GOOGLE_API_CLIENT_AVAILABLE:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        GOOGLE_API_CONFIGURED = True
        print("✅ Google Custom Search API configured (API Key and CSE ID found).")
    else:
        print("🔴 Google Custom Search API client is available, but credentials are not fully configured.")
        if not GOOGLE_API_KEY: print("   Missing GOOGLE_API_KEY")
        if not GOOGLE_CSE_ID: print("   Missing GOOGLE_CSE_ID")
        # GOOGLE_API_CONFIGURED remains False, web checking effectively disabled
else:
    print("   Web checking via Google API is disabled (client library not found).")

print("🔴🔴🔴 GOOGLE WEB CHECKING IS CURRENTLY COMMENTED OUT IN THE SCRIPT FOR REFINEMENT. 🔴🔴🔴")

# --- NLTK Global Word Bank ---
GLOBAL_NLTK_WORD_BANK = []

# --- Critic Personalities (Standard Critics) ---
CRITIC_PERSONALITIES = [
    {
        "name": "Mr. Alistair Finch (CEO)", "emoji": "👴",
        "system_prompt": "You are Alistair Finch, a pragmatic CEO. Value professional, trustworthy, marketable names. Avoid trendy, frivolous, or hard-to-pronounce names. Focus on longevity and brand potential.",
        "model": CRITIC_BASE_MODEL, "temperature": 0.6
    },
    {
        "name": "Dr. Evelyn Hayes (Social Advocate)", "emoji": "🌍",
        "system_prompt": "You are Dr. Evelyn Hayes, a progressive social advocate. Appreciate inclusive, thoughtful, unique names, possibly hinting at ethical values. Critical of generic, exploitative, or insensitive names. Value creativity and meaning.",
        "model": CRITIC_BASE_MODEL, "temperature": 1.0
    },
    {
        "name": "Zip (The Trendsetter)", "emoji": "😎",
        "system_prompt": "You are Zip, a 17-year-old trendsetter. Like catchy, fresh, edgy, short, 'shareable' names. Dislike old-fashioned, boring, formal, or try-hard names. Think viral potential and modern appeal.",
        "model": CRITIC_BASE_MODEL, "temperature": 1.2
    },
    {
        "name": "Seraphina \"Sparkle\" Moon (Name Champion)", "emoji": "✨",
        "system_prompt": "You are Seraphina \"Sparkle\" Moon, an incredibly enthusiastic Name Champion! Find reasons to love every name. Highlight unique charm, imaginative possibilities, positive feelings. Be warm, generous, uplifting. Frame challenges as unique characteristics. Score generously.",
        "model": CRITIC_BASE_MODEL, "temperature": 1.1
    }
]

# --- Synthesis Critic Profile (Dr. Synthia Verdict) ---
SYNTHESIS_CRITIC_PROFILE = {
    "name": "Dr. Synthia Verdict (Lead Synthesizer)",
    "emoji": "🧐",
    "system_prompt": (
        "You are Dr. Synthia Verdict, a highly analytical Lead Synthesizer. Your role is to synthesize feedback from a diverse panel of critics "
        "regarding a proposed name for a given theme. You will be provided with the theme, the name, and a list of critiques from other analysts. "
        "Your task is to: \n"
        "1. Briefly summarize the key convergent and divergent points from the provided critiques. \n"
        "2. Evaluate the name's overall suitability for the theme, considering the summarized feedback. \n"
        "3. Provide a clear, final textual verdict (e.g., 'Highly Promising', 'Promising with Reservations', 'Significant Concerns', 'Not Recommended for this Theme'). \n"
        "Focus on actionable insights and a decisive conclusion for THIS SPECIFIC NAME."
    ),
    "model": SYNTHESIS_CRITIC_MODEL, "temperature": 0.4
}

# --- Chief Ranking Officer Profile ---
CHIEF_RANKING_OFFICER_PROFILE = {
    "name": "The Boardroom Oracle (Global Ranker)",
    "emoji": "⚖️🏆",
    "system_prompt": (
        "You are The Boardroom Oracle, a decisive C-suite executive known for strategic insight. You have been presented with a list of candidate names, "
        "each accompanied by a synthesis summary and a qualitative verdict from Dr. Synthia Verdict, the Lead Synthesizer. "
        "The overarching theme for these names is also provided. Your sole and critical task is to review ALL these names and their synthesized evaluations "
        "in the context of the given theme. Then, produce a definitive final ranking of these names, from best to worst, based on their overall strategic potential, "
        "brandability, and alignment with the theme. For the top 3 ranked names, provide a very concise (one-sentence) justification for their position. "
        "Your output MUST be a JSON object containing a single key 'ranked_finalists', which is a list of objects. Each object in the list must have 'rank' (integer), "
        "'name' (string), and 'justification' (string, can be empty or null for ranks below top 3)."
    # Added null possibility for justification
    ),
    "model": SYNTHESIS_CRITIC_MODEL,  # Using same model as synthesis, can be different
    "temperature": 0.3
}


# --- NLTK Functions ---
def download_nltk_resources():
    print(f"ℹ️ NLTK will attempt to use project-specific data path: {CUSTOM_NLTK_DATA_PATH}")
    print(f"ℹ️ Full NLTK search path list: {nltk.data.path}")
    resources_to_download = {'wordnet': 'corpora/wordnet', 'omw-1.4': 'corpora/omw-1.4'}
    all_successful = True
    for name, path_suffix in resources_to_download.items():
        try:
            nltk.data.find(path_suffix)
            print(f"✅ NLTK resource '{name}' ({path_suffix}) already available.")
        except LookupError:
            print(
                f"⏳ NLTK resource '{name}' ({path_suffix}) not found. Attempting download to {CUSTOM_NLTK_DATA_PATH}...")
            try:
                nltk.download(name, download_dir=CUSTOM_NLTK_DATA_PATH, quiet=False, raise_on_error=True)
                print(f"   Verifying '{name}' after download...")
                nltk.data.find(path_suffix)
                print(f"✅ Successfully downloaded and verified NLTK resource '{name}'.")
            except Exception as e_download:
                print(f"❌ Failed to download/verify NLTK resource '{name}'. Error: {e_download}")
                all_successful = False
        except Exception as e_check:
            print(f"❌ Error checking NLTK resource '{name}'. Error: {e_check}")
            all_successful = False
    return all_successful


def load_nltk_word_bank(min_len=4, max_len=12):
    global GLOBAL_NLTK_WORD_BANK
    print("⏳ Initializing NLTK word bank from WordNet...")
    if not download_nltk_resources():
        print("🔴 NLTK resources unavailable. Word bank may be empty or fallback.")
        GLOBAL_NLTK_WORD_BANK = ["nexus", "vector", "zenith", "catalyst", "aura", "default", "words"]
        return
    try:
        lemmas = set()
        for pos_tag in ['n', 'a', 'r']:
            for synset in nltk.corpus.wordnet.all_synsets(pos=pos_tag):
                for lemma in synset.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    if ' ' not in word and word.isalpha() and min_len <= len(word) <= max_len:
                        lemmas.add(word)
        GLOBAL_NLTK_WORD_BANK = sorted(list(lemmas))
        if not GLOBAL_NLTK_WORD_BANK:
            print(f"⚠️ NLTK WordNet bank is empty after filtering. Using fallback.")
            GLOBAL_NLTK_WORD_BANK = ["creative", "engine", "spark", "idea", "inspiration"]
        else:
            print(f"✅ NLTK WordNet bank initialized with {len(GLOBAL_NLTK_WORD_BANK)} unique words.")
    except Exception as e:
        print(f"🔴 An unexpected error occurred while loading NLTK word bank: {e}")
        GLOBAL_NLTK_WORD_BANK = ["error", "loading", "bank", "fallback", "words"]


def get_random_nltk_inspiration_words(count=3):
    if not GLOBAL_NLTK_WORD_BANK: return ["placeholder", "words", "default"]
    actual_count = min(count, len(GLOBAL_NLTK_WORD_BANK))
    return random.sample(GLOBAL_NLTK_WORD_BANK, actual_count)


# --- "Craziness" Helper ---
def get_craziness_settings(level: int) -> tuple[str, float, float, str]:
    norm_level = (level - 1) / 99.0
    temp = 0.4 + norm_level * 1.0
    top_p = 0.95 - norm_level * 0.30
    if 1 <= level <= 15:
        desc = "extremely conventional, traditional, safe, and very professional-sounding"
        insp_intro = "Subtly draw upon these common concepts if directly relevant:"
    elif 16 <= level <= 30:
        desc = "conventional, established, clear, and professional-sounding, with a hint of creativity"
        insp_intro = "Consider these established concepts for gentle inspiration:"
    elif 31 <= level <= 45:
        desc = "moderately creative, memorable, and broadly appealing, balancing tradition with novelty"
        insp_intro = "Draw moderate creative inspiration from these concepts:"
    elif 46 <= level <= 60:
        desc = "creative, imaginative, and unique, aiming for marketability with a distinct flair"
        insp_intro = "Creatively weave in elements inspired by these diverse concepts:"
    elif 61 <= level <= 75:
        desc = "highly imaginative, unconventional, and boldly unique, pushing some stylistic boundaries"
        insp_intro = "Boldly incorporate or twist ideas inspired by these eclectic concepts:"
    elif 76 <= level <= 90:
        desc = "wildly creative, abstract, avant-garde, and very unconventional, even bizarre but catchy"
        insp_intro = "Daringly fuse or draw wild and unexpected inspiration from these abstract concepts:"
    else:
        desc = "extremely experimental, surreal, boundary-shattering, provocative, and potentially nonsensical but highly memorable"
        insp_intro = "Explode conventions! Conjure surreal names from the ether, perhaps bizarrely inspired by:"
    return desc, round(temp, 2), round(top_p, 2), insp_intro


# --- Helper Functions ---
def generate_names(theme: str, count: int, craziness_level: int, name_type_instruction: str, name_type_label: str) -> \
list[str]:
    craziness_desc, temp, top_p, insp_intro_style = get_craziness_settings(craziness_level)
    generator_options = {"temperature": temp, "top_p": top_p, "repeat_penalty": 1.15}
    num_inspirations = random.randint(2, 4)
    random_inspiration_words = get_random_nltk_inspiration_words(num_inspirations)
    inspiration_clause = ""
    if random_inspiration_words:
        inspiration_clause = f"{insp_intro_style} {', '.join(random_inspiration_words)}."

    prompt = (
        f"You are a naming expert with a flair for the {''.join(craziness_desc.split(',')[-1:])}. "
        f"Your task is to generate names for the theme: '{theme}'.\n"
        f"The overall desired style is: '{craziness_desc}'.\n"
        f"**Specifically for THIS batch, {name_type_instruction}**\n"
        f"{inspiration_clause}\n"
        f"Suggest {count} names. Provide ONLY the names, one name per line. No numbering, no markdown, no extra text. Just plain text names."
    )
    print(
        f"\n🤖 Asking {GENERATOR_MODEL} to generate {count} {name_type_label} for theme: '{theme}' (Craziness: {craziness_level}/100)")
    print(f"   Style Hint: '{craziness_desc}', Temp: {temp}, Top_p: {top_p}")
    if random_inspiration_words: print(
        f"   NLTK Inspiration Seeds ({insp_intro_style.split(' ')[0]}): {', '.join(random_inspiration_words)}")

    try:
        response = ollama.chat(model=GENERATOR_MODEL, messages=[{'role': 'user', 'content': prompt}],
                               options=generator_options)
        content = response['message']['content'].strip()
        names = [name.strip().lstrip('*- ').rstrip('.,') for name in content.split('\n') if
                 name.strip() and 1 < len(name.strip().lstrip('*- ').rstrip('.,')) < 45]
        if not names:
            print(
                f"⚠️ Generator ({GENERATOR_MODEL}) returned no usable names for {name_type_label} or unexpected format. Raw: {content[:200]}...")
            potential_names = re.findall(r"^\s*([a-zA-Z0-9]+(?:[ \-'][a-zA-Z0-9]+)*)\s*$", content, re.MULTILINE)
            names = [name.strip() for name in potential_names if 1 < len(name.strip()) < 45][:count]
        if not names:
            print(
                f"❌ Error: Generator ({GENERATOR_MODEL}) did not return usable names for {name_type_label}. Response: {content}")
            return []
        print(f"✅ Got {len(names)} initial {name_type_label}.")
        return list(dict.fromkeys(names))[:count]
    except Exception as e:
        print(f"❌ Error communicating with Ollama (Generator: {GENERATOR_MODEL}) for {name_type_label}: {e}")
        return []


def check_name_online_google_api(name: str, theme: str) -> tuple[bool, str]:
    # --- WEB CHECKING TEMPORARILY DISABLED FOR REFINEMENT ---
    # To re-enable, uncomment the original function body below and comment out/delete the next 2 lines.
    # print(f"    (Web check for '{name}' would occur here if Google API calls were enabled)")
    return False, "Web check disabled (manually commented out in script)."
    # --- END OF TEMPORARY DISABLE ---

    """
    # global GOOGLE_API_CONFIGURED # Original function body below
    # if not GOOGLE_API_CONFIGURED: return False, "Web check skipped (Google API not configured)."
    # print(f"    🔎 Web checking (Google API): '{name}'...")
    # search_query = f"\"{name}\" company OR business OR official site"
    # try:
    #     service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    #     res = service.cse().list(q=search_query, cx=GOOGLE_CSE_ID, num=3).execute()
    #     items = res.get('items', [])
    #     if not items: print(f"       No direct results for '{search_query}'.")
    #     else:
    #         print(f"       Found {len(items)} results for '{search_query}'. Analyzing top...")
    #         name_lower = name.lower()
    #         indicators = ["linkedin.com/company", "facebook.com/", "instagram.com/", "crunchbase.com/organization", "bloomberg.com/profile/company"]
    #         for item in items:
    #             title, link = item.get('title', '').lower(), item.get('link', '').lower()
    #             if any(ind in link for ind in indicators): return True, f"Potential entity (platform): {item.get('link','')}"
    #             if name_lower in title and ("company" in title or "inc" in title or "llc" in title or name_lower == title.split('|')[0].strip() or name_lower == title.split('-')[0].strip()):
    #                 return True, f"Title suggests existing company: '{item.get('title', '')}' at {item.get('link', '')}"
    #     commons = ["the", "a", "is", "in", "of", "for", "tech", "solutions", "group", "global", "app", "data", "cloud", "digital", "systems", "labs"]
    #     parts = name.lower().split()
    #     if len(parts) == 1 and parts[0] in commons and len(parts[0]) > 2: return True, f"Name '{name}' is common/unbrandable."
    #     return False, "Name seems relatively clear (top Google API results)."
    # except HttpError as e:
    #     err_details = json.loads(e.content).get('error', {})
    #     msg, reason = err_details.get('message', str(e)), err_details.get('reason', '')
    #     if e.resp.status in [429, 403] or 'quotaExceeded' in reason or 'billingNotEnabled' in reason:
    #         print(f"    ⚠️ Google API Quota/Billing Issue for '{name}': {msg}. Disabling Google checks for this session.")
    #         GOOGLE_API_CONFIGURED = False # Disable for this session
    #         return False, f"Web check skipped due to API issue: {msg}"
    #     print(f"    ⚠️ Google API HttpError for '{name}': {msg}")
    #     return False, f"Google API HttpError: {msg}"
    # except Exception as e:
    #     print(f"    ⚠️ Unexpected error during Google API web check for '{name}': {e}")
    #     return False, f"Unexpected Google API error: {e}"
    """


def critique_name_with_personality(name_to_critique: str, theme: str, critic_profile: dict) -> tuple[
    str | None, int | None]:
    system_prompt, user_prompt_example = critic_profile[
        "system_prompt"], "{\"critique\": \"This name feels a bit outdated.\", \"score\": 4}"
    user_prompt = (
        f"Theme: '{theme}'. Name to critique: '{name_to_critique}'.\n\nProvide brief critique (1-3 sentences) & score (1-10). STRICT JSON: {user_prompt_example}")
    model_options = {"temperature": critic_profile.get("temperature", 0.8)}
    try:
        response = ollama.chat(model=critic_profile["model"], messages=[{'role': 'system', 'content': system_prompt},
                                                                        {'role': 'user', 'content': user_prompt}],
                               options=model_options, format="json")
        content = response['message']['content'].strip()
        json_str = content
        if not (content.startswith("{") and content.endswith("}")):
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL) or re.search(r'(\{[\s\S]*?\})',
                                                                                                  content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise json.JSONDecodeError("No JSON in response", content, 0)
        data = json.loads(json_str)
        critique, score_val = data.get('critique'), data.get('score')
        score = int(score_val) if score_val is not None and isinstance(score_val, (int, float, str)) and str(
            score_val).isdigit() and 1 <= int(score_val) <= 10 else None
        return critique, score
    except (json.JSONDecodeError, KeyError) as je:
        print(
            f"⚠️ Critic ({critic_profile['name']}) for '{name_to_critique}' JSON error. Error: {je}. Raw: {content[:150]}...")
        s_match = re.search(r'\b(?:score|rating)[\s:]*(\b(?:[1-9]|10)\b)', content, re.IGNORECASE)
        s_fallback = int(s_match.group(1)) if s_match else None
        c_fallback = re.sub(r'\b(?:score|rating)[\s:]*(\b(?:[1-9]|10)\b)', '', content, flags=re.IGNORECASE).strip(
            '{}[]().,: "\'') if content else "Critique parsing failed."
        return c_fallback, s_fallback
    except Exception as e:
        print(f"❌ Error with Ollama ({critic_profile['model']}/{critic_profile['name']}) for '{name_to_critique}': {e}")
        return "Error during critique.", None


def run_synthesis_critique(name_to_critique: str, theme: str, previous_critiques: list, profile: dict) -> tuple[
    str | None, str | None]:
    system_prompt = profile["system_prompt"]
    critiques_text = "\n".join(
        [f"- {c['critic_name']} (Score: {c.get('score', 'N/A')}): \"{c.get('critique', 'N/A')}\"" for c in
         previous_critiques])
    user_prompt_example = "{\"summary\": \"Catchy but concerns valid.\", \"verdict\": \"Proceed with Caution\"}"
    user_prompt = (f"Theme: '{theme}'. Name: '{name_to_critique}'.\nCritiques:\n{critiques_text}\n\n"
                   f"Your synthesis. STRICT JSON (summary, verdict): {user_prompt_example}")
    model_options = {"temperature": profile.get("temperature", 0.5)}
    summary, verdict = None, None
    try:
        response = ollama.chat(model=profile["model"], messages=[{'role': 'system', 'content': system_prompt},
                                                                 {'role': 'user', 'content': user_prompt}],
                               options=model_options, format="json")
        content = response['message']['content'].strip()
        json_str = content
        if not (content.startswith("{") and content.endswith("}")):
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL) or re.search(r'(\{[\s\S]*?\})',
                                                                                                  content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise json.JSONDecodeError("No JSON in synthesis response", content, 0)
        data = json.loads(json_str)
        summary, verdict = data.get('summary'), data.get('verdict')
        return summary, verdict
    except (json.JSONDecodeError, KeyError) as je:
        print(
            f"⚠️ Synthesis Critic ({profile['name']}) for '{name_to_critique}' JSON error. Error: {je}. Raw: {content[:150]}...")
        v_match = re.search(
            r'verdict["\']?\s*:\s*("?)([\w\s]+(?:Recommended|Caution|Promising|Concerns|Advised|Suitable|Unsuitable|Viable))\1?',
            content, re.IGNORECASE)
        verdict_fallback = v_match.group(2).strip() if v_match else "Verdict unclear"
        return content or "Synthesis parsing failed.", verdict_fallback
    except Exception as e:
        print(f"❌ Error with Ollama ({profile['model']} for Synthesis) for '{name_to_critique}': {e}")
        return "Error during synthesis.", "Error"


def run_global_ranking(theme: str, synthesized_names_data: list, profile: dict) -> list:
    if not synthesized_names_data: return []
    input_for_ranker = ["- Name: \"{}\"\n  Synthesized Summary: \"{}\"\n  Synthesized Verdict: \"{}\"".format(
        item['name'], item.get('synthesis_summary', 'N/A'), item.get('synthesis_verdict', 'N/A')
    ) for item in synthesized_names_data]
    names_list_str = "\n\n".join(input_for_ranker)
    user_prompt_example = "{\"ranked_finalists\": [{\"rank\": 1, \"name\": \"NovaCore\", \"justification\": \"Excellent blend...\"}, ...]}"  # Shortened for brevity
    user_prompt = (
        f"Theme: '{theme}'.\n\nCandidate names and synthesized evaluations from Dr. Synthia Verdict:\n{names_list_str}\n\n"
        f"Your task: Generate the final ranked list of these {len(synthesized_names_data)} names (best to worst). "
        f"Provide concise justification ONLY for top 3. Output MUST be JSON as specified (key 'ranked_finalists'). Example:\n{user_prompt_example}")
    model_options = {"temperature": profile.get("temperature", 0.3)}
    print(f"\n{profile['emoji']} Asking {profile['name']} for global ranking of {len(synthesized_names_data)} names...")
    try:
        response = ollama.chat(model=profile["model"],
                               messages=[{'role': 'system', 'content': profile["system_prompt"]},
                                         {'role': 'user', 'content': user_prompt}], options=model_options,
                               format="json")
        content = response['message']['content'].strip()
        json_str = content
        if not (content.startswith("{") and content.endswith("}")):
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL) or re.search(r'(\{[\s\S]*?\})',
                                                                                                  content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise json.JSONDecodeError("No JSON in ranking response", content, 0)
        data = json.loads(json_str)
        ranked_list = data.get('ranked_finalists', [])
        if not isinstance(ranked_list, list) or not all(
                isinstance(item, dict) and 'name' in item and 'rank' in item for item in ranked_list):
            print(f"⚠️ Global Ranker data format error: {ranked_list}. Attempting recovery...")
            extracted = [item.get('name') for item in ranked_list if isinstance(item, dict) and item.get('name')]
            if extracted: return [{"rank": i + 1, "name": name, "justification": ""} for i, name in
                                  enumerate(extracted)]
            return []
        print(f"✅ Global ranking received for {len(ranked_list)} names.")
        return ranked_list
    except (json.JSONDecodeError, KeyError) as je:
        print(f"⚠️ Global Ranker ({profile['name']}) JSON error. Error: {je}. Raw: {content[:300]}...")
        return []
    except Exception as e:
        print(f"❌ Error with Ollama ({profile['model']} for Global Ranker): {e}")
        return []


def sanitize_for_html_id(text: str) -> str:
    text = re.sub(r'[^\w\s-]', '', text.lower()).strip()
    text = re.sub(r'[-\s]+', '-', text)
    return text if text else "unnamed-section"


def generate_html_report(theme: str, results: list, filename="name_critique_report.html",
                         global_ranking_data: list | None = None):
    print(f"\n📄 Generating HTML report: {filename}...")
    # --- MODIFIED WEB NOTE ---
    web_note = "Web check via Google API MANUALLY COMMENTED OUT in script for refinement."
    if GOOGLE_API_CLIENT_AVAILABLE and GOOGLE_API_CONFIGURED:  # If user uncomments the Google search...
        # This part would only be relevant if the Google API check wasn't manually bypassed in check_name_online_google_api
        # For now, the manual comment out message takes precedence.
        # web_note = "Web check using Google Custom Search API (but calls are currently bypassed in script)."
        pass

    ranked_list_html = "<details open class='ranked-list-container'><summary class='ranked-list-summary'>🏆 Final Name Ranking (by The Boardroom Oracle)</summary><ol class='ranked-list'>"
    if not global_ranking_data:
        ranked_list_html += "<li>Global ranking was not available or failed. Names below are sorted by internal processing order. Dr. Synthia's per-name synthesis shown if available:</li>"
        for i, item in enumerate(results):  # results list is already sorted by CRO if CRO succeeded
            name = item["name"]
            sanitized_name_id = sanitize_for_html_id(name)
            synthia_verdict = item.get("synthesis", {}).get("verdict", "N/A")
            ranked_list_html += f"<li><a href='#{sanitized_name_id}'>{i + 1}. {name}</a> - Dr. Synthia's Verdict: {synthia_verdict}</li>"
    else:
        for ranked_item in global_ranking_data:
            name, rank = ranked_item["name"], ranked_item["rank"]
            justification = ranked_item.get("justification") or ""  # Ensure empty string if null/None
            sanitized_name_id = sanitize_for_html_id(name)
            justification_html = f"<span class='rank-justification'> - {justification}</span>" if justification else ""
            ranked_list_html += f"<li><a href='#{sanitized_name_id}'>{rank}. {name}</a>{justification_html}</li>"
    ranked_list_html += "</ol></details>"

    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Name Generation & Critique Report: {theme}</title><style>
        body {{ scroll-behavior: smooth; font-family: Segoe UI, sans-serif; margin:0; padding:20px; background-color:#f0f2f5; color:#333; line-height:1.6; }}
        .container {{ max-width:1000px; margin:20px auto; padding:25px; background-color:#fff; box-shadow:0 6px 20px rgba(0,0,0,0.08); border-radius:10px; }}
        h1 {{ color:#1a2533; text-align:center; border-bottom:3px solid #4a90e2; padding-bottom:15px; margin-bottom:25px; font-size:2.2em; }}
        .ranked-list-container {{ margin-bottom:30px; border:1px solid #dce4ec; border-radius:8px; background-color:#f8f9fa; padding:15px 20px; }}
        .ranked-list-summary {{ font-size:1.5em; font-weight:600; color:#2c5282; cursor:pointer; margin-bottom:10px; }}
        .ranked-list {{ list-style-type:decimal; padding-left:25px; }} .ranked-list li {{ margin-bottom:8px; font-size:1.1em; }}
        .ranked-list li a {{ text-decoration:none; color:#3182ce; font-weight:500; }} .ranked-list li a:hover {{ text-decoration:underline; color:#2b6cb0; }}
        .rank-justification {{ font-style: italic; color: #555; font-size: 0.9em; }}
        .name-block {{ margin-bottom:40px; padding:25px; background-color:#fcfdff; border:1px solid #dce4ec; border-radius:8px; box-shadow:0 3px 10px rgba(0,0,0,0.04);}}
        .name-title {{ font-size:2.4em; color:#2c5282; margin-bottom:25px; text-align:center; font-weight:600; }}
        .critic-review-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:20px; margin-bottom:25px; }}
        .critic-review {{ border:1px solid #cbd5e0; border-left-width:6px; padding:18px; border-radius:6px; background-color:#fff; box-shadow:0 2px 5px rgba(0,0,0,0.03);}}
        .critic-header {{ display:flex; align-items:center; margin-bottom:10px;}} .critic-emoji {{ font-size:2em; margin-right:12px; }}
        .critic-name {{ font-weight:bold; font-size:1.25em; color:#4a5568; }}
        .critique-text {{ margin:10px 0; font-style:italic; color:#52525b; white-space:pre-wrap; }}
        .score {{ font-weight:bold; font-size:1.1em; }} .score-good {{color:#38a169;}} .score-medium {{color:#dd6b20;}} .score-bad {{color:#e53e3e;}}
        .synthesis-critique {{ margin-top:30px; padding:20px; background-color:#e6f0ff; border:1px solid #a3c6ff; border-left:6px solid #2b6cb0; border-radius:6px; }}
        .synthesis-header {{ font-size:1.4em; font-weight:bold; color:#2c5282; margin-bottom:10px; display:flex; align-items:center;}}
        .synthesis-summary, .synthesis-verdict {{ margin-bottom:8px; }}
        .synthesis-verdict strong {{ color:#2a4365; }}
        .meta-info {{ text-align:center; font-size:0.95em; color:#555; margin-bottom:10px; }}
        .web-check-note {{ font-size:0.9em; color:#e53e3e; font-weight:bold; text-align:center; margin-bottom:20px; padding:8px; background-color:#fff0f0; border:1px solid #ffc0c0; border-radius:4px;}}
        .footer {{ text-align:center; margin-top:40px; padding-top:20px; border-top:1px solid #e0e0e0; font-size:0.9em; color:#777; }}
    </style></head><body><div class="container">
        <h1>Name Generation & Critique Report</h1>
        <div class="meta-info"><strong>Theme:</strong> {theme}<br>Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <p class='web-check-note'>{web_note}</p>{ranked_list_html}"""
    if not results: html_content += "<p style='text-align:center;'>No names were approved and critiqued.</p>"
    for item in results:  # results is already sorted by CRO's ranking if successful
        name, s_id = item["name"], sanitize_for_html_id(item["name"])
        critiques, synthesis_data = item["critiques"], item.get("synthesis")
        html_content += f"<div class='name-block' id='{s_id}'><h2 class='name-title'>{name}</h2><div class='critic-review-grid'>"
        for c_res in critiques:
            c_text_esc = (c_res["critique"] or "N/A").replace('<', '&lt;').replace('>', '&gt;')
            s_display, s_class = ("N/A", "")
            if c_res["score"] is not None:
                s_display = str(c_res["score"])
                if c_res["score"] >= 8:
                    s_class = "score-good"
                elif c_res["score"] >= 5:
                    s_class = "score-medium"
                else:
                    s_class = "score-bad"
            b_color_map = {"Finch": "#c0392b", "Hayes": "#27ae60", "Zip": "#f39c12", "Moon": "#8e44ad"}
            b_color = next((b_color_map[cn_part] for cn_part in b_color_map if cn_part in c_res["critic_name"]),
                           "#3498db")
            html_content += f"""<div class="critic-review" style="border-left-color:{b_color};">
                <div class="critic-header"><span class="critic-emoji">{c_res["emoji"]}</span><span class="critic-name">{c_res["critic_name"]}:</span></div>
                <p class="critique-text">"{c_text_esc}"</p><p class="score">Score: <span class="{s_class}">{s_display}</span>/10</p></div>"""
        html_content += "</div>"
        if synthesis_data:  # This is Dr. Synthia's per-name synthesis
            sum_esc = (synthesis_data.get("summary", "N/A")).replace('<', '&lt;').replace('>', '&gt;')
            ver_esc = (synthesis_data.get("verdict", "N/A")).replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"""<div class="synthesis-critique"> 
                <div class="synthesis-header"><span class="critic-emoji">{synthesis_data["emoji"]}</span>{synthesis_data["critic_name"]}: (Per-Name Synthesis)</div>
                <p class="synthesis-summary"><strong>Summary:</strong> {sum_esc}</p>
                <p class="synthesis-verdict"><strong>Verdict:</strong> {ver_esc}</p>
                </div>"""
        html_content += "</div>"
    html_content += f"<div class='footer'>Models: Gen - {GENERATOR_MODEL}, Critics - {CRITIC_BASE_MODEL}, Synth/Ranker - {SYNTHESIS_CRITIC_MODEL}.</div></div></body></html>"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ HTML report: {filename}")
    except IOError as e:
        print(f"❌ Error writing HTML file: {e}")


# --- Main Application Logic ---
def main():
    global GOOGLE_API_CONFIGURED  # Keep this for potential future re-enabling
    print("⚙️ Initializing NLTK resources (one-time setup if needed)...")
    load_nltk_word_bank()
    print("\n✨ Welcome to the Multi-Personality Name Idea Generator & Critic! ✨")
    print(
        f"Models: Gen - {GENERATOR_MODEL}, Critic Base - {CRITIC_BASE_MODEL}, Synth/Ranker - {SYNTHESIS_CRITIC_MODEL}")

    # Inform user about Google API status (even if bypassed)
    if GOOGLE_API_CLIENT_AVAILABLE and GOOGLE_API_CONFIGURED:
        print("✅ Google Custom Search API seems configured (API Key and CSE ID found).")
    else:
        print("🔴 Google Custom Search API is NOT configured/available.")
    print(
        "🔴🔴🔴 NOTE: Google Web Checking is CURRENTLY COMMENTED OUT in the script to save API costs during refinement. 🔴🔴🔴")

    print(f"Standard Critics: {len(CRITIC_PERSONALITIES)} (+1 Per-Name Synthesizer, +1 Global Ranker)")
    print("-" * 70)

    theme = input("➡️ Enter the theme for the names: ")
    if not theme.strip(): print("⚠️ No theme. Exiting."); return

    while True:
        try:
            craziness_input = input("➡️ Enter 'craziness' level for name generation (1=tame, 100=wild): ")
            craziness_level = int(craziness_input)
            if 1 <= craziness_level <= 100:
                break
            else:
                print("   Please enter a number between 1 and 100.")
        except ValueError:
            print("   Invalid input. Please enter a number.")

    name_generation_tasks = [
        {
            "instruction": "generate only single, impactful one-word brand names. Ensure each suggestion is strictly a single word.",
            "count": 20, "type_label": "One-Word Names"},
        {
            "instruction": "generate creative one-word portmanteaus by blending two or more words (or their sounds/meanings) into a new single word (e.g., 'smog' from smoke/fog, 'brunch' from breakfast/lunch). Each suggestion must be a single, blended word related to the theme.",
            "count": 20, "type_label": "Portmanteaus"},
        {
            "instruction": "generate compelling two-word brand names. Each name should consist of exactly two distinct words. Focus on evocative or descriptive combinations.",
            "count": 20, "type_label": "Two-Word Names"}
    ]
    initial_generated_names = []
    for task in name_generation_tasks:
        names_batch = generate_names(theme, task['count'], craziness_level, task['instruction'], task['type_label'])
        if names_batch: initial_generated_names.extend(names_batch)
    initial_generated_names = list(dict.fromkeys(initial_generated_names))

    if not initial_generated_names: print(f"😥 No names generated for '{theme}'. Try again or check Ollama."); return

    print(f"\n--- Selecting names for critique (Web Checking is MANUALLY COMMENTED OUT) ---")
    approved_names = []
    # This loop now effectively "approves" all names up to MAX_NAMES_TO_CRITIQUE
    # because check_name_online_google_api always returns (False, "disabled message")
    for i, name_to_check in enumerate(initial_generated_names):
        if len(approved_names) >= MAX_NAMES_TO_CRITIQUE:
            print(f"\n🏁 Reached target of {MAX_NAMES_TO_CRITIQUE} names for critique list.")
            break
        is_problematic, reason = check_name_online_google_api(name_to_check, theme)  # Will return False
        if is_problematic:  # This path will not be taken due to commented out check
            print(f"    ⛔ Rejected: '{name_to_check}'. Reason: {reason}")
        else:
            print(
                f"    ☑️ Adding to critique list (web check bypassed): '{name_to_check}'. Reason from checker: {reason}")
            approved_names.append(name_to_check)
        # API_REQUEST_DELAY is not needed here as the actual call is bypassed

    if not approved_names: print(f"\n😥 No names available for critique. Unable to proceed."); return

    final_names_for_critique = list(dict.fromkeys(approved_names))[:MAX_NAMES_TO_CRITIQUE]
    print(f"\n--- Critiquing {len(final_names_for_critique)} names ---")

    all_results_for_report = []
    data_for_global_ranker = []

    for i, name in enumerate(final_names_for_critique):
        print(f"\n🔄 Critiquing Name {i + 1}/{len(final_names_for_critique)}: '{name}'")
        current_name_critiques_for_synthesis, name_report_entry = [], {"name": name, "critiques": []}
        for critic_profile in CRITIC_PERSONALITIES:
            print(f"   ↳ Asking {critic_profile['emoji']} {critic_profile['name']} ...")
            critique, score = critique_name_with_personality(name, theme, critic_profile)
            critique_entry = {"critic_name": critic_profile["name"], "emoji": critic_profile["emoji"],
                              "critique": critique, "score": score}
            name_report_entry["critiques"].append(critique_entry)
            current_name_critiques_for_synthesis.append(critique_entry)
            crit_snip = (critique[:77] + "...") if critique and len(critique) > 80 else (critique or "N/A")
            print(f"     🗣️ Critique: {crit_snip} Score: {score if score is not None else 'N/A'}")

        if current_name_critiques_for_synthesis:
            print(
                f"   ↳ Asking {SYNTHESIS_CRITIC_PROFILE['emoji']} {SYNTHESIS_CRITIC_PROFILE['name']} for per-name synthesis...")
            summary, verdict = run_synthesis_critique(name, theme, current_name_critiques_for_synthesis,
                                                      SYNTHESIS_CRITIC_PROFILE)
            name_report_entry["synthesis"] = {"critic_name": SYNTHESIS_CRITIC_PROFILE["name"],
                                              "emoji": SYNTHESIS_CRITIC_PROFILE["emoji"], "summary": summary,
                                              "verdict": verdict}
            data_for_global_ranker.append({"name": name, "synthesis_summary": summary, "synthesis_verdict": verdict})
            sum_snip = (summary[:77] + "...") if summary and len(summary) > 80 else (summary or "N/A")
            print(f"     🧐 Per-Name Summary: {sum_snip} Verdict: {verdict or 'N/A'}")
        all_results_for_report.append(name_report_entry)

    final_ranked_order_from_cro = []
    if data_for_global_ranker:
        final_ranked_order_from_cro = run_global_ranking(theme, data_for_global_ranker, CHIEF_RANKING_OFFICER_PROFILE)

    if final_ranked_order_from_cro:
        name_to_global_rank = {item['name'].lower(): item['rank'] for item in
                               final_ranked_order_from_cro}  # Use lower for robustness
        all_results_for_report.sort(key=lambda x: name_to_global_rank.get(x["name"].lower(), float('inf')))
        print(f"\n✅ All names processed and re-sorted by Global Ranker's output.")
    else:
        print("\n⚠️ Global Ranker did not return a valid ranking. Report will use original processing order.")

    if all_results_for_report:
        safe_theme = re.sub(r'[^\w\s-]', '', theme.lower()).strip()
        safe_theme = re.sub(r'[-\s]+', '_', safe_theme) if safe_theme else "report"
        report_filename = f"name_report_{safe_theme}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        generate_html_report(theme, all_results_for_report, filename=report_filename,
                             global_ranking_data=final_ranked_order_from_cro)
    else:
        print("\nNo results to generate a report for.")
    print("\n👋 All done!")


if __name__ == "__main__":
    main()