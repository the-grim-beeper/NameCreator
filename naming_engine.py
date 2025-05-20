import ollama
import json
import datetime
import re
import os
import time

# --- NLTK Imports ---
import nltk
import random

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
GENERATOR_MODEL = 'gemma3'  # Or your preferred name generator model (e.g., gemma:27b if you prefer larger)
CRITIC_BASE_MODEL = 'llama3.2'  # Base model for critics, e.g., 'mistral', 'llama3'
SYNTHESIS_CRITIC_MODEL = 'mistral'  # Model for the synthesis critic (can be same or different, e.g. llama3:70b for more complex reasoning)
# If using Gemma 3 for critics/synthesis, ensure it's pulled in Ollama.

NUMBER_OF_NAMES_TO_GENERATE = 60  # Consider reducing if NLTK + Synthesis makes it too long
MAX_NAMES_TO_CRITIQUE = 15  # Adjusted for potentially longer processing per name
API_REQUEST_DELAY = 1.0

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
        print("   Web checking via Google API will be disabled.")
else:
    print("   Web checking via Google API is disabled (client library not found).")

# --- NLTK Global Word Bank ---
GLOBAL_NLTK_WORD_BANK = []

# --- Critic Personalities (Standard Critics) ---
CRITIC_PERSONALITIES = [
    {
        "name": "Mr. Alistair Finch (CEO)",
        "emoji": "👴",
        "system_prompt": (
            "You are Alistair Finch, a pragmatic and experienced CEO in his late 50s. "
            "You value names that sound professional, trustworthy, marketable, and have a strong, clear presence. "
            "Avoid overly trendy, frivolous, or difficult-to-pronounce names. Focus on longevity and brand potential."
        ),
        "model": CRITIC_BASE_MODEL, "temperature": 0.7  # Slightly lower temp for more focused CEO
    },
    {
        "name": "Dr. Evelyn Hayes (Social Advocate)",
        "emoji": "⚖️",  # Changed emoji for distinctness
        "system_prompt": (
            "You are Dr. Evelyn Hayes, a progressive, left-leaning academic and social advocate in her 40s. "
            "You appreciate names that are inclusive, thoughtful, unique, and perhaps hint at ethical or sustainable values. "
            "You are critical of names that might seem exploitative, generic, overly corporate, or culturally insensitive. "
            "You value creativity and meaning."
        ),
        "model": CRITIC_BASE_MODEL, "temperature": 1.0
    },
    {
        "name": "Zip (The Cool Kid)",
        "emoji": "😎",
        "system_prompt": (
            "You are Zip, a 17-year-old who is always ahead of the trends (or thinks they are!). "
            "You like names that are catchy, fresh, a bit edgy, short, and 'shareable' online. "
            "You dislike anything that sounds old-fashioned, boring, too formal, or trying too hard to be cool if it's not genuine. "
            "Think viral potential and modern appeal."
        ),
        "model": CRITIC_BASE_MODEL, "temperature": 1.2  # Higher temp for more unpredictable Zip
    },
    {
        "name": "Seraphina \"Sparkle\" Moon (The Name Champion)",
        "emoji": "✨",
        "system_prompt": (
            "You are Seraphina \"Sparkle\" Moon, an incredibly enthusiastic and creative Name Champion! Your energy is infectious, and you always see the brightest potential in every idea. "
            "For the given name and theme, find every reason to love it. Highlight its unique charm, its imaginative possibilities, and the positive feelings it evokes. "
            "Be warm, generous, and uplifting in your critique. Even if there are perceived challenges with the name, try to frame them as unique characteristics or opportunities for creative brilliance! "
            "Provide a score from 1 to 10, but you're naturally inclined to see the good, so lean generously towards higher scores if you can creatively justify its potential or positive aspects."
        ),
        "model": CRITIC_BASE_MODEL, "temperature": 1.1
    }
]

# --- Synthesis Critic Profile ---
SYNTHESIS_CRITIC_PROFILE = {
    "name": "Dr. Synthia Verdict (Lead Strategist)",
    "emoji": "🧐",
    "system_prompt": (
        "You are Dr. Synthia Verdict, a highly analytical Lead Strategist. Your role is to synthesize feedback from a diverse panel of critics "
        "regarding a proposed name for a given theme. You will be provided with the theme, the name, and a list of critiques from other analysts. "
        "Your task is to: \n"
        "1. Briefly summarize the key convergent and divergent points from the provided critiques. \n"
        "2. Evaluate the name's overall suitability for the theme, considering the summarized feedback. \n"
        "3. Provide a clear, final verdict (e.g., 'Strongly Recommended', 'Recommended with Considerations', 'Proceed with Caution', 'Not Recommended'). \n"
        "Focus on actionable insights and a decisive conclusion."
    ),
    "model": SYNTHESIS_CRITIC_MODEL,
    "temperature": 0.5  # Lower temperature for analytical synthesis
}


# --- NLTK Functions ---
def download_nltk_resources():
    """
    Downloads NLTK WordNet and OMW-1.4 if not already present.
    More robustly handles download and verification.
    """
    resources_to_download = {
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'  # Open Multilingual Wordnet, often used with WordNet
    }
    all_successful = True  # Track if all resources are successfully available

    for resource_name, resource_path_suffix in resources_to_download.items():
        try:
            # Check if the resource is already available
            nltk.data.find(resource_path_suffix)
            print(f"✅ NLTK resource '{resource_name}' ({resource_path_suffix}) already available.")
        except LookupError:
            # This is the correct exception when a resource is not found by nltk.data.find()
            print(f"⏳ NLTK resource '{resource_name}' ({resource_path_suffix}) not found. Attempting download...")
            try:
                # Attempt to download the resource using its short name (e.g., 'wordnet')
                # nltk.download() can be a bit inconsistent with return values for success,
                # so we rely on a subsequent find operation to confirm.
                nltk.download(resource_name, quiet=False)  # Set quiet=False for more visibility during download

                # After attempting download, try to find it again to verify
                print(f"   Verifying '{resource_name}' after download attempt...")
                nltk.data.find(resource_path_suffix)
                print(f"✅ Successfully downloaded and verified NLTK resource '{resource_name}'.")
            except LookupError:
                # If it's still not found after download attempt
                print(f"❌ Failed to find NLTK resource '{resource_name}' even after download attempt.")
                print(f"   Please try running the following in a Python interpreter manually:")
                print(f"   >>> import nltk")
                print(f"   >>> nltk.download('{resource_name}')")
                all_successful = False
            except Exception as e_download:
                # Catch any other error during the download process itself (e.g., network issues)
                print(f"❌ Failed to download NLTK resource '{resource_name}'. Error: {e_download}")
                all_successful = False
        except Exception as e_check:
            # Catch other potential errors during the initial find (should be rare if LookupError is main one)
            print(f"❌ Error checking NLTK resource '{resource_name}'. Error: {e_check}")
            all_successful = False

    return all_successful


def load_nltk_word_bank(min_len=4, max_len=12):
    global GLOBAL_NLTK_WORD_BANK
    print("⏳ Initializing NLTK word bank from WordNet...")
    if not download_nltk_resources():
        print("🔴 NLTK resources could not be verified or downloaded. Word bank may be empty or fallback.")
        GLOBAL_NLTK_WORD_BANK = ["nexus", "vector", "zenith", "catalyst", "aura"]  # Minimal fallback
        return

    try:
        lemmas = set()
        for pos_tag in ['n', 'a']:  # Nouns and Adjectives
            for synset in nltk.corpus.wordnet.all_synsets(pos=pos_tag):
                for lemma in synset.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    if ' ' not in word and word.isalpha() and min_len <= len(word) <= max_len:
                        lemmas.add(word)
        GLOBAL_NLTK_WORD_BANK = sorted(list(lemmas))
        if not GLOBAL_NLTK_WORD_BANK:
            print(f"⚠️ NLTK WordNet bank is empty after filtering. Using fallback.")
            GLOBAL_NLTK_WORD_BANK = ["default", "inspiration", "words", "creative", "engine"]
        else:
            print(f"✅ NLTK WordNet bank initialized with {len(GLOBAL_NLTK_WORD_BANK)} unique words.")
    except Exception as e:
        print(f"🔴 An unexpected error occurred while loading NLTK word bank: {e}")
        GLOBAL_NLTK_WORD_BANK = ["error", "occurred", "loading", "bank", "fallback"]


def get_random_nltk_inspiration_words(count=3):
    if not GLOBAL_NLTK_WORD_BANK:
        return ["placeholder", "words"]  # Should not happen if load_nltk_word_bank is called
    actual_count = min(count, len(GLOBAL_NLTK_WORD_BANK))
    return random.sample(GLOBAL_NLTK_WORD_BANK, actual_count)


# --- Helper Functions ---
def generate_names(theme: str, count: int) -> list[str]:
    generator_options = {"temperature": 0.9, "top_p": 0.9, "repeat_penalty": 1.1}
    num_inspirations = random.randint(2, 4)
    random_inspiration_words = get_random_nltk_inspiration_words(num_inspirations)
    inspiration_clause = ""
    if random_inspiration_words:
        inspiration_clause = f"Additionally, draw subtle thematic inspiration from some of these diverse concepts if appropriate: {', '.join(random_inspiration_words)}."

    prompt = (
        f"You are a supremely creative naming expert. Your task is to generate an exceptionally diverse and imaginative list of names. "
        f"The core theme is: '{theme}'. {inspiration_clause} "
        f"Aim for significant variety in style, length, and conceptual origin (e.g., abstract, descriptive, modern, classic, playful, evocative). "
        f"Suggest {count} names. Provide ONLY the names, one name per line. No numbering, no markdown, no extra text. Just plain text names."
    )
    print(
        f"\n🤖 Asking {GENERATOR_MODEL} (options: {generator_options}) to generate {count} names for theme: '{theme}'...")
    if random_inspiration_words: print(f"   NLTK Inspiration Seeds: {', '.join(random_inspiration_words)}")
    try:
        response = ollama.chat(model=GENERATOR_MODEL, messages=[{'role': 'user', 'content': prompt}],
                               options=generator_options)
        content = response['message']['content'].strip()
        names = [name.strip().lstrip('*- ').rstrip('.,') for name in content.split('\n') if
                 name.strip() and 1 < len(name.strip().lstrip('*- ').rstrip('.,')) < 35]  # Added length constraint
        if not names:
            print(
                f"⚠️ Generator model ({GENERATOR_MODEL}) returned no usable names or unexpected format. Raw: {content[:200]}...")
            # Fallback parsing attempt for names (more robust)
            potential_names = re.findall(r"^\s*([a-zA-Z0-9]+(?:[ \-'][a-zA-Z0-9]+)*)\s*$", content, re.MULTILINE)
            names = [name.strip() for name in potential_names if 1 < len(name.strip()) < 35][:count]
        if not names:
            print(
                f"❌ Error: Generator model ({GENERATOR_MODEL}) did not return any usable names even after fallback. Response: {content}")
            return []
        print(f"✅ Got {len(names)} initial names.")
        return list(dict.fromkeys(names))[:count]  # Remove duplicates, then slice
    except Exception as e:
        print(f"❌ Error communicating with Ollama (Generator: {GENERATOR_MODEL}): {e}")
        return []


def check_name_online_google_api(name: str, theme: str) -> tuple[bool, str]:
    global GOOGLE_API_CONFIGURED
    if not GOOGLE_API_CONFIGURED: return False, "Web check skipped (Google API not configured)."

    print(f"    🔎 Web checking name (Google API): '{name}'...")
    search_query = f"\"{name}\" company OR business OR official site"
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=search_query, cx=GOOGLE_CSE_ID, num=3).execute()  # Reduced num to 3 for speed/quota
        items = res.get('items', [])
        if not items:
            print(f"       No direct results for '{search_query}'.")
        else:
            print(f"       Found {len(items)} results for '{search_query}'. Analyzing top...")
            name_lower = name.lower()
            problematic_indicators = ["linkedin.com/company", "facebook.com/", "instagram.com/",
                                      "crunchbase.com/organization", "bloomberg.com/profile/company"]
            for item in items:
                title, link = item.get('title', '').lower(), item.get('link', '').lower()
                if any(indicator in link for indicator in
                       problematic_indicators): return True, f"Potential existing entity (platform/database): {link}"
                if name_lower in title and (
                        "company" in title or "inc." in title or "llc" in title or name_lower == title.split('|')[
                    0].strip() or name_lower == title.split('-')[0].strip()):
                    return True, f"Title strongly suggests existing company: '{item.get('title', '')}' at {item.get('link', '')}"

        # Simplified common words check (less aggressive)
        common_words_list = ["the", "a", "is", "in", "of", "for", "and", "tech", "solutions", "group", "global",
                             "world", "net", "corp", "io", "app", "data", "cloud", "digital", "systems", "labs", "co",
                             "inc", "biz", "info", "online", "services"]
        name_parts = name.lower().split()
        if len(name_parts) == 1 and name_parts[0] in common_words_list and len(
                name_parts[0]) > 2:  # only if it's a somewhat substantial common word
            return True, f"Name '{name}' is a very common, likely unbrandable single word."

        return False, "Name seems relatively clear (top Google API results)."
    except HttpError as e:
        error_details = json.loads(e.content).get('error', {})
        msg = error_details.get('message', str(e))
        reason = error_details.get('reason', '')
        if e.resp.status in [429, 403] or 'quotaExceeded' in reason or 'billingNotEnabled' in reason:
            print(
                f"    ⚠️ Google API Quota/Billing/Permission Issue for '{name}': {msg}. Disabling further Google API checks.")
            GOOGLE_API_CONFIGURED = False
            return False, f"Web check skipped due to API issue: {msg}"
        print(f"    ⚠️ Google API HttpError for '{name}': {msg}")
        return False, f"Google API HttpError: {msg}"
    except Exception as e:
        print(f"    ⚠️ Unexpected error during Google API web check for '{name}': {e}")
        return False, f"Unexpected Google API error: {e}"


def critique_name_with_personality(name_to_critique: str, theme: str, critic_profile: dict) -> tuple[
    str | None, int | None]:
    system_prompt = critic_profile["system_prompt"]
    user_prompt = (
        f"The overall theme is '{theme}'. The specific name to critique is: '{name_to_critique}'.\n\n"
        f"Based on your persona, provide a brief critique (1-3 sentences) and a score from 1 (poor) to 10 (excellent). "
        f"Format your response STRICTLY as a JSON object with two keys: 'critique' (string) and 'score' (integer). "
        f"Example: {{\"critique\": \"This name feels a bit outdated for the current market.\", \"score\": 4}}"
    )
    model_options = {"temperature": critic_profile.get("temperature", 0.8)}  # Use critic's temp

    try:
        response = ollama.chat(
            model=critic_profile["model"],
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            options=model_options,
            format="json"  # Request JSON format directly from Ollama if model supports it well
        )
        content = response['message']['content'].strip()
        # If format="json" is used, Ollama's response['message']['content'] should already be a JSON string
        # If the model doesn't strictly adhere, we might need regex to extract it from ```json ... ```

        json_str = content  # Assume content is the JSON string if format="json" is effective

        # Fallback if format="json" didn't yield pure JSON or if model wrapped it
        if not (content.startswith("{") and content.endswith("}")):
            match_json_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL)
            if match_json_block:
                json_str = match_json_block.group(1)
            else:  # Try to find any JSON-like structure
                match_loose = re.search(r'(\{[\s\S]*?\})', content, re.DOTALL)
                if match_loose:
                    json_str = match_loose.group(1)
                else:  # If no JSON found, raise error to be caught by fallback logic
                    raise json.JSONDecodeError("No JSON object found in response", content, 0)

        data = json.loads(json_str)
        critique = data.get('critique')
        score_val = data.get('score')
        score = None
        if score_val is not None:
            try:
                score = int(score_val)
                if not (1 <= score <= 10): score = None
            except ValueError:
                score = None
        return critique, score
    except (json.JSONDecodeError, KeyError) as je:  # Added KeyError for missing keys
        print(
            f"⚠️ Critic ({critic_profile['name']}) for '{name_to_critique}' didn't return valid JSON or expected keys. Error: {je}. Response: {content[:200]}...")
        # Fallback parsing for score and critique from raw text
        score_fallback_match = re.search(r'\b(?:score|rating)[\s:]*(\b(?:[1-9]|10)\b)', content, re.IGNORECASE)
        score_fallback = int(score_fallback_match.group(1)) if score_fallback_match else None
        critique_fallback = content  # Use full content as critique if JSON fails
        if score_fallback is not None:  # Remove score part from critique if found
            critique_fallback = re.sub(r'\b(?:score|rating)[\s:]*(\b(?:[1-9]|10)\b)', '', critique_fallback,
                                       flags=re.IGNORECASE).strip('{}[]().,: "\'')
        return critique_fallback or "Critique parsing failed.", score_fallback
    except Exception as e:
        print(
            f"❌ Error with Ollama ({critic_profile['model']} for {critic_profile['name']}) for '{name_to_critique}': {e}")
        return "Error during critique.", None


def run_synthesis_critique(name_to_critique: str, theme: str, previous_critiques: list, profile: dict) -> tuple[
    str | None, str | None]:
    system_prompt = profile["system_prompt"]
    critiques_summary_text = "\n".join([
        f"- {c['critic_name']} (Score: {c['score'] if c['score'] is not None else 'N/A'}): \"{c['critique']}\""
        for c in previous_critiques
    ])

    user_prompt = (
        f"The overall theme is: '{theme}'.\nThe name being evaluated is: '{name_to_critique}'.\n\n"
        f"Here are the critiques from other analysts:\n{critiques_summary_text}\n\n"
        f"Based on your persona, please provide your synthesis. "
        f"Format your response STRICTLY as a JSON object with two keys: 'summary' (string: your overall analysis and summary of points) and 'verdict' (string: your final recommendation like 'Strongly Recommended', 'Recommended with Considerations', 'Proceed with Caution', 'Not Recommended')."
        f"Example: {{\"summary\": \"The name is catchy and modern, aligning with Zip's view, but Mr. Finch's concerns about professionalism are valid for a broader market.\", \"verdict\": \"Proceed with Caution\"}}"
    )
    model_options = {"temperature": profile.get("temperature", 0.5)}

    try:
        response = ollama.chat(
            model=profile["model"],
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            options=model_options,
            format="json"  # Request JSON format
        )
        content = response['message']['content'].strip()
        json_str = content
        if not (content.startswith("{") and content.endswith("}")):  # Fallback parsing
            match_json_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL)
            if match_json_block:
                json_str = match_json_block.group(1)
            else:
                match_loose = re.search(r'(\{[\s\S]*?\})', content, re.DOTALL)
                if match_loose:
                    json_str = match_loose.group(1)
                else:
                    raise json.JSONDecodeError("No JSON object found in synthesis response", content, 0)

        data = json.loads(json_str)
        summary = data.get('summary')
        verdict = data.get('verdict')
        return summary, verdict
    except (json.JSONDecodeError, KeyError) as je:
        print(
            f"⚠️ Synthesis Critic ({profile['name']}) for '{name_to_critique}' didn't return valid JSON or keys. Error: {je}. Raw: {content[:200]}...")
        return content or "Synthesis parsing failed.", "Verdict unclear"  # Fallback
    except Exception as e:
        print(f"❌ Error with Ollama ({profile['model']} for Synthesis) for '{name_to_critique}': {e}")
        return "Error during synthesis.", "Error"


def generate_html_report(theme: str, results: list, filename="name_critique_report.html"):
    print(f"\n📄 Generating HTML report: {filename}...")
    # ... (web check method note logic remains the same) ...
    web_check_method_note = "Web checking via Google API was disabled or not configured for this session."  # Simplified for brevity, keep original if preferred
    if GOOGLE_API_CLIENT_AVAILABLE and GOOGLE_API_CONFIGURED:
        web_check_method_note = "Web check using Google Custom Search API."
    elif GOOGLE_API_CLIENT_AVAILABLE and not GOOGLE_API_CONFIGURED:
        web_check_method_note = "Google API not fully configured (missing credentials)."
    elif not GOOGLE_API_CLIENT_AVAILABLE:
        web_check_method_note = "Google API client library not found for web checks."

    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Name Generation & Critique Report: {theme}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f0f2f5; color: #333; line-height: 1.6; }}
        .container {{ max-width: 950px; margin: 20px auto; padding: 25px; background-color: #fff; box-shadow: 0 6px 20px rgba(0,0,0,0.08); border-radius: 10px; }}
        h1 {{ color: #1a2533; text-align: center; border-bottom: 3px solid #4a90e2; padding-bottom: 15px; margin-bottom: 25px; font-size: 2.2em; }}
        .name-block {{ margin-bottom: 40px; padding: 25px; background-color: #fcfdff; border: 1px solid #dce4ec; border-radius: 8px; box-shadow: 0 3px 10px rgba(0,0,0,0.04);}}
        .name-title {{ font-size: 2.4em; color: #2c5282; margin-bottom: 25px; text-align: center; font-weight: 600; }}
        .critic-review-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; margin-bottom:25px;}}
        .critic-review {{ border: 1px solid #cbd5e0; border-left-width: 6px; padding: 18px; border-radius: 6px; background-color: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.03);}}
        .critic-header {{ display: flex; align-items: center; margin-bottom: 10px;}}
        .critic-emoji {{ font-size: 2em; margin-right: 12px; }}
        .critic-name {{ font-weight: bold; font-size: 1.25em; color: #4a5568; }}
        .critique-text {{ margin: 10px 0; font-style: italic; color: #52525b; white-space: pre-wrap; }} /* zinc-600 */
        .score {{ font-weight: bold; font-size: 1.1em; }} .score-good {{ color: #38a169; }} .score-medium {{ color: #dd6b20; }} .score-bad {{ color: #e53e3e; }}
        .synthesis-critique {{ margin-top:30px; padding: 20px; background-color: #e6f0ff; border: 1px solid #a3c6ff; border-left: 6px solid #2b6cb0; border-radius: 6px; }}
        .synthesis-header {{ font-size: 1.4em; font-weight: bold; color: #2c5282; margin-bottom:10px; display:flex; align-items:center;}}
        .synthesis-summary, .synthesis-verdict {{ margin-bottom: 8px; }}
        .synthesis-verdict strong {{ color: #2a4365; }}
        .meta-info {{ text-align: center; font-size: 0.95em; color: #555; margin-bottom: 25px; }}
        .web-check-note {{ font-size: 0.9em; color: #666; text-align: center; margin-bottom:20px; padding:8px; background-color:#eef2f7; border-radius:4px;}}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #777; }}
    </style></head><body><div class="container">
        <h1>Name Generation & Critique Report</h1>
        <div class="meta-info"><strong>Theme:</strong> {theme}<br>Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <p class='web-check-note'>{web_check_method_note}</p>
    """
    if not results: html_content += "<p style='text-align:center;'>No names were approved and critiqued.</p>"
    for item in results:
        name = item["name"]
        critiques_list = item["critiques"]
        synthesis = item.get("synthesis")  # Get synthesis if available

        html_content += f"""<div class="name-block"><h2 class="name-title">{name}</h2><div class="critic-review-grid">"""
        for critic_res in critiques_list:
            critique_text_escaped = (critic_res["critique"] or "N/A").replace('<', '&lt;').replace('>', '&gt;')
            score_display, score_class = "N/A", ""
            if critic_res["score"] is not None:
                score_display = str(critic_res["score"])
                if critic_res["score"] >= 8:
                    score_class = "score-good"
                elif critic_res["score"] >= 5:
                    score_class = "score-medium"
                else:
                    score_class = "score-bad"

            # Assign border colors based on critic name (simplified)
            border_color = {"Finch": "#c0392b", "Hayes": "#27ae60", "Zip": "#f39c12", "Moon": "#8e44ad"}.get(
                next((n for n in ["Finch", "Hayes", "Zip", "Moon"] if n in critic_res["critic_name"]), ""), "#3498db")

            html_content += f"""
                <div class="critic-review" style="border-left-color: {border_color};">
                    <div class="critic-header"><span class="critic-emoji">{critic_res["emoji"]}</span><span class="critic-name">{critic_res["critic_name"]}:</span></div>
                    <p class="critique-text">"{critique_text_escaped}"</p>
                    <p class="score">Score: <span class="{score_class}">{score_display}</span>/10</p>
                </div>"""
        html_content += "</div>"  # Close critic-review-grid

        if synthesis:
            summary_escaped = (synthesis["summary"] or "N/A").replace('<', '&lt;').replace('>', '&gt;')
            verdict_escaped = (synthesis["verdict"] or "N/A").replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"""
                <div class="synthesis-critique">
                    <div class="synthesis-header"><span class="critic-emoji">{synthesis["emoji"]}</span>{synthesis["critic_name"]}:</div>
                    <p class="synthesis-summary"><strong>Summary:</strong> {summary_escaped}</p>
                    <p class="synthesis-verdict"><strong>Verdict:</strong> {verdict_escaped}</p>
                </div>"""
        html_content += "</div>"  # Close name-block
    html_content += f"""
        <div class="footer">Models: Generator - {GENERATOR_MODEL}, Critics - {CRITIC_BASE_MODEL}, Synthesis - {SYNTHESIS_CRITIC_MODEL}.</div>
        </div></body></html>"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ HTML report successfully generated: {filename}")
    except IOError as e:
        print(f"❌ Error writing HTML file: {e}")


# --- Main Application Logic ---
def main():
    global GOOGLE_API_CONFIGURED

    # --- NLTK Setup ---
    print("⚙️ Initializing NLTK resources (one-time setup if needed)...")
    load_nltk_word_bank()  # This also handles downloads

    print("\n✨ Welcome to the Multi-Personality Name Idea Generator & Critic! ✨")
    print(f"Generator Model: {GENERATOR_MODEL}, Critic Base: {CRITIC_BASE_MODEL}, Synthesis: {SYNTHESIS_CRITIC_MODEL}")
    # ... (Google API status print remains the same) ...
    if GOOGLE_API_CLIENT_AVAILABLE and GOOGLE_API_CONFIGURED:
        print("✅ Google Custom Search API is configured.")
    else:
        print("🔴 Google Custom Search API is NOT configured/available. Web checking disabled.")
    print(f"Number of standard critics: {len(CRITIC_PERSONALITIES)} (+1 Synthesis Critic)")
    print("-" * 70)

    theme = input("➡️ Enter the theme for the names: ")
    if not theme.strip():
        print("⚠️ No theme provided. Exiting.")
        return

    initial_generated_names = generate_names(theme, NUMBER_OF_NAMES_TO_GENERATE)
    if not initial_generated_names:
        print(f"😥 No names were generated for '{theme}'. Try a different theme or check Ollama setup.")
        return

    print(f"\n--- Filtering {len(initial_generated_names)} names with web check (if enabled) ---")
    approved_names = []
    if GOOGLE_API_CONFIGURED:
        for i, name_to_check in enumerate(initial_generated_names):
            if len(approved_names) >= MAX_NAMES_TO_CRITIQUE:
                print(f"\n🏁 Reached target of {MAX_NAMES_TO_CRITIQUE} approved names. Skipping further web checks.")
                break
            if not GOOGLE_API_CONFIGURED:  # Check again in case it was disabled mid-run
                print(f"    ☑️ Approving '{name_to_check}' (Google API disabled mid-run).")
                approved_names.append(name_to_check)
                continue

            is_problematic, reason = check_name_online_google_api(name_to_check, theme)
            if is_problematic:
                print(f"    ⛔ Rejected: '{name_to_check}'. Reason: {reason}")
            else:
                print(f"    ✅ Approved: '{name_to_check}'. Reason: {reason}")
                approved_names.append(name_to_check)

            if GOOGLE_API_CONFIGURED and i < len(initial_generated_names) - 1:  # if API still on and not the last name
                time.sleep(API_REQUEST_DELAY)
    else:
        print("\n⚠️ Google API web checking disabled. Using first generated names (up to limit) without online checks.")
        approved_names = initial_generated_names[:MAX_NAMES_TO_CRITIQUE]

    if not approved_names and len(initial_generated_names) > 0:  # If web check rejected all, but we have names
        print(
            f"\nℹ️ Web checks rejected all names or was disabled. Taking first {MAX_NAMES_TO_CRITIQUE} generated names for critique as fallback.")
        approved_names = initial_generated_names[:MAX_NAMES_TO_CRITIQUE]

    if not approved_names:
        print(f"\n😥 No names available for critique. Unable to proceed.")
        return

    final_names_for_critique = list(dict.fromkeys(approved_names))[:MAX_NAMES_TO_CRITIQUE]  # Ensure unique and capped
    print(
        f"\n--- Critiquing {len(final_names_for_critique)} names with {len(CRITIC_PERSONALITIES)} personalities + Synthesis Critic ---")

    all_results_for_report = []
    for i, name in enumerate(final_names_for_critique):
        print(f"\n🔄 Critiquing Name {i + 1}/{len(final_names_for_critique)}: '{name}'")
        current_name_critiques_for_synthesis = []  # For feeding into synthesis critic

        name_report_entry = {"name": name, "critiques": []}

        for critic_profile in CRITIC_PERSONALITIES:
            print(
                f"   ↳ Asking {critic_profile['emoji']} {critic_profile['name']} (Model: {critic_profile['model']}, Temp: {critic_profile.get('temperature', 'N/A')})...")
            critique, score = critique_name_with_personality(name, theme, critic_profile)
            critique_entry = {
                "critic_name": critic_profile["name"], "emoji": critic_profile["emoji"],
                "critique": critique, "score": score
            }
            name_report_entry["critiques"].append(critique_entry)
            current_name_critiques_for_synthesis.append(critique_entry)  # Add for synthesis
            crit_snippet = (critique[:77] + "...") if critique and len(critique) > 80 else (critique or "N/A")
            print(f"     🗣️ Critique: {crit_snippet} Score: {score if score is not None else 'N/A'}")

        # --- Run Synthesis Critic ---
        if current_name_critiques_for_synthesis:  # Only if there are prior critiques
            print(
                f"   ↳ Asking {SYNTHESIS_CRITIC_PROFILE['emoji']} {SYNTHESIS_CRITIC_PROFILE['name']} for synthesis...")
            summary, verdict = run_synthesis_critique(name, theme, current_name_critiques_for_synthesis,
                                                      SYNTHESIS_CRITIC_PROFILE)
            name_report_entry["synthesis"] = {
                "critic_name": SYNTHESIS_CRITIC_PROFILE["name"], "emoji": SYNTHESIS_CRITIC_PROFILE["emoji"],
                "summary": summary, "verdict": verdict
            }
            sum_snippet = (summary[:77] + "...") if summary and len(summary) > 80 else (summary or "N/A")
            print(f"     🧐 Summary: {sum_snippet} Verdict: {verdict or 'N/A'}")

        all_results_for_report.append(name_report_entry)

    if all_results_for_report:
        safe_theme = re.sub(r'[^\w\s-]', '', theme.lower()).strip()
        safe_theme = re.sub(r'[-\s]+', '_', safe_theme) if safe_theme else "report"
        report_filename = f"name_report_{safe_theme}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        generate_html_report(theme, all_results_for_report, filename=report_filename)
    else:
        print("\nNo results to generate a report for.")

    print("\n👋 All done!")


if __name__ == "__main__":
    # Ensure NLTK downloads path is discoverable if running in unusual environments
    # nltk.data.path.append("/path/to/custom/nltk_data") # If needed
    main()