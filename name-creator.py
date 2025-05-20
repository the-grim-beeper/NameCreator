import ollama
import json
import datetime
import re
import os
import time

# For loading .env file
try:
    from dotenv import load_dotenv

    load_dotenv()  # Loads variables from .env into os.environ
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
GENERATOR_MODEL = 'gemma3'  # Or your preferred name generator model
CRITIC_BASE_MODEL = 'deepseek-r1'  # Base model for critics, e.g., 'mistral', 'llama3', 'gemma3'
# If using Gemma 3 for critics, ensure it's pulled in Ollama, e.g., 'gemma:27b' or the specific tag you have.
# CRITIC_BASE_MODEL = 'gemma' # Example if you were using Gemma for critics

NUMBER_OF_NAMES_TO_GENERATE = 60
MAX_NAMES_TO_CRITIQUE = 30
API_REQUEST_DELAY = 1.0  # Delay in seconds between Google API calls

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
        if not GOOGLE_API_KEY:
            print("   Missing GOOGLE_API_KEY (check .env file or environment variables).")
        if not GOOGLE_CSE_ID:
            print("   Missing GOOGLE_CSE_ID (check .env file or environment variables).")
        print("   Web checking via Google API will be disabled.")
else:
    print("   Web checking via Google API is disabled (client library not found).")

# --- Critic Personalities ---
# Temperature set to 1.0 for all critics for more creative/diverse responses
CRITIC_PERSONALITIES = [
    {
        "name": "Mr. Alistair Finch (CEO)",
        "emoji": "👴",
        "system_prompt": (
            "You are Alistair Finch, a pragmatic and experienced CEO in his late 50s. "
            "You value names that sound professional, trustworthy, marketable, and have a strong, clear presence. "
            "Avoid overly trendy, frivolous, or difficult-to-pronounce names. Focus on longevity and brand potential."
        ),
        "model": CRITIC_BASE_MODEL,
        "temperature": 1.0
    },
    {
        "name": "Dr. Evelyn Hayes (Social Advocate)",
        "emoji": "⚖️",
        "system_prompt": (
            "You are Dr. Evelyn Hayes, a progressive, left-leaning academic and social advocate in her 40s. "
            "You appreciate names that are inclusive, thoughtful, unique, and perhaps hint at ethical or sustainable values. "
            "You are critical of names that might seem exploitative, generic, overly corporate, or culturally insensitive. "
            "You value creativity and meaning."
        ),
        "model": CRITIC_BASE_MODEL,
        "temperature": 1.0
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
        "model": CRITIC_BASE_MODEL,
        "temperature": 1.0
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
        "model": CRITIC_BASE_MODEL,
        "temperature": 1.0
    }
]


# --- Helper Functions ---

def generate_names(theme: str, count: int) -> list[str]:
    # Temperature for generator model can also be set if desired, e.g., higher for more diverse names
    generator_options = {"temperature": 0.9}  # Example: slightly less than critics for more focus
    prompt = (
        f"You are a creative naming expert. "
        f"Suggest {count} unique and imaginative names based on the theme: '{theme}'. "
        f"Provide only the names, one name per line, without any additional text, numbering, or markdown like '*' or '-'."
        f"Just the names, plain text."
    )
    print(
        f"\n🤖 Asking {GENERATOR_MODEL} (temp: {generator_options.get('temperature', 'default')}) to generate {count} names for the theme: '{theme}'...")
    try:
        response = ollama.chat(
            model=GENERATOR_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options=generator_options
        )
        content = response['message']['content'].strip()
        names = [
            name.strip().lstrip('*- ').rstrip('.,')
            for name in content.split('\n')
            if name.strip() and len(name.strip().lstrip('*- ').rstrip('.,')) > 1
        ]
        if not names:
            print("⚠️ Generator model did not return names as expected, attempting broader parse.")
            potential_names = re.findall(r'\b[A-Z][a-zA-Z0-9]{2,}\b', content)
            if not potential_names:
                potential_names = re.split(r'[,\n;.]+', content)
            names = [name.strip().lstrip('*- ').rstrip('.,') for name in potential_names if
                     name.strip() and len(name.strip().lstrip('*- ').rstrip('.,')) > 2][:count]
        if not names:
            print(
                f"❌ Error: The generator model ({GENERATOR_MODEL}) did not return any usable names. Response: {content}")
            return []
        print(f"✅ Got {len(names)} initial names.")
        return names[:count]
    except Exception as e:
        print(f"❌ Error communicating with Ollama (Generator: {GENERATOR_MODEL}): {e}")
        return []


def check_name_online_google_api(name: str, theme: str) -> tuple[bool, str]:
    global GOOGLE_API_CONFIGURED

    if not GOOGLE_API_CONFIGURED:
        return False, "Web check skipped (Google Custom Search API not configured or disabled)."

    print(f"    🔎 Web checking name (via Google API): '{name}'...")
    search_query = f"\"{name}\" company OR business OR official site"

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=search_query, cx=GOOGLE_CSE_ID, num=5).execute()
        search_items = res.get('items', [])

        if not search_items:
            print(f"       No direct results found for '{search_query}'.")
        else:
            print(f"       Found {len(search_items)} results for '{search_query}'. Analyzing top results...")
            name_lower_no_space = name.lower().replace(" ", "")
            name_lower_with_dash = name.lower().replace(" ", "-")
            domain_variations = [name_lower_no_space, name_lower_with_dash]
            problematic_indicators_domains = [
                "linkedin.com/company", "facebook.com/", "instagram.com/",
                "crunchbase.com/organization", "bloomberg.com/profile/company",
            ]

            for item in search_items:
                title = item.get('title', '')
                link = item.get('link', '')
                print(f"       Found: {link} (Title: {title})")
                url_lower = link.lower()
                title_lower = title.lower()

                if any(domain_var in url_lower for domain_var in domain_variations):
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url_lower).netloc
                        if any(domain_var in domain for domain_var in domain_variations):
                            return True, f"Potential existing company with close domain match found: {link}"
                    except ImportError:
                        pass

                if any(indicator in url_lower for indicator in problematic_indicators_domains):
                    return True, f"Potential existing business entity found (platform/database): {link}"

                if name.lower() in title_lower and (
                        "company" in title_lower or "inc." in title_lower or "llc" in title_lower or
                        name.lower() == title_lower.split('|')[0].strip().lower() or
                        name.lower() == title_lower.split('-')[0].strip().lower()
                ):
                    return True, f"Title strongly suggests existing company: '{title}' at {link}"

        common_words_list = ["the", "a", "is", "in", "of", "for", "and", "tech", "solution", "group", "global", "world",
                             "net", "corp", "io", "app", "data", "cloud", "link", "digital", "systems", "labs", "co",
                             "inc", "biz", "info", "online", "services"]
        name_parts = name.lower().split()
        if len(name_parts) == 1 and name_parts[0] in common_words_list:
            return True, f"Name '{name}' is a very common, likely unbrandable single word."
        if len(name_parts) > 0 and all(part in common_words_list for part in name_parts):
            return True, f"Name '{name}' consists entirely of very common, likely unbrandable words."

        return False, "Name seems relatively clear based on Google API check (top results)."

    except HttpError as e:
        error_content_str = "Unknown error"
        try:
            error_details = json.loads(e.content).get('error', {})
            error_message = error_details.get('message', e.resp.reason)
            error_reason = error_details.get('reason', '')
            error_content_str = f"{error_message} (Reason: {error_reason}, Status: {e.resp.status})"

            if 'quotaExceeded' in error_reason or 'usageLimits' in error_reason or \
                    'billingNotEnabled' in error_reason or e.resp.status in [429, 403]:
                print(f"    ⚠️ Google API Quota/Billing/Permission Issue for '{name}': {error_content_str}")
                print("    Disabling further Google API checks for this session.")
                GOOGLE_API_CONFIGURED = False
                return False, f"Web check skipped due to API quota/permission/billing issue: {error_message}"
        except (json.JSONDecodeError, AttributeError):
            error_content_str = str(e)
            print(
                f"    ⚠️ Google API HttpError (non-JSON response or unexpected structure) for '{name}': {error_content_str}")

        return False, f"Google API HttpError: {error_content_str}"
    except Exception as e:
        print(f"    ⚠️ Unexpected error during Google API web check for '{name}': {e}")
        return False, f"Unexpected Google API error: {e}"


def critique_name_with_personality(name_to_critique: str, theme: str, critic_profile: dict) -> tuple[str, int | None]:
    system_prompt = critic_profile["system_prompt"]
    user_prompt = (
        f"The overall theme is '{theme}'. "
        f"The specific name to critique is: '{name_to_critique}'.\n\n"
        f"Based on your persona, provide a brief critique (1-3 sentences) and a score from 1 (poor) to 10 (excellent). "
        f"Format your response STRICTLY as a JSON object with two keys: 'critique' (string) and 'score' (integer). "
        f"Example: {{\"critique\": \"This name feels a bit outdated for the current market.\", \"score\": 4}}"
    )

    model_options = {}
    # Set temperature from critic's profile (and other params if defined there)
    if "temperature" in critic_profile:
        model_options["temperature"] = critic_profile["temperature"]
    if "top_k" in critic_profile:  # Example for other parameters
        model_options["top_k"] = critic_profile["top_k"]
    if "top_p" in critic_profile:
        model_options["top_p"] = critic_profile["top_p"]

    try:
        response = ollama.chat(
            model=critic_profile["model"],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options=model_options if model_options else None  # Pass options if any are set
        )
        content = response['message']['content'].strip()
        json_str = None
        match_json_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL)
        if match_json_block:
            json_str = match_json_block.group(1)
        elif content.startswith("{") and content.endswith("}"):
            json_str = content
        else:
            match_loose = re.search(r'(\{[\s\S]*?\})', content, re.DOTALL)
            if match_loose:
                json_str = match_loose.group(1)
        if not json_str:
            raise json.JSONDecodeError("No JSON object found in response", content, 0)
        data = json.loads(json_str)
        critique = data.get('critique', "Critique not provided or malformed.")
        score_val = data.get('score')
        score = None
        if score_val is not None:
            try:
                score = int(score_val)
                if not (1 <= score <= 10): score = None
            except ValueError:
                score = None
        return critique, score
    except json.JSONDecodeError as je:
        print(
            f"⚠️ Warning: Critic ({critic_profile['name']}) did not return valid JSON for '{name_to_critique}'. Error: {je}. Response: {content[:200]}...")
        score_fallback = None
        score_match = re.search(r'\b(?:score|rating)[\s:]*(\b(?:[1-9]|10)\b)', content, re.IGNORECASE)
        if score_match:
            try:
                score_fallback = int(score_match.group(1))
            except ValueError:
                pass
        critique_fallback = content
        if score_fallback is not None:
            critique_fallback = re.sub(r'\b(?:score|rating)[\s:]*(\b(?:[1-9]|10)\b)', '', critique_fallback,
                                       flags=re.IGNORECASE).strip()
        return critique_fallback, score_fallback
    except Exception as e:
        print(
            f"❌ Error communicating with Ollama ({critic_profile['model']} for {critic_profile['name']}) for name '{name_to_critique}': {e}")
        return "Error during critique.", None


def generate_html_report(theme: str, results: list, filename="name_critique_report.html"):
    print(f"\n📄 Generating HTML report: {filename}...")
    web_check_method_note = "Web checking via Google API was disabled or not configured for this session."
    if GOOGLE_API_CLIENT_AVAILABLE and GOOGLE_API_CONFIGURED:
        web_check_method_note = "Web check for name availability performed using Google Custom Search API (results are indicative)."
    elif GOOGLE_API_CLIENT_AVAILABLE and not GOOGLE_API_CONFIGURED:
        web_check_method_note = "Web checking via Google API was not fully configured (missing credentials)."
    elif not GOOGLE_API_CLIENT_AVAILABLE:
        web_check_method_note = "Web checking via Google API was disabled (client library not found)."

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Name Generation & Critique Report: {theme}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: 20px auto; padding: 25px; background-color: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 15px; margin-bottom: 20px; font-size: 2em; }}
            h2 {{ color: #34495e; margin-top: 35px; border-bottom: 1px solid #eee; padding-bottom: 8px; font-size: 1.6em; }}
            .name-block {{ margin-bottom: 35px; padding: 20px; background-color: #fdfdfd; border: 1px solid #e0e0e0; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}}
            .name-title {{ font-size: 2em; color: #2980b9; margin-bottom: 20px; text-align: center; }}
            .critic-review {{ border: 1px solid #bdc3c7; border-left-width: 6px; padding: 15px; margin-bottom: 15px; border-radius: 4px; background-color: #fff; }}
            .critic-header {{ display: flex; align-items: center; margin-bottom: 8px;}}
            .critic-emoji {{ font-size: 1.8em; margin-right: 10px; }}
            .critic-name {{ font-weight: bold; font-size: 1.2em; color: #555; }}
            .critique-text {{ margin: 8px 0; font-style: italic; color: #444; white-space: pre-wrap; }}
            .score {{ font-weight: bold; font-size: 1.1em; }}
            .score-good {{ color: #27ae60; }}
            .score-medium {{ color: #f39c12; }}
            .score-bad {{ color: #c0392b; }}
            .meta-info {{ text-align: center; font-size: 0.95em; color: #666; margin-bottom: 25px; }}
            .web-check-note {{ font-size: 0.85em; color: #777; text-align: center; margin-bottom:15px;}}
            .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Name Generation & Critique Report</h1>
            <div class="meta-info">
                <p><strong>Theme:</strong> {theme}</p>
                <p>Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            <p class='web-check-note'>{web_check_method_note}</p>
    """
    if not results:
        html_content += "<p style='text-align:center;'>No names were approved and critiqued.</p>"
    for item in results:
        name = item["name"]
        critiques_list = item["critiques"]
        html_content += f"""
            <div class="name-block">
                <h2 class="name-title">{name}</h2>
        """
        for critic_response in critiques_list:
            critic_name = critic_response["critic_name"]
            emoji = critic_response["emoji"]
            critique_text = critic_response["critique"] if critic_response["critique"] else "No critique text provided."
            critique_text_escaped = critique_text.replace('<', '&lt;').replace('>', '&gt;')
            score = critic_response["score"]
            score_display = "N/A"
            score_class = ""
            if score is not None:
                score_display = str(score)
                if score >= 8:
                    score_class = "score-good"
                elif score >= 5:
                    score_class = "score-medium"
                else:
                    score_class = "score-bad"
            border_color = "#3498db"
            if "Finch" in critic_name:
                border_color = "#c0392b"
            elif "Hayes" in critic_name:
                border_color = "#27ae60"
            elif "Zip" in critic_name:
                border_color = "#f39c12"
            elif "Seraphina" in critic_name or "Moon" in critic_name:
                border_color = "#8e44ad"
            html_content += f"""
                <div class="critic-review" style="border-left-color: {border_color};">
                    <div class="critic-header">
                        <span class="critic-emoji">{emoji}</span>
                        <span class="critic-name">{critic_name}:</span>
                    </div>
                    <p class="critique-text">"{critique_text_escaped}"</p>
                    <p class="score">Score: <span class="{score_class}">{score_display}</span>/10</p>
                </div>
            """
        html_content += "</div>"
    html_content += f"""
        <div class="footer">
            <p>Models used: Generator - {GENERATOR_MODEL}, Critics - {CRITIC_BASE_MODEL} (customized per personality).</p>
        </div>
        </div>
    </body>
    </html>
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ HTML report successfully generated: {filename}")
    except IOError as e:
        print(f"❌ Error writing HTML file: {e}")


# --- Main Application Logic ---
def main():
    global GOOGLE_API_CONFIGURED

    print("✨ Welcome to the Multi-Personality Name Idea Generator & Critic! ✨")
    print(f"Generator Model: {GENERATOR_MODEL}")
    print(f"Critic Base Model: {CRITIC_BASE_MODEL}")
    if not GOOGLE_API_CLIENT_AVAILABLE:
        print("🔴 Google API client library not found. Web checking disabled.")
    elif not GOOGLE_API_CONFIGURED:
        print("🔴 Google API credentials not configured (check .env or environment variables). Web checking disabled.")
    else:
        print("✅ Google Custom Search API is configured for web checking.")
    print(f"Number of critics: {len(CRITIC_PERSONALITIES)}")
    print("-" * 50)

    theme = input("➡️ Enter the theme for the names: ")
    if not theme.strip():
        print("⚠️ No theme provided. Exiting.")
        return

    initial_generated_names = generate_names(theme, NUMBER_OF_NAMES_TO_GENERATE)
    if not initial_generated_names:
        print(f"😥 No names were generated for the theme '{theme}'. Try a different theme or check your Ollama setup.")
        return

    print(f"\n--- Filtering {len(initial_generated_names)} generated names with web check (if enabled) ---")
    approved_names = []

    if GOOGLE_API_CONFIGURED:
        for name_to_check in initial_generated_names:
            if len(approved_names) >= MAX_NAMES_TO_CRITIQUE:
                print(f"\n🏁 Reached target of {MAX_NAMES_TO_CRITIQUE} approved names. Skipping further web checks.")
                break

            if not GOOGLE_API_CONFIGURED:
                print(f"    ☑️ Approving '{name_to_check}' (Google API checking was disabled mid-run).")
                approved_names.append(name_to_check)
                continue

            is_problematic, reason = check_name_online_google_api(name_to_check, theme)
            if is_problematic:
                print(f"    ⛔ Rejected Name: '{name_to_check}'. Reason: {reason}")
            else:
                if "Web check skipped" in reason or "API error" in reason or "HttpError" in reason:
                    print(f"    ⚠️ Name '{name_to_check}' approved due to web check issue: {reason}")
                else:
                    print(f"    ✅ Approved Name: '{name_to_check}'. Reason: {reason}")
                approved_names.append(name_to_check)

            if GOOGLE_API_CONFIGURED and len(approved_names) < MAX_NAMES_TO_CRITIQUE:
                print(f"    ... pausing for {API_REQUEST_DELAY}s before next API call ...")
                time.sleep(API_REQUEST_DELAY)
    else:
        print(
            "\n⚠️ Google API web checking is disabled. Using the first generated names (up to limit) without online checks.")
        approved_names = initial_generated_names[:MAX_NAMES_TO_CRITIQUE]

    if len(approved_names) < MAX_NAMES_TO_CRITIQUE and (
            not GOOGLE_API_CONFIGURED or not GOOGLE_API_CLIENT_AVAILABLE) and len(initial_generated_names) > 0:
        current_approved_count = len(approved_names)
        needed = MAX_NAMES_TO_CRITIQUE - current_approved_count
        if needed > 0:
            additional_names_pool = [n for n in initial_generated_names if n not in approved_names]
            additional_names = additional_names_pool[:needed]
            if additional_names:
                print(
                    f"\nℹ️ Web checking was disabled; adding {len(additional_names)} more generated names to reach target for critique.")
                approved_names.extend(additional_names)

    if not approved_names:
        print(f"\n😥 No names were approved or available for critique. Unable to proceed.")
        return

    final_names_for_critique = approved_names[:MAX_NAMES_TO_CRITIQUE]
    if len(final_names_for_critique) < MAX_NAMES_TO_CRITIQUE and len(initial_generated_names) > 0 and len(
            approved_names) > 0:
        print(
            f"\nℹ️ Note: Fewer than {MAX_NAMES_TO_CRITIQUE} names available/approved for critique ({len(final_names_for_critique)}).")

    print(f"\n--- Critiquing {len(final_names_for_critique)} names with {len(CRITIC_PERSONALITIES)} personalities ---")

    all_results_for_report = []
    for i, name in enumerate(final_names_for_critique):
        print(f"\n🔄 Critiquing Name {i + 1}/{len(final_names_for_critique)}: '{name}'")
        current_name_critiques = []
        for critic_profile in CRITIC_PERSONALITIES:
            critic_temp = critic_profile.get("temperature", "default (0.8)")  # Get temp for logging
            print(
                f"   ↳ Asking {critic_profile['emoji']} {critic_profile['name']} (model: {critic_profile['model']}, temp: {critic_temp})...")
            critique, score = critique_name_with_personality(name, theme, critic_profile)
            current_name_critiques.append({
                "critic_name": critic_profile["name"],
                "emoji": critic_profile["emoji"],
                "critique": critique,
                "score": score
            })
            critique_snippet = (critique[:77] + "...") if critique and len(critique) > 80 else critique
            print(
                f"     🗣️ Critique: {critique_snippet if critique_snippet else 'N/A'} Score: {score if score is not None else 'N/A'}")

        all_results_for_report.append({
            "name": name,
            "critiques": current_name_critiques
        })

    if all_results_for_report:
        safe_theme_filename = re.sub(r'[^\w\s-]', '', theme.lower())
        safe_theme_filename = re.sub(r'[-\s]+', '_', safe_theme_filename).strip('_')
        if not safe_theme_filename: safe_theme_filename = "report"
        report_filename = f"name_report_{safe_theme_filename}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        generate_html_report(theme, all_results_for_report, filename=report_filename)
    else:
        print("\nNo results to generate a report for.")

    print("\n👋 All done!")


if __name__ == "__main__":
    main()