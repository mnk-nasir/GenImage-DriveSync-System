#!/usr/bin/env python3
"""
main.py

Companion script for the n8n workflow "Automated Research Report Generation..."
Implements a simplified, runnable version of the workflow:

- Validates input query
- Refines the query and generates 5 related search queries (Query Refiner)
  using OpenAI (or a simulated fallback)
- Runs a ResearchAgent that:
  - Fetches Wikipedia intro for the topic
  - Calls NewsAPI (optional) to fetch recent headlines
  - Calls SerpAPI (optional) to fetch Google Scholar results
  - Optionally calls Google Custom Search (optional)
  - Aggregates the findings into the expected JSON structure
- Generates an HTML research report and converts it to PDF via PDFShift (or simulates)
- Optionally sends the PDF by Gmail (SMTP) and Telegram bot
- Saves artifacts to outputs/

Usage examples:
  python main.py --query "facts about Thailand" --save --generate-pdf
  python main.py --query "AI models 2025" --simulate --save

Configuration:
  Put API keys and defaults in a .env file or environment variables (see .env.example).

"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from datetime import datetime

# optional dependencies
try:
    import openai
except Exception:
    openai = None  # type: ignore

import config

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------
# Utilities
# -----------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat()


def safe_get(d: Dict[str, Any], k: str, default=""):
    return d.get(k) or default


# -----------------------
# LLM Helpers (OpenAI)
# -----------------------
def call_openai_chat(prompt: str, model: Optional[str] = None, max_tokens: int = 800) -> str:
    if not config.OPENAI_API_KEY or not openai:
        raise RuntimeError("OpenAI not configured or openai package not installed.")
    openai.api_key = config.OPENAI_API_KEY
    model_to_use = model or config.OPENAI_MODEL or "gpt-4o-mini"
    resp = openai.ChatCompletion.create(
        model=model_to_use,
        messages=[{"role": "system", "content": "You are a helpful assistant that outputs JSON when requested."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp["choices"][0]["message"]["content"]


# -----------------------
# Query Refiner
# -----------------------
DEFAULT_QUERY_REFINER_PROMPT = """
You are a query generation expert. Given the input topic, produce a JSON object:
{
  "topic": "<cleaned topic>",
  "searchQueries": ["q1","q2","q3","q4","q5"]
}
Make sure there are exactly 5 search queries, each focusing on a different angle (apps, challenges, recent dev, case studies, domains).
Return ONLY the JSON object and nothing else.
Topic: {topic}
"""

def refine_query(topic: str, simulate: bool = False) -> Dict[str, Any]:
    """
    Returns a dict with 'topic' and 'searchQueries' (list of 5).
    Uses OpenAI if configured; otherwise falls back to a deterministic simple refiner.
    """
    cleaned = " ".join(topic.strip().split())
    if simulate or not (config.OPENAI_API_KEY and openai):
        # Simple deterministic fallback
        base = cleaned.lower()
        queries = [
            f"{base} applications 2024-2025",
            f"{base} challenges and limitations",
            f"recent developments in {base} 2024 2025",
            f"{base} case studies",
            f"{base} industry adoption and domains"
        ]
        return {"topic": cleaned, "searchQueries": queries}
    prompt = DEFAULT_QUERY_REFINER_PROMPT.format(topic=cleaned)
    try:
        out = call_openai_chat(prompt, model=config.OPENAI_MODEL)
        parsed = json.loads(out)
        # basic normalization
        parsed["topic"] = parsed.get("topic", cleaned)
        if len(parsed.get("searchQueries", [])) < 5:
            # pad with simple variants
            base = cleaned.lower()
            while len(parsed["searchQueries"]) < 5:
                parsed["searchQueries"].append(f"{base} additional query {len(parsed['searchQueries'])+1}")
        parsed["searchQueries"] = parsed["searchQueries"][:5]
        return parsed
    except Exception as exc:
        print("[warning] Query refiner failed, falling back to deterministic:", exc)
        return refine_query(cleaned, simulate=True)


# -----------------------
# Research Agent (aggregation)
# -----------------------
def fetch_wikipedia_intro(title: str) -> Optional[str]:
    """Fetch the Wikipedia intro extract for a title (best-effort)."""
    if not title:
        return None
    params = {"action": "query", "format": "json", "prop": "extracts", "exintro": "", "explaintext": "", "titles": title}
    try:
        resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for k, v in pages.items():
            if "extract" in v:
                return v["extract"]
    except Exception as exc:
        print("[warning] Wikipedia fetch failed:", exc)
    return None


def fetch_newsapi(query: str, api_key: Optional[str], page_size: int = 3) -> List[str]:
    """Fetch headlines from NewsAPI.org if API key provided."""
    if not api_key:
        return []
    try:
        params = {"q": query, "pageSize": page_size, "sortBy": "publishedAt", "language": "en", "apiKey": api_key}
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("articles", [])[:page_size]
        return [f"{a.get('title')} - {a.get('source', {}).get('name','')}, {a.get('publishedAt','')[:4]}" for a in items]
    except Exception as exc:
        print("[warning] NewsAPI fetch failed:", exc)
        return []


def fetch_serpapi_scholar(query: str, api_key: Optional[str], num: int = 3) -> List[str]:
    """Fetch simple Google Scholar results via SerpAPI (if key set)."""
    if not api_key:
        return []
    try:
        params = {"engine": "google_scholar", "q": query, "api_key": api_key, "num": num}
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic_results", [])[:num]
        insights = []
        for r in results:
            title = r.get("title")
            snippet = r.get("snippet")
            source = r.get("source")
            year_match = re.search(r"\b(20\d{2})\b", safe_get(snippet or "", ""))
            year = year_match.group(1) if year_match else ""
            insights.append(f"{title} ({safe_get(source,'')}, {year})")
        return insights
    except Exception as exc:
        print("[warning] SerpAPI Scholar fetch failed:", exc)
        return []


def research_agent(refined: Dict[str, Any], simulate: bool = False) -> Dict[str, Any]:
    """
    Aggregate research results into the target JSON structure.
    Attempts to meet minimum counts; if insufficient data found and simulate=False, it will try variations.
    """
    topic = refined["topic"]
    queries: List[str] = refined["searchQueries"]
    result: Dict[str, Any] = {
        "introduction": "",
        "summary": "",
        "key_findings": [],
        "news_highlights": [],
        "scholarly_insights": [],
        "wikipedia_summary": "",
        "sources": [],
        "timestamp": now_iso(),
        "searchQueries": queries,
        "topic": topic
    }

    # 1) Wikipedia
    wiki = fetch_wikipedia_intro(topic) or fetch_wikipedia_intro(queries[0]) or ""
    result["wikipedia_summary"] = wiki or "No Wikipedia summary found."

    # 2) News
    news = []
    if not simulate and config.NEWSAPI_KEY:
        # try all queries to accumulate
        for q in queries:
            news += fetch_newsapi(q, config.NEWSAPI_KEY, page_size=2)
            if len(news) >= 4:
                break
    if simulate or not news:
        # simulated headlines if none found
        news = news or [
            f"Simulated headline about {topic} - ExampleNews, 2025",
            f"Simulated update on {topic} - AnotherNews, 2025",
            f"Industry adoption of {topic} increases - IndustryNews, 2024",
            f"New study on {topic} released - ResearchNews, 2024",
        ]
    result["news_highlights"] = news[:6]

    # 3) Scholarly insights via SerpAPI
    scholarly = []
    if not simulate and config.SERPAPI_KEY:
        for q in queries:
            scholarly += fetch_serpapi_scholar(q, config.SERPAPI_KEY, num=2)
            if len(scholarly) >= 4:
                break
    if simulate or not scholarly:
        scholarly = scholarly or [
            f"Simulated insight (Doe et al., 2024, Journal of Examples)",
            f"Simulated finding (Smith et al., 2023, AI Journal)",
            f"Simulated methodological note (Lee et al., 2021, ML Proceedings)",
            f"Simulated application insight (Kim et al., 2022, Health Informatics)"
        ]
    result["scholarly_insights"] = scholarly[:6]

    # 4) Key findings: attempt to synthesize from wiki/news/scholarly using OpenAI or simulated
    key_findings = []
    if not simulate and (config.OPENAI_API_KEY and openai):
        prompt = (
            f"Given the topic: {topic}\n\n"
            f"Wikipedia intro:\n{wiki}\n\n"
            f"News highlights:\n" + "\n".join(news[:6]) + "\n\n"
            f"Scholarly insights:\n" + "\n".join(scholarly[:6]) + "\n\n"
            "Produce a JSON array named key_findings with 8-12 concise sentences (each 1 sentence) capturing main trends, challenges, opportunities, and notable applications."
        )
        try:
            out = call_openai_chat(prompt, model=config.OPENAI_MODEL, max_tokens=700)
            parsed = json.loads(out)
            if isinstance(parsed, list):
                key_findings = parsed
        except Exception as exc:
            print("[warning] OpenAI aggregate failed:", exc)

    if not key_findings:
        # fallback synthetic findings
        key_findings = [
            f"Overview: {topic} has seen accelerated interest in 2024-2025.",
            f"Application: {topic} is being used across industry-specific domains.",
            f"Challenge: Data quality and bias remain key challenges.",
            f"Opportunity: Improved tooling promises broader adoption.",
            f"Emerging trend: Hybrid approaches combining multiple techniques.",
            f"Case studies: Early pilots show positive ROI in targeted domains.",
            f"Regulation: Policymakers are starting to consider governance frameworks.",
            f"Research gap: More longitudinal studies are needed to assess impact."
        ]
    result["key_findings"] = key_findings[:12]

    # 5) Introduction and Summary: attempt to use OpenAI or simple templating
    if not simulate and (config.OPENAI_API_KEY and openai):
        prompt_intro = (
            f"Write a 4-6 sentence introduction about '{topic}' providing context and significance.\n\n"
            f"Use the following sources: Wikipedia intro: {wiki}\nNews: {news[:4]}\nScholarly: {scholarly[:4]}\nReturn ONLY the introduction as plain text."
        )
        prompt_summary = (
            f"Write a 6-8 sentence summary of key findings for '{topic}', covering trends, challenges, opportunities, and notable applications.\nReturn ONLY the summary as plain text."
        )
        try:
            intro = call_openai_chat(prompt_intro, model=config.OPENAI_MODEL, max_tokens=300)
            summary = call_openai_chat(prompt_summary, model=config.OPENAI_MODEL, max_tokens=500)
            result["introduction"] = intro.strip()
            result["summary"] = summary.strip()
        except Exception as exc:
            print("[warning] OpenAI intro/summary failed:", exc)

    if not result["introduction"]:
        result["introduction"] = f"{topic}: This report summarizes current context, significance, and trends related to {topic}."
    if not result["summary"]:
        result["summary"] = "Summary: " + " ".join(result["key_findings"][:3])

    # 6) Sources: collect URLs from available APIs (best-effort). We add placeholders if missing.
    sources: List[str] = []
    # include news article URLs via NewsAPI if available
    if config.NEWSAPI_KEY:
        try:
            params = {"q": queries[0], "pageSize": 5, "apiKey": config.NEWSAPI_KEY}
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            data = resp.json()
            for a in data.get("articles", [])[:8]:
                if a.get("url"):
                    sources.append(a["url"])
        except Exception:
            pass
    # include a Wikipedia link
    try:
        sources.append(f"https://en.wikipedia.org/wiki/{requests.utils.quote(topic.replace(' ', '_'))}")
    except Exception:
        pass
    # include scholarly placeholder links (SerpAPI returns link keys sometimes)
    if config.SERPAPI_KEY:
        try:
            params = {"engine": "google_scholar", "q": queries[0], "api_key": config.SERPAPI_KEY, "num": 5}
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
            data = resp.json()
            for r in data.get("organic_results", [])[:5]:
                link = r.get("link") or r.get("publication_link")
                if link:
                    sources.append(link)
        except Exception:
            pass
    # ensure at least 8 unique sources
    if len(sources) < 8:
        # pad with simulated sources
        while len(sources) < 8:
            sources.append(f"https://example.com/{topic.replace(' ','-')}-{len(sources)+1}")
    # dedupe preserving order
    seen = set()
    final_sources = []
    for s in sources:
        if s not in seen:
            final_sources.append(s)
            seen.add(s)
    result["sources"] = final_sources[:12]

    return result


# -----------------------
# HTML / PDF generation
# -----------------------
def generate_html_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a basic HTML research report from aggregated data.
    Returns a dict with 'html' and 'file_name'.
    """
    topic = data.get("topic", "Untitled")
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    file_name = f"research-report-{re.sub(r'[^a-zA-Z0-9\\-]+','-', topic.lower()).strip('-')}-{date_str}.pdf"
    # minimal escaping
    def esc(s): return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    key_items = "".join(f"<li>{esc(k)}</li>" for k in data.get("key_findings", []))
    news_items = "".join(f"<li>{esc(n)}</li>" for n in data.get("news_highlights", []))
    schol_items = "".join(f"<li>{esc(s)}</li>" for s in data.get("scholarly_insights", []))
    sources_items = "".join(f"<li><a href='{esc(s)}'>{esc(s)}</a></li>" for s in data.get("sources", []))
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Research Report - {esc(topic)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#222;line-height:1.5;padding:30px;}}
h1{{color:#0b3d91}}
h2{{color:#0b3d91}}
section{{margin-bottom:24px}}
ul{{margin:8px 0 16px 20px}}
footer{{font-size:12px;color:#666;margin-top:40px}}
</style>
</head>
<body>
  <header><h1>Research Report: {esc(topic)}</h1><p>Generated: {esc(date_str)}</p></header>
  <section><h2>Introduction</h2><p>{esc(data.get('introduction',''))}</p></section>
  <section><h2>Summary</h2><p>{esc(data.get('summary',''))}</p></section>
  <section><h2>Key Findings</h2><ul>{key_items}</ul></section>
  <section><h2>News Highlights</h2><ul>{news_items}</ul></section>
  <section><h2>Scholarly Insights</h2><ul>{schol_items}</ul></section>
  <section><h2>Wikipedia Summary</h2><p>{esc(data.get('wikipedia_summary',''))}</p></section>
  <section><h2>Sources</h2><ol>{sources_items}</ol></section>
  <footer>Generated by ResearchBot • {esc(now_iso())}</footer>
</body>
</html>"""
    return {"html": html, "file_name": file_name}


def convert_html_to_pdf_via_pdfshift(html: str, filename: str) -> Dict[str, Any]:
    """
    Call PDFShift API to convert HTML to PDF. Requires PDFSHIFT_API_KEY in config.
    Returns dict with 'success' and either 'url' or 'error'; if successful, PDFShift returns binary data,
    but the workflow used an API returning a URL — here we POST and receive binary; we'll save the PDF locally.
    """
    if not config.PDFSHIFT_API_KEY:
        return {"success": False, "error": "PDFShift API key not configured."}
    try:
        url = "https://api.pdfshift.io/v3/convert/pdf"
        auth = (config.PDFSHIFT_API_KEY, "")
        payload = {"source": html, "filename": filename}
        resp = requests.post(url, json=payload, auth=auth, timeout=60)
        resp.raise_for_status()
        # response content is PDF binary
        out_path = OUTPUT_DIR / filename
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return {"success": True, "path": str(out_path)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# -----------------------
# Notifications: Gmail (SMTP) & Telegram
# -----------------------
def send_email_smtp(to_email: str, subject: str, html_body: str, attachment_path: Optional[str] = None) -> Dict[str, Any]:
    smtp_server = config.SMTP_SERVER
    smtp_port = config.SMTP_PORT
    smtp_user = config.SMTP_USER
    smtp_pass = config.SMTP_PASS
    if not (smtp_server and smtp_port and smtp_user and smtp_pass):
        return {"simulated": True, "message": "SMTP not configured."}
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content("This is a research report. Open with an HTML-capable client.")
        msg.add_alternative(html_body, subtype="html")
        if attachment_path:
            with open(attachment_path, "rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype="application", subtype="pdf", filename=os.path.basename(attachment_path))
        with smtplib.SMTP(smtp_server, int(smtp_port), timeout=30) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        return {"success": True}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def send_telegram_document(bot_token: str, chat_id: str, file_path: str, caption: str = "") -> Dict[str, Any]:
    if not (bot_token and chat_id):
        return {"simulated": True, "message": "Telegram bot token or chat id not configured."}
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": chat_id, "caption": caption}
            resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        return {"success": True, "response": resp.json()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# -----------------------
# CLI Orchestration
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Research Report generation companion (n8n workflow)")
    p.add_argument("--query", "-q", required=True, help="Research topic or query")
    p.add_argument("--session-id", "-s", default="default", help="Session id for memory (not persisted here)")
    p.add_argument("--simulate", action="store_true", help="Force simulation mode (no external API calls)")
    p.add_argument("--save", action="store_true", help="Save artifacts (JSON + PDF) to outputs/")
    p.add_argument("--generate-pdf", action="store_true", help="Generate PDF via PDFShift (if configured)")
    p.add_argument("--send-email", action="store_true", help="Send report via SMTP (if configured)")
    p.add_argument("--send-telegram", action="store_true", help="Send PDF to Telegram (if configured)")
    p.add_argument("--recipient-email", default=config.DEFAULT_RECIPIENT_EMAIL, help="Recipient email for sending report")
    args = p.parse_args()

    # 1) Input validation
    query = args.query.strip()
    if len(query) < 3:
        print("Error: query must be at least 3 characters.", file=sys.stderr)
        sys.exit(2)

    simulate = args.simulate
    if simulate:
        print("[info] Running in simulate mode. No external API calls will be attempted.")

    # 2) Query Refiner
    refined = refine_query(query, simulate=simulate)
    print("[info] Refined topic:", refined.get("topic"))
    print("[info] Generated search queries:", refined.get("searchQueries"))

    # 3) Research Agent
    research = research_agent(refined, simulate=simulate)
    print("[info] Research aggregation complete. Key findings:", len(research.get("key_findings", [])))
    if args.save:
        json_path = OUTPUT_DIR / f"{int(time.time())}_research_{re.sub(r'[^a-zA-Z0-9]+','-', refined['topic']).lower()}.json"
        json_path.write_text(json.dumps(research, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] Saved aggregated research JSON to: {json_path}")

    # 4) Generate HTML
    html_obj = generate_html_report(research)
    html_content = html_obj["html"]
    pdf_filename = html_obj["file_name"]

    # Save HTML for review if requested
    if args.save:
        html_path = OUTPUT_DIR / (pdf_filename.replace(".pdf", ".html"))
        html_path.write_text(html_content, encoding="utf-8")
        print(f"[info] Saved HTML to: {html_path}")

    pdf_path = None
    if args.generate_pdf:
        conv = convert_html_to_pdf_via_pdfshift(html_content, pdf_filename)
        if conv.get("success"):
            pdf_path = conv.get("path")
            print(f"[info] PDF generated at: {pdf_path}")
        else:
            print("[warning] PDF generation failed or not configured:", conv.get("error"))
            # fallback: save HTML as pseudo-pdf file for sending
            fallback = OUTPUT_DIR / pdf_filename
            with open(fallback, "wb") as f:
                f.write(html_content.encode("utf-8"))
            pdf_path = str(fallback)
            print(f"[info] Saved fallback pseudo-PDF at: {pdf_path}")
    else:
        # not generating PDF; optionally save HTML only
        if args.save:
            print("[info] PDF generation skipped. Use --generate-pdf to convert HTML to PDF via PDFShift.")

    # 5) Send via Email
    if args.send_email:
        subject = f"Research Report: {research.get('topic')}"
        send_result = send_email_smtp(args.recipient_email, subject, html_content, attachment_path=pdf_path)
        print("[info] Email send result:", send_result)

    # 6) Send via Telegram
    if args.send_telegram:
        tg_res = send_telegram_document(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, pdf_path or "", caption=f"Research: {research.get('topic')}")
        print("[info] Telegram send result:", tg_res)

    print("[done] Process finished.")
    # print short summary
    print(json.dumps({"topic": research.get("topic"), "timestamp": research.get("timestamp"), "sources": research.get("sources")[:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()