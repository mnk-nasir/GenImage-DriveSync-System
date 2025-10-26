```markdown
# Research Report Generator — n8n Companion

This repository contains a small Python companion that mirrors the "Automated Research Report
Generation" n8n workflow (Query refinement, web research, aggregation, PDF generation, and notifications).

Files
- main.py — main executable script (CLI)
- config.py — loads environment variables from `.env`
- requirements.txt — Python dependencies
- .env.example — example environment variables

Quickstart
1. Create a virtualenv and install deps:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file (see `.env.example`) and populate any API keys you have:
   - OPENAI_API_KEY — (optional) to refine queries and synthesize text
   - NEWSAPI_KEY — (optional) to fetch recent news
   - SERPAPI_KEY — (optional) to query Google Scholar (via SerpAPI)
   - PDFSHIFT_API_KEY — (optional) to convert HTML to PDF
   - SMTP_SERVER / SMTP_PORT / SMTP_USER / SMTP_PASS — (optional) to send email
   - TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID — (optional) to send report to Telegram

3. Run:
   ```
   python main.py --query "current trends in artificial intelligence 2025" --save --generate-pdf
   ```

Notes
- The script supports a `--simulate` mode which avoids external API calls and returns deterministic placeholders
  so you can test the pipeline offline.
- PDF generation uses PDFShift. If you do not have a PDFShift key the script will save the HTML as a fallback "pseudo-PDF".
- The script writes artifacts to `./outputs/` when `--save` is provided.

What I implemented
- Query refining (5 related search queries), research aggregation (Wikipedia, NewsAPI, SerpAPI), basic orchestration,
  HTML report generation, PDF conversion via PDFShift, SMTP email and Telegram sending helpers, and simulate mode.
- The script attempts to mirror the n8n nodes and structure you provided while keeping the code compact and runnable.

Next steps you might want
- Add JSON schema validation for the ResearchAgent output (e.g., jsonschema).
- Add Google Sheets logging (to match the workflow's metadata storage).
- Integrate with Google Drive to store PDFs and return Drive IDs.
- Add a small webhook (Flask/FastAPI) to accept HTTP triggers like the n8n chat trigger.

```