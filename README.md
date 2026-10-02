# Relay

**Say it once. Consider it done.**

Relay is a self-hosted assistant for college students that lives in Slack. Tell it what you need in plain language. It remembers deadlines so you do not have to.

It is built around four ideas:

- **Self-hosted.** You run it yourself. Your course material and messages stay on your own setup.
- **Free-tier friendly.** It runs on free model tiers, with no per-seat pricing.
- **Model-agnostic.** The model sits behind a thin router. Today that is Qwen through OpenRouter, with Groq as fallback. Swapping models is a config change.
- **Approval-first.** Anything that touches the outside world should wait for your yes. (Planned, see below.)

Status: early prototype. The list below separates what runs today from what is planned.

## Built today

- Chat in Slack DMs, channels, or by @mention (Slack Bolt, Socket Mode, no public URL needed)
- Model router: Qwen via OpenRouter first, automatic Groq fallback
- Memory: last 20 messages per user, stored in SQLite (`relay.db`) so it survives restarts
- Reminders: "remind me in 2 hours to submit the lab report" pings you in Slack at that time (up to 7 days). Pending reminders are stored in SQLite and re-armed on restart; ones that came due while offline fire on boot
- Course PDFs: upload a PDF in Slack and Relay indexes it per user (SQLite full-text search). Ask questions about it and get answers with document and page references, or ask for a quiz with an answer key. Text PDFs just, no OCR for scans
- Deadlines: tell Relay about an assignment or exam ("DBMS assignment 3 is due Friday 5pm") and it saves it with an automatic reminder a day before. Ask "what's due this week", mark things done, or ask what to do first and get a ranked triage of your open deadlines
- Channel allowlist and optional mention-required mode

## Planned

| Feature | What it will do |
|---|---|
| Smart reminders, automatic | Pick up assignment, exam and competition dates from your LMS and inbox and set deadlines and reminders for you (today you tell Relay yourself) |
| Course-aware research, deeper | Pull course material automatically from the LMS and handle scanned PDFs (today you upload text PDFs by hand) |
| Study packs | Summaries and revision plans from course PDFs (quizzes already work) |
| Announcement digest | One daily Slack message that summarizes college emails and notices |
| Connectors | GitHub and Google (Docs, Slides, Calendar) status and actions in chat |
| Event radar | Surface hackathons, fests and competitions worth your time |
| Attendance and admin nudges | Reminders about attendance thresholds, fee dates and forms |
| Group project coordination (vision) | Each teammate runs their own Relay instance. The instances coordinate with each other: shared project memory of who did what, tasks handed between teammates' agents, nudges when someone goes quiet, repo and doc links in one pinned place |
| Approval prompts | Ask before sending, posting or changing anything outside Slack |

Group project coordination is the long-term differentiator: separately owned instances that can still cooperate. It is a design goal, not something that exists yet.

## Setup

1. Create the Slack app: at https://api.slack.com/apps choose **Create New App > From a manifest** and paste `slack-app-manifest.yml`. Install it to your workspace.
2. Copy the **Bot User OAuth Token** (`xoxb-...`, OAuth & Permissions) into `SLACK_BOT_TOKEN`.
3. Under **Basic Information > App-Level Tokens**, create a token with the `connections:write` scope and copy it (`xapp-...`) into `SLACK_APP_TOKEN`.
4. PDF reading needs the `files:read` scope, which is in the manifest. If the app was installed before, reinstall it to the workspace so Slack grants it.
5. Create an OpenRouter key at https://openrouter.ai/keys. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`. Optionally add `GROQ_API_KEY` and `GROQ_MODEL` for the fallback.
6. Run:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

Invite the bot to a channel (`/invite @Relay`) or DM it from the Apps section.

## Configuration

| Variable | Purpose |
|---|---|
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | Slack bot and Socket Mode tokens |
| `OPENROUTER_API_KEY` | Primary model key (required) |
| `OPENROUTER_MODEL` | Primary model ID (default in `.env.example`) |
| `GROQ_API_KEY`, `GROQ_MODEL` | Fallback model (optional) |
| `ALLOWED_CHANNEL_IDS` | Comma-separated channel IDs to answer in; empty means all |
| `RELAY_DB_PATH` | SQLite file for memory and reminders (default `relay.db`) |
| `REQUIRE_MENTION` | `true` = in channels, answer when @mentioned (DMs always answered) |

## Tests

```bash
pip install reportlab  # used to generate test PDFs
python test_history.py
```

## License

MIT, see [LICENSE](LICENSE).
