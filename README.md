# Relay

A personal AI assistant that lives in Slack. Message it in plain language and it replies, remembers the conversation, and sets reminders that ping you back.

**Status: prototype / idea stage.** This repo is a working starting point ported from an earlier Discord assistant. It is not the full design yet.

## What works today

- Chat in DMs, channels, or by @mention (Slack Bolt, Socket Mode - no public URL needed)
- Gemini as the primary model with automatic Groq fallback
- Short-term memory: last 20 messages per user, kept in process
- Reminders: "remind me in 2 hours to call Sam" schedules a Slack @mention at that time (max 7 days)
- Channel allowlist and optional mention-only mode
- Hourglass reaction while it thinks

## What it does not do yet

- Long-term memory (current memory is in-process and resets on restart; reminders are lost on restart too)
- Calendar, email or other account integrations
- Approval prompts before outside actions
- Multi-step planner/executor or multi-agent structure (this is a single model with one tool loop)
- Proactive briefings

These are the planned next steps.

## Setup

1. Create the Slack app: at https://api.slack.com/apps choose **Create New App > From a manifest** and paste `slack-app-manifest.yml`. Install it to your workspace.
2. Copy the **Bot User OAuth Token** (`xoxb-...`, OAuth & Permissions) into `SLACK_BOT_TOKEN`.
3. Under **Basic Information > App-Level Tokens**, create a token with the `connections:write` scope and copy it (`xapp-...`) into `SLACK_APP_TOKEN`.
4. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`, `GEMINI_MODEL`, and optionally `GROQ_API_KEY` / `GROQ_MODEL`. Model IDs are not hard-coded, so set ones your keys can use.
5. Run:

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
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Primary model (required) |
| `GROQ_API_KEY`, `GROQ_MODEL` | Fallback model (optional) |
| `ALLOWED_CHANNEL_IDS` | Comma-separated channel IDs to answer in; empty means all |
| `REQUIRE_MENTION` | `true` = in channels only answer when @mentioned (DMs always answered) |

## License

MIT, see [LICENSE](LICENSE).
