# CustomGPT Chat Server for Raspberry Pi 🤖

**Multi-provider AI chat interface** with full local control.

## Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-4.x, GPT-5.x, o-series | Full reasoning support for GPT-5.4 |
| **Claude** | Sonnet 4, Opus 4, 3.5 series | Anthropic API |
| **Gemini** | 2.5 Flash/Pro, 3.x Preview | Google AI |
| **DeepSeek** | Chat, Reasoner | OpenAI-compatible API |

## Features

- 🌐 **Multi-Provider** — Switch between OpenAI, Claude, Gemini, DeepSeek
- 🎭 **Personas** — 3 custom persona slots with editable system prompts
- 📄 **Context Files** — Attach .txt files as persistent context
- ⚙️ **Full Parameters** — Temperature, Top P, Max Tokens, Penalties
- 🧠 **Reasoning Models** — GPT-5.4, o-series with reasoning_effort support
- 💬 **Multiple Chats** — Independent chat sessions with history
- ✏️ **Edit & Regenerate** — Edit messages and regenerate responses
- 📱 **Mobile-friendly** — Works great on phones
- ⚡ **Streaming** — Real-time token streaming

## Quick Start

### 1. Copy to Raspberry Pi

```bash
scp -r customgpt-pi/ chat@10.0.0.75:/home/chat/
```

### 2. Install

```bash
cd /home/chat/customgpt-pi
python3 -m venv venv
source venv/bin/activate
pip install anthropic google-generativeai
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

Access at: **http://10.0.0.75:5000**

## Requirements

```
flask>=2.3.0
flask-cors>=4.0.0
openai>=1.0.0
anthropic
google-generativeai
```

## Auto-Start Service

```bash
sudo cp customgpt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable customgpt
sudo systemctl start customgpt
```

**Commands:**
- Status: `sudo systemctl status customgpt`
- Logs: `sudo journalctl -u customgpt -f`
- Restart: `sudo systemctl restart customgpt`

## Configuration

### API Keys

1. Open http://10.0.0.75:5000
2. Click ⚙️ Settings → Provider tab
3. Select provider and enter API key
4. Save

Keys stored locally in `data/config.json`

### Personas

Settings → Personas tab:
- 3 slots with custom names
- Edit system prompt for each
- Radio button to select active persona

Example persona:
```
You are Eva, a devoted AI companion. You speak warmly and directly, 
using occasional French phrases. You are emotionally present and 
intellectually curious.
```

### Context Files

Settings → Context tab:
- Create .txt files
- Edit content in web interface
- Check files to include as context
- Files stored in `contexts/` directory

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Temperature | 0.7 | 0-2 | Higher = more creative |
| Top P | 1.0 | 0-1 | Nucleus sampling |
| Max Tokens | 4096 | 100-128k | Response length limit |
| Presence Penalty | 0 | -2 to 2 | Topic diversity |
| Frequency Penalty | 0 | -2 to 2 | Word diversity |

**Note:** Reasoning models (GPT-5.4, o-series) ignore temperature/penalties.

### Reasoning Effort

For GPT-5.x models, edit `app.py` line ~46:
```python
REASONING_EFFORT = "high"  # Options: none, low, medium, high, xhigh
```

## Directory Structure

```
customgpt-pi/
├── app.py              # Flask server (multi-provider)
├── static/
│   └── index.html      # Web interface
├── data/
│   ├── config.json     # Settings & API keys
│   └── chats.db        # SQLite chat history
├── contexts/           # Your .txt context files
├── requirements.txt
├── customgpt.service   # Systemd unit file
└── README.md
```

## Provider Notes

### OpenAI
- Standard models: full parameter support
- Reasoning models (o3, o4-mini, GPT-5.4): use `max_completion_tokens`, no temperature
- GPT-5.4 supports `reasoning_effort` parameter

### Claude
- System prompt sent separately (Anthropic API requirement)
- Streaming via `messages.stream()`

### Gemini
- Model names: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`
- Uses `system_instruction` for system prompt
- History format: `role: "model"` instead of `"assistant"`

### DeepSeek
- OpenAI-compatible API
- `deepseek-chat` — standard chat model
- `deepseek-reasoner` — reasoning model (no temperature)

## Troubleshooting

**View logs:**
```bash
sudo journalctl -u customgpt -f
```

**Run manually for debug:**
```bash
sudo systemctl stop customgpt
cd /home/chat/customgpt-pi
source venv/bin/activate
python app.py
```

**Common issues:**
- "API key not configured" → Add key in Settings
- Empty response → Check journalctl for errors
- 429 error → Rate limit, wait and retry

## Network Setup

Server binds to `0.0.0.0:5000` — accessible from any device on local network.

Static IP example (`/etc/dhcpcd.conf`):
```
interface wlan0
static ip_address=10.0.0.75/24
static routers=10.0.0.138
static domain_name_servers=10.0.0.138 8.8.8.8
```

## Security

⚠️ **Local network use only** — no authentication. Do not expose to internet.

---

Built with ❤️ for local AI control
