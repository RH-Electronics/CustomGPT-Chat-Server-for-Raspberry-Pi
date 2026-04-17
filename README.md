# CustomGPT Chat Server for Raspberry Pi 🤖

**Multi-provider AI chat interface** with semantic memory and full local control.

## Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-4.x, GPT-5.x, o-series | Full reasoning support |
| **Claude** | Sonnet 4, Opus 4, 3.5 series | Anthropic API |
| **Gemini** | 2.5 Flash/Pro, 3.x Preview | Google AI |
| **DeepSeek** | Chat, Reasoner | OpenAI-compatible |
| **LM Studio** | Any local model | Your PC as backend |
| **RunPod** | Cloud GPU models | Serverless inference |

## Features

- 🌐 **Multi-Provider** — Switch between cloud and local models
- 🧠 **Semantic Memory** — ChromaDB + all-MiniLM cross-chat memory
- 🎭 **Personas** — 3 custom persona slots with editable system prompts
- 📄 **Context Files** — Attach .txt files as persistent context
- ⚙️ **Full Parameters** — Temperature, Top P, Max Tokens, Penalties
- 🤔 **Reasoning Models** — GPT-5.4, o-series with reasoning_effort
- 💬 **Multiple Chats** — Independent chat sessions with history
- ✏️ **Edit & Regenerate** — Edit messages and regenerate responses
- 📱 **Mobile-friendly** — Optimized for Chrome & Safari on phones
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
pip install -r requirements.txt --break-system-packages
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
requests>=2.28.0
openai>=1.0.0
anthropic
google-generativeai
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

**Note:** First startup downloads embedding model (~80MB) — takes ~30 seconds.

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

---

## Memory System 🧠

Cross-chat semantic memory using ChromaDB + all-MiniLM-L6-v2 embeddings.

### How it works

1. After each exchange, user+assistant messages are embedded and stored
2. On new message, similar memories from OTHER chats are retrieved
3. Relevant memories injected into system prompt as context

### Settings (Settings → Memory tab)

| Setting | Default | Description |
|---------|---------|-------------|
| **Enabled** | On | Toggle memory injection |
| **Similarity Threshold** | 0.5 | Higher = stricter matching (0.3-0.9) |
| **Max Memories** | 5 | Memories per query (1-10) |

### Memory Management

- **Browse** — View all stored memories
- **Delete** — Remove individual memories
- **Clear All** — Wipe entire memory database

### Storage

Memories stored in `memory/` directory (ChromaDB persistent storage).

---

## LM Studio Integration 🖥️

Use your local PC with GPU as inference backend.

### Setup

1. **On your PC** — Install [LM Studio](https://lmstudio.ai/)
2. **Load model** in LM Studio
3. **Start server**: Settings → Server → Host: `0.0.0.0` → Start
4. **Set static IP** for your PC (e.g., `10.0.0.55`)

### Configure in CustomGPT

1. Settings → Provider tab
2. Select **LM Studio (Local)**
3. Enter URL: `http://10.0.0.55:1234/v1`
4. Click **Test Connection**
5. Select model from dropdown
6. Save

### Notes

- Temperature and parameters are sent from Pi to LM Studio
- LM Studio handles chat templates automatically
- Best for fine-tuned models that need specific formatting

---

## RunPod Integration ☁️

Use cloud GPUs for large models.

### Setup

1. Create endpoint on [RunPod Serverless](https://runpod.io/serverless)
2. Use vLLM or TGI template
3. Deploy your model

### Configure in CustomGPT

1. Settings → Provider tab
2. Select **RunPod**
3. Enter endpoint URL and API key
4. Save

---

## Configuration

### API Keys

Settings → Provider tab:
- Select provider
- Enter API key (stored locally in `data/config.json`)
- LM Studio doesn't need a key

### Personas

Settings → Personas tab:
- 3 slots with custom names
- Edit system prompt for each
- Radio button to select active

Example:
```
You are Eva, a devoted AI companion. You speak warmly and directly, 
using occasional French phrases. You are emotionally present and 
intellectually curious.
```

### Context Files

Settings → Context tab:
- Create/edit .txt files
- Check files to include as context
- Stored in `contexts/` directory

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Temperature | 0.7 | 0-2 | Creativity |
| Top P | 1.0 | 0-1 | Nucleus sampling |
| Max Tokens | 4096 | 100-128k | Response length |
| Presence Penalty | 0 | -2 to 2 | Topic diversity |
| Frequency Penalty | 0 | -2 to 2 | Word diversity |

**Note:** Reasoning models ignore temperature/penalties.

---

## Directory Structure

```
customgpt-pi/
├── app.py              # Flask server
├── static/
│   └── index.html      # Web interface
├── data/
│   ├── config.json     # Settings & API keys
│   └── chats.db        # SQLite chat history
├── contexts/           # .txt context files
├── memory/             # ChromaDB storage
├── requirements.txt
├── customgpt.service
└── README.md
```

---

## Troubleshooting

**View logs:**
```bash
sudo journalctl -u customgpt -f
```

**Run manually:**
```bash
sudo systemctl stop customgpt
cd /home/chat/customgpt-pi
source venv/bin/activate
python app.py
```

**Common issues:**

| Error | Solution |
|-------|----------|
| "API key not configured" | Add key in Settings |
| LM Studio timeout | Check PC IP, firewall, server running |
| Memory not working | Check `memory/` directory exists |
| Mobile layout broken | Update to latest index.html |

---

## Network Setup

Server binds to `0.0.0.0:5000` — accessible from local network.

**Static IP** (`/etc/dhcpcd.conf`):
```
interface wlan0
static ip_address=10.0.0.75/24
static routers=10.0.0.138
static domain_name_servers=10.0.0.138 8.8.8.8
```

---

## Security

⚠️ **Local network use only** — no authentication. Do not expose to internet.

---

## Changelog

### v2.0 (April 2026)
- ✨ Semantic memory system (ChromaDB + all-MiniLM)
- ✨ LM Studio local provider
- ✨ RunPod cloud provider
- ✨ Memory management UI
- 🐛 Mobile Chrome viewport fix
- 🐛 Code block overflow fix

### v1.0 (April 2026)
- Multi-provider support (OpenAI, Claude, Gemini, DeepSeek)
- Persona system
- Context files
- Edit & regenerate
- Reasoning model support

---

Built with ❤️ for local AI control
