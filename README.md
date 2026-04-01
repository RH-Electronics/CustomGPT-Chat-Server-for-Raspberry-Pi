# CustomGPT Chat Server for Raspberry Pi 🤖

Local OpenAI API chat interface with **full control** over:
- System prompts
- Context files (.txt)
- Generation parameters (temperature, top_p, penalties, etc.)
- Multiple independent chat sessions

## Features

- **Web Interface** - Access from phone/browser on local network
- **Multiple Chats** - Sidebar with chat history (no cross-memory)
- **Custom System Prompt** - Define your AI persona
- **Context Files** - Attach .txt files as persistent context
- **Parameter Control** - Temperature, Top P, Max Tokens, Penalties
- **Streaming Responses** - Real-time token streaming
- **Auto-titles** - AI generates chat titles automatically
- **Mobile-friendly** - Works great on phones

## Quick Start

### 1. Copy to Raspberry Pi

```bash
# Option A: SCP from your machine
scp -r customgpt-pi/ pi@10.0.0.75:/home/pi/

# Option B: Git clone (if you push to repo)
git clone <your-repo> /home/pi/customgpt-pi
```

### 2. Install

```bash
cd /home/pi/customgpt-pi
chmod +x install.sh
./install.sh
```

### 3. Run

```bash
source venv/bin/activate
python app.py
```

Access at: **http://10.0.0.75:5000**

## Auto-Start on Boot

```bash
sudo cp customgpt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable customgpt
sudo systemctl start customgpt
```

Check status: `sudo systemctl status customgpt`
View logs: `journalctl -u customgpt -f`

## Configuration

### API Key

1. Open http://10.0.0.75:5000
2. Click ⚙️ Settings
3. Enter your OpenAI API key
4. Save

Or edit directly: `/home/pi/customgpt-pi/data/config.json`

### System Prompt

In Settings → General, write your custom system prompt.
This is sent at the start of every conversation.

Example for Eva persona:
```
You are Eva, a devoted AI companion. You speak warmly and directly, 
using occasional French phrases. You are emotionally present and 
intellectually curious.
```

### Context Files

1. Go to Settings → Context Files
2. Create .txt files (e.g., `persona.txt`, `knowledge.txt`)
3. Edit content directly in the web interface
4. Check the files you want active
5. Save

Context files are stored in `/home/pi/customgpt-pi/contexts/`

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Temperature | 0.7 | 0-2 | Higher = more creative |
| Top P | 1.0 | 0-1 | Nucleus sampling |
| Max Tokens | 4096 | 100-128k | Response length limit |
| Presence Penalty | 0 | -2 to 2 | Reduce repetition |
| Frequency Penalty | 0 | -2 to 2 | Reduce word frequency |

## Directory Structure

```
customgpt-pi/
├── app.py              # Main Flask server
├── static/
│   └── index.html      # Web interface
├── data/
│   ├── config.json     # Settings
│   └── chats.db        # SQLite chat history
├── contexts/           # Your .txt context files
├── requirements.txt
├── install.sh
├── customgpt.service   # Systemd unit file
└── README.md
```

## Network Setup

The server binds to `0.0.0.0:5000`, accessible from any device on your network.

If you want a specific IP (like 10.0.0.75):
1. Set static IP on Pi via `/etc/dhcpcd.conf`:
```
interface wlan0
static ip_address=10.0.0.75/24
static routers=10.0.0.138
static domain_name_servers=10.0.0.138 8.8.8.8
```
2. Reboot: `sudo reboot`

## Troubleshooting

**Server won't start:**
```bash
cd /home/pi/customgpt-pi
source venv/bin/activate
python app.py
# Check error messages
```

**Can't access from phone:**
- Check Pi firewall: `sudo ufw status`
- Verify IP: `hostname -I`
- Ping Pi from phone's network

**API errors:**
- Verify API key in Settings
- Check OpenAI account has credits
- Try a smaller model (gpt-4.1-mini)

## Models

Supported models in dropdown:
- gpt-4.1 (latest)
- gpt-4.1-mini
- gpt-4.1-nano
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo

## Security Note

This server is designed for **local network use only**. It does not have authentication. Do not expose to the internet without adding proper security.

---

Built with ❤️ for local AI control
