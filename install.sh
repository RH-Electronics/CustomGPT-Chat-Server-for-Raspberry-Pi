#!/bin/bash
# ============================================================================
# CustomGPT Installation Script for Raspberry Pi
# ============================================================================

set -e

echo "=============================================="
echo "  CustomGPT Chat Server - Installation"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running on Pi
if [ ! -f /etc/rpi-issue ] && [ ! -d /opt/vc ]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    echo "Continuing anyway..."
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}[1/5]${NC} Updating system packages..."
sudo apt-get update

echo -e "\n${GREEN}[2/5]${NC} Installing Python dependencies..."
sudo apt-get install -y python3-venv python3-pip

echo -e "\n${GREEN}[3/5]${NC} Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo -e "\n${GREEN}[4/5]${NC} Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n${GREEN}[5/5]${NC} Setting up directories..."
mkdir -p data contexts

# Create default config
if [ ! -f data/config.json ]; then
    cat > data/config.json << 'EOF'
{
  "api_key": "",
  "model": "gpt-4.1",
  "system_prompt": "You are a helpful AI assistant.",
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 4096,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "context_files": []
}
EOF
    echo -e "${GREEN}Created default config file${NC}"
fi

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "=============================================="
echo -e "  ${GREEN}Installation Complete!${NC}"
echo "=============================================="
echo ""
echo "To start the server manually:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Access the chat interface at:"
echo -e "  ${GREEN}http://${LOCAL_IP}:5000${NC}"
echo ""
echo "=============================================="
echo "  Optional: Set up auto-start on boot"
echo "=============================================="
echo ""
echo "Run these commands:"
echo "  sudo cp customgpt.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable customgpt"
echo "  sudo systemctl start customgpt"
echo ""
echo "Then check status with:"
echo "  sudo systemctl status customgpt"
echo ""
