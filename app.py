#!/usr/bin/env python3
"""
CustomGPT Chat Server for Raspberry Pi
Local OpenAI API chat interface with full control over prompts and parameters
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import openai
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CONTEXTS_DIR = APP_DIR / "contexts"  # Directory for .txt context files
DB_PATH = DATA_DIR / "chats.db"
CONFIG_PATH = DATA_DIR / "config.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CONTEXTS_DIR.mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "api_key": "",  # Set via environment or config
    "model": "gpt-4.1",
    "system_prompt": "You are a helpful AI assistant.",
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 4096,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "context_files": []  # List of .txt files to include as context
}

# ============================================================================
# DATABASE
# ============================================================================

def init_db():
    """Initialize SQLite database for chat storage."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Chats table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    
    # Messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config():
    """Load configuration from file or create default."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def get_context_content():
    """Read all configured context files and return combined content."""
    config = load_config()
    context_parts = []
    
    for filename in config.get("context_files", []):
        filepath = CONTEXTS_DIR / filename
        if filepath.exists() and filepath.suffix == '.txt':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        context_parts.append(f"=== {filename} ===\n{content}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return "\n\n".join(context_parts)

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize database on startup
init_db()

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main chat interface."""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# ----- Configuration -----

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config = load_config()
    # Don't expose full API key
    if config.get('api_key'):
        config['api_key_set'] = True
        config['api_key'] = '***' + config['api_key'][-4:] if len(config['api_key']) > 4 else '****'
    else:
        config['api_key_set'] = False
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration."""
    data = request.json
    config = load_config()
    
    # Update allowed fields
    allowed_fields = ['model', 'system_prompt', 'temperature', 'top_p', 
                      'max_tokens', 'presence_penalty', 'frequency_penalty', 'context_files']
    
    for field in allowed_fields:
        if field in data:
            config[field] = data[field]
    
    # Handle API key separately (only update if provided and not masked)
    if 'api_key' in data and not data['api_key'].startswith('***'):
        config['api_key'] = data['api_key']
    
    save_config(config)
    return jsonify({"status": "ok"})

@app.route('/api/context-files', methods=['GET'])
def list_context_files():
    """List available .txt files in contexts directory."""
    files = []
    for f in CONTEXTS_DIR.glob('*.txt'):
        stat = f.stat()
        files.append({
            "name": f.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    return jsonify(files)

@app.route('/api/context-files/<filename>', methods=['GET'])
def get_context_file(filename):
    """Get content of a context file."""
    filepath = CONTEXTS_DIR / filename
    if not filepath.exists() or filepath.suffix != '.txt':
        return jsonify({"error": "File not found"}), 404
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({"name": filename, "content": content})

@app.route('/api/context-files/<filename>', methods=['PUT'])
def save_context_file(filename):
    """Save/update a context file."""
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    filepath = CONTEXTS_DIR / filename
    content = request.json.get('content', '')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return jsonify({"status": "ok", "name": filename})

@app.route('/api/context-files/<filename>', methods=['DELETE'])
def delete_context_file(filename):
    """Delete a context file."""
    filepath = CONTEXTS_DIR / filename
    if filepath.exists():
        filepath.unlink()
    return jsonify({"status": "ok"})

# ----- Chats -----

@app.route('/api/chats', methods=['GET'])
def list_chats():
    """List all chats."""
    conn = get_db()
    chats = conn.execute(
        'SELECT * FROM chats ORDER BY updated_at DESC'
    ).fetchall()
    conn.close()
    return jsonify([dict(chat) for chat in chats])

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat."""
    chat_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    title = request.json.get('title', 'New Chat')
    
    conn = get_db()
    conn.execute(
        'INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
        (chat_id, title, now, now)
    )
    conn.commit()
    conn.close()
    
    return jsonify({"id": chat_id, "title": title, "created_at": now, "updated_at": now})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get chat with messages."""
    conn = get_db()
    chat = conn.execute('SELECT * FROM chats WHERE id = ?', (chat_id,)).fetchone()
    if not chat:
        conn.close()
        return jsonify({"error": "Chat not found"}), 404
    
    messages = conn.execute(
        'SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id',
        (chat_id,)
    ).fetchall()
    conn.close()
    
    return jsonify({
        "chat": dict(chat),
        "messages": [dict(msg) for msg in messages]
    })

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat and its messages."""
    conn = get_db()
    conn.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
    conn.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/api/chats/<chat_id>/delete-last', methods=['POST'])
def delete_last_messages(chat_id):
    """Delete last N messages (for edit/regenerate)."""
    count = request.json.get('count', 1)  # How many messages to delete
    
    conn = get_db()
    
    # Get IDs of last N messages
    last_msgs = conn.execute(
        'SELECT id FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?',
        (chat_id, count)
    ).fetchall()
    
    if last_msgs:
        ids = [msg['id'] for msg in last_msgs]
        placeholders = ','.join('?' * len(ids))
        conn.execute(f'DELETE FROM messages WHERE id IN ({placeholders})', ids)
        conn.commit()
    
    conn.close()
    return jsonify({"status": "ok", "deleted": len(last_msgs)})

@app.route('/api/chats/<chat_id>/title', methods=['PUT'])
def update_chat_title(chat_id):
    """Update chat title."""
    title = request.json.get('title', 'Untitled')
    now = datetime.now().isoformat()
    
    conn = get_db()
    conn.execute(
        'UPDATE chats SET title = ?, updated_at = ? WHERE id = ?',
        (title, now, chat_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

# ----- Chat Completion -----

@app.route('/api/chats/<chat_id>/send', methods=['POST'])
def send_message(chat_id):
    """Send message and get AI response (streaming)."""
    config = load_config()
    
    if not config.get('api_key'):
        return jsonify({"error": "API key not configured"}), 400
    
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Get chat history
    conn = get_db()
    chat = conn.execute('SELECT * FROM chats WHERE id = ?', (chat_id,)).fetchone()
    if not chat:
        conn.close()
        return jsonify({"error": "Chat not found"}), 404
    
    history = conn.execute(
        'SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id',
        (chat_id,)
    ).fetchall()
    
    # Sliding window: keep only last N messages to control costs and prevent context overflow
    MAX_HISTORY_MESSAGES = 50
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    
    # Save user message
    now = datetime.now().isoformat()
    conn.execute(
        'INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)',
        (chat_id, 'user', user_message, now)
    )
    conn.execute('UPDATE chats SET updated_at = ? WHERE id = ?', (now, chat_id))
    conn.commit()
    conn.close()
    
    # Build messages for API
    messages = []
    
    # System prompt with context
    system_content = config.get('system_prompt', '')
    context = get_context_content()
    if context:
        system_content += f"\n\n--- CONTEXT ---\n{context}"
    
    if system_content:
        messages.append({"role": "system", "content": system_content})
    
    # Add history
    for msg in history:
        messages.append({"role": msg['role'], "content": msg['content']})
    
    # Add new user message
    messages.append({"role": "user", "content": user_message})
    
    # Stream response
    def generate():
        try:
            client = OpenAI(api_key=config['api_key'])
            
            response = client.chat.completions.create(
                model=config.get('model', 'gpt-4.1'),
                messages=messages,
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 1.0),
                max_tokens=config.get('max_tokens', 4096),
                presence_penalty=config.get('presence_penalty', 0.0),
                frequency_penalty=config.get('frequency_penalty', 0.0),
                stream=True
            )
            
            full_response = ""
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Save assistant response
            conn = get_db()
            now = datetime.now().isoformat()
            conn.execute(
                'INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)',
                (chat_id, 'assistant', full_response, now)
            )
            conn.commit()
            conn.close()
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# ----- Auto-title -----

@app.route('/api/chats/<chat_id>/auto-title', methods=['POST'])
def auto_title(chat_id):
    """Generate title based on first message."""
    config = load_config()
    
    if not config.get('api_key'):
        return jsonify({"error": "API key not configured"}), 400
    
    conn = get_db()
    first_msg = conn.execute(
        'SELECT content FROM messages WHERE chat_id = ? AND role = "user" ORDER BY id LIMIT 1',
        (chat_id,)
    ).fetchone()
    conn.close()
    
    if not first_msg:
        return jsonify({"title": "New Chat"})
    
    try:
        client = OpenAI(api_key=config['api_key'])
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Use cheaper model for titles
            messages=[
                {"role": "system", "content": "Generate a short title (3-5 words) for this chat based on the first message. Reply with just the title, no quotes."},
                {"role": "user", "content": first_msg['content'][:500]}
            ],
            max_tokens=20,
            temperature=0.5
        )
        title = response.choices[0].message.content.strip()[:50]
        
        # Update title
        conn = get_db()
        conn.execute('UPDATE chats SET title = ? WHERE id = ?', (title, chat_id))
        conn.commit()
        conn.close()
        
        return jsonify({"title": title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CustomGPT Chat Server")
    print("="*60)
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Context files:  {CONTEXTS_DIR}")
    print(f"  Database:       {DB_PATH}")
    print("="*60)
    print("  Starting server on http://10.0.0.75:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
