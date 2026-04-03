#!/usr/bin/env python3
"""
CustomGPT Chat Server for Raspberry Pi
Multi-provider AI chat interface with personas and full parameter control
Supports: OpenAI (including GPT-5.x reasoning), Claude, Gemini, DeepSeek

Reasoning models (GPT-5.x, o-series, deepseek-reasoner) use different API parameters.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CONTEXTS_DIR = APP_DIR / "contexts"
DB_PATH = DATA_DIR / "chats.db"
CONFIG_PATH = DATA_DIR / "config.json"

DATA_DIR.mkdir(exist_ok=True)
CONTEXTS_DIR.mkdir(exist_ok=True)

# Models that are "reasoning" models and don't support temperature/penalties
# These use max_completion_tokens instead of max_tokens
# GPT-5.x support reasoning.effort parameter
REASONING_MODELS = {
    # OpenAI o-series (older reasoning)
    'o1', 'o1-mini', 'o1-preview',
    'o3', 'o3-mini', 'o4-mini',
    # GPT-5.x series (all are reasoning models)
    'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
    'gpt-5.1', 'gpt-5.2', 'gpt-5.3', 'gpt-5.4',
    # DeepSeek reasoner
    'deepseek-reasoner',
}

# Reasoning effort level for GPT-5.x models
# Options: none, low, medium, high, xhigh
# Change this to adjust reasoning depth
REASONING_EFFORT = "medium"  # <-- CHANGE THIS TO: none, low, medium, high, xhigh

def is_reasoning_model(model_name):
    """Check if model is a reasoning model that needs special handling."""
    model_lower = model_name.lower()
    
    # GPT-5.x-chat-latest are Instant models - NOT reasoning!
    # They work like regular models with temperature support
    if 'chat-latest' in model_lower:
        return False
    
    # Check exact matches and prefixes
    for rm in REASONING_MODELS:
        if model_lower == rm or model_lower.startswith(rm + '-') or model_lower.startswith(rm + '.'):
            return True
    # Also catch gpt-5.x patterns (but not chat-latest which we excluded above)
    if 'gpt-5' in model_lower:
        return True
    return False

def is_gpt5_model(model_name):
    """Check if model is GPT-5.x reasoning model (supports reasoning_effort parameter)."""
    model_lower = model_name.lower()
    # chat-latest are Instant models, NOT reasoning
    if 'chat-latest' in model_lower:
        return False
    return 'gpt-5' in model_lower

# Provider configurations
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": [
            # Standard models
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
            "gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
            # Reasoning models (o-series)
            "o4-mini", "o3", "o3-mini",
            # GPT-5.x reasoning models
            "gpt-5.4-2026-03-05", "gpt-5.4-mini-2026-03-17"
        ]
    },
    "claude": {
        "name": "Claude",
        "base_url": "https://api.anthropic.com",
        "models": [
            "claude-sonnet-4-20250514", "claude-opus-4-20250514", 
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"
        ]
    },
    "gemini": {
        "name": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": [
            "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"
        ]
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"]
    }
}

DEFAULT_PERSONAS = [
    {"name": "Assistant", "prompt": "You are a helpful AI assistant.", "active": True},
    {"name": "Persona 2", "prompt": "", "active": False},
    {"name": "Persona 3", "prompt": "", "active": False}
]

DEFAULT_CONFIG = {
    "provider": "openai",
    "api_keys": {
        "openai": "",
        "claude": "",
        "gemini": "",
        "deepseek": ""
    },
    "model": "gpt-4.1",
    "personas": DEFAULT_PERSONAS,
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 4096,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "context_files": [],
    "reasoning_effort": "high"  # For GPT-5.x: none, low, medium, high, xhigh
}

# ============================================================================
# DATABASE
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            if 'api_keys' not in config:
                config['api_keys'] = DEFAULT_CONFIG['api_keys'].copy()
                if config.get('api_key'):
                    config['api_keys']['openai'] = config['api_key']
            return config
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def get_active_persona(config):
    personas = config.get('personas', DEFAULT_PERSONAS)
    for p in personas:
        if p.get('active'):
            return p.get('prompt', '')
    return personas[0].get('prompt', '') if personas else ''

def get_context_content():
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
# AI PROVIDER CLIENTS
# ============================================================================

def call_openai_stream(config, messages):
    """
    OpenAI and DeepSeek API with full support for:
    - Standard models (GPT-4.x, etc): temperature, top_p, penalties
    - Reasoning models (o-series): max_completion_tokens only
    - GPT-5.x models: max_completion_tokens + reasoning.effort
    """
    from openai import OpenAI
    
    provider = config.get('provider', 'openai')
    api_key = config['api_keys'].get(provider, '')
    model = config.get('model', 'gpt-4.1')
    
    if provider == 'deepseek':
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    else:
        client = OpenAI(api_key=api_key)
    
    # Check if this is a reasoning model
    if is_reasoning_model(model):
        # Reasoning models use max_completion_tokens instead of max_tokens
        
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": config.get('max_tokens', 4096),
            "stream": True
        }
        
        # GPT-5.x models support reasoning_effort parameter in Chat Completions API
        # Format: "reasoning_effort": "none" | "low" | "medium" | "high" | "xhigh"
        # NOTE: temperature and other params are ONLY supported when reasoning_effort="none"
        if is_gpt5_model(model):
            effort = config.get('reasoning_effort', REASONING_EFFORT)
            params["reasoning_effort"] = effort  # Chat Completions uses this format!
            print(f"[GPT-5.x] Using reasoning_effort: {effort}")
        
        print(f"[Reasoning Model] {model} - no temperature/penalties")
        
    else:
        # Standard models - full parameter support
        params = {
            "model": model,
            "messages": messages,
            "temperature": config.get('temperature', 0.7),
            "top_p": config.get('top_p', 1.0),
            "max_tokens": config.get('max_tokens', 4096),
            "presence_penalty": config.get('presence_penalty', 0.0),
            "frequency_penalty": config.get('frequency_penalty', 0.0),
            "stream": True
        }
    
    try:
        response = client.chat.completions.create(**params)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"[OpenAI Error] {type(e).__name__}: {e}")
        raise

def call_claude_stream(config, messages):
    """Anthropic Claude API"""
    import anthropic
    
    api_key = config['api_keys'].get('claude', '')
    client = anthropic.Anthropic(api_key=api_key)
    
    system_content = ""
    chat_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_content = msg['content']
        else:
            chat_messages.append(msg)
    
    try:
        with client.messages.stream(
            model=config.get('model', 'claude-sonnet-4-20250514'),
            max_tokens=config.get('max_tokens', 4096),
            system=system_content,
            messages=chat_messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        print(f"[Claude Error] {type(e).__name__}: {e}")
        raise

def call_gemini_stream(config, messages):
    """Google Gemini API"""
    import google.generativeai as genai
    
    api_key = config['api_keys'].get('gemini', '')
    genai.configure(api_key=api_key)
    
    model_name = config.get('model', 'gemini-2.0-flash')
    
    # Convert messages to Gemini format
    system_instruction = ""
    history = []
    current_message = ""
    
    for msg in messages:
        if msg['role'] == 'system':
            system_instruction = msg['content']
        elif msg['role'] == 'user':
            if current_message:
                history.append({"role": "user", "parts": [current_message]})
                current_message = ""
            current_message = msg['content']
        elif msg['role'] == 'assistant':
            if current_message:
                history.append({"role": "user", "parts": [current_message]})
                current_message = ""
            history.append({"role": "model", "parts": [msg['content']]})
    
    generation_config = {
        "temperature": config.get('temperature', 0.7),
        "top_p": config.get('top_p', 1.0),
        "max_output_tokens": config.get('max_tokens', 4096),
    }
    
    try:
        if system_instruction:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=generation_config
            )
        else:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
        
        chat = model.start_chat(history=history)
        response = chat.send_message(current_message, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        print(f"[Gemini Error] {type(e).__name__}: {e}")
        raise

def call_provider_stream(config, messages):
    """Route to appropriate provider"""
    provider = config.get('provider', 'openai')
    
    if provider in ['openai', 'deepseek']:
        yield from call_openai_stream(config, messages)
    elif provider == 'claude':
        yield from call_claude_stream(config, messages)
    elif provider == 'gemini':
        yield from call_gemini_stream(config, messages)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder='static')
CORS(app)
init_db()

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# ----- Providers Info -----

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get available providers and their models."""
    return jsonify(PROVIDERS)

# ----- Configuration -----

@app.route('/api/config', methods=['GET'])
def get_config():
    config = load_config()
    # Mask API keys
    masked_keys = {}
    for provider, key in config.get('api_keys', {}).items():
        if key:
            masked_keys[provider] = '***' + key[-4:] if len(key) > 4 else '****'
        else:
            masked_keys[provider] = ''
    config['api_keys_masked'] = masked_keys
    config['api_keys'] = {k: '' for k in config.get('api_keys', {})}
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    config = load_config()
    
    allowed_fields = ['provider', 'model', 'temperature', 'top_p', 
                      'max_tokens', 'presence_penalty', 'frequency_penalty', 
                      'context_files', 'personas', 'reasoning_effort']
    
    for field in allowed_fields:
        if field in data:
            config[field] = data[field]
    
    # Handle API keys separately
    if 'api_keys' in data:
        for provider, key in data['api_keys'].items():
            if key and not key.startswith('***'):
                config['api_keys'][provider] = key
    
    save_config(config)
    return jsonify({"status": "ok"})

# ----- Context Files -----

@app.route('/api/context-files', methods=['GET'])
def list_context_files():
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
    filepath = CONTEXTS_DIR / filename
    if not filepath.exists() or filepath.suffix != '.txt':
        return jsonify({"error": "File not found"}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({"name": filename, "content": content})

@app.route('/api/context-files/<filename>', methods=['PUT'])
def save_context_file(filename):
    if not filename.endswith('.txt'):
        filename += '.txt'
    filepath = CONTEXTS_DIR / filename
    content = request.json.get('content', '')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return jsonify({"status": "ok", "name": filename})

@app.route('/api/context-files/<filename>', methods=['DELETE'])
def delete_context_file(filename):
    filepath = CONTEXTS_DIR / filename
    if filepath.exists():
        filepath.unlink()
    return jsonify({"status": "ok"})

# ----- Chats -----

@app.route('/api/chats', methods=['GET'])
def list_chats():
    conn = get_db()
    chats = conn.execute('SELECT * FROM chats ORDER BY updated_at DESC').fetchall()
    conn.close()
    return jsonify([dict(chat) for chat in chats])

@app.route('/api/chats', methods=['POST'])
def create_chat():
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
    return jsonify({"chat": dict(chat), "messages": [dict(msg) for msg in messages]})

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    conn = get_db()
    conn.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
    conn.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/api/chats/<chat_id>/delete-last', methods=['POST'])
def delete_last_messages(chat_id):
    count = request.json.get('count', 1)
    conn = get_db()
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
    title = request.json.get('title', 'Untitled')
    now = datetime.now().isoformat()
    conn = get_db()
    conn.execute('UPDATE chats SET title = ?, updated_at = ? WHERE id = ?', (title, now, chat_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

# ----- Chat Completion -----

@app.route('/api/chats/<chat_id>/send', methods=['POST'])
def send_message(chat_id):
    config = load_config()
    provider = config.get('provider', 'openai')
    api_key = config.get('api_keys', {}).get(provider, '')
    
    if not api_key:
        return jsonify({"error": f"API key not configured for {provider}"}), 400
    
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    conn = get_db()
    chat = conn.execute('SELECT * FROM chats WHERE id = ?', (chat_id,)).fetchone()
    if not chat:
        conn.close()
        return jsonify({"error": "Chat not found"}), 404
    
    history = conn.execute(
        'SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id',
        (chat_id,)
    ).fetchall()
    
    # Sliding window
    MAX_HISTORY_MESSAGES = 50
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    
    now = datetime.now().isoformat()
    conn.execute(
        'INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)',
        (chat_id, 'user', user_message, now)
    )
    conn.execute('UPDATE chats SET updated_at = ? WHERE id = ?', (now, chat_id))
    conn.commit()
    conn.close()
    
    # Build messages
    messages = []
    system_content = get_active_persona(config)
    context = get_context_content()
    if context:
        system_content += f"\n\n--- CONTEXT ---\n{context}"
    
    if system_content:
        messages.append({"role": "system", "content": system_content})
    
    for msg in history:
        messages.append({"role": msg['role'], "content": msg['content']})
    
    messages.append({"role": "user", "content": user_message})
    
    def generate():
        try:
            full_response = ""
            for content in call_provider_stream(config, messages):
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
            
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
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[Stream Error] {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# ----- Auto-title -----

@app.route('/api/chats/<chat_id>/auto-title', methods=['POST'])
def auto_title(chat_id):
    config = load_config()
    provider = config.get('provider', 'openai')
    api_key = config.get('api_keys', {}).get(provider, '')
    
    if not api_key:
        return jsonify({"title": "New Chat"})
    
    conn = get_db()
    first_msg = conn.execute(
        'SELECT content FROM messages WHERE chat_id = ? AND role = "user" ORDER BY id LIMIT 1',
        (chat_id,)
    ).fetchone()
    conn.close()
    
    if not first_msg:
        return jsonify({"title": "New Chat"})
    
    try:
        # Use a simple approach - just use the provider's API
        title_messages = [
            {"role": "system", "content": "Generate a short title (3-5 words) for this chat. Reply with just the title, no quotes."},
            {"role": "user", "content": first_msg['content'][:500]}
        ]
        
        title_parts = []
        for content in call_provider_stream(config, title_messages):
            title_parts.append(content)
        
        title = ''.join(title_parts).strip()[:50]
        
        conn = get_db()
        conn.execute('UPDATE chats SET title = ? WHERE id = ?', (title, chat_id))
        conn.commit()
        conn.close()
        
        return jsonify({"title": title})
    except Exception as e:
        print(f"[Auto-title Error] {e}")
        return jsonify({"error": str(e), "title": "New Chat"})

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CustomGPT Chat Server - Multi-Provider Edition")
    print("="*60)
    print(f"  Providers: OpenAI, Claude, Gemini, DeepSeek")
    print(f"  Reasoning models: GPT-5.x, o-series, deepseek-reasoner")
    print(f"  Default reasoning effort: {REASONING_EFFORT}")
    print(f"  Data: {DATA_DIR}")
    print("="*60)
    print("  Starting server on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
