#!/usr/bin/env python
"""
Google Account Rotator & AI Pro Quota Dashboard
Supports Antigravity & Google AI models:
- Gemini 3.8 Flash Medium (Fast)
- Gemini 3.7 Flash Medium (Fast)
- Gemini 3.6 Flash Medium (Fast)
- Gemini 3.1 Pro Low
- Claude Sonnet 4.6 (Thinking)
- Claude Opus 4.6 (Thinking)
- GPT-OSS 120B (Medium)
- Plus Gemini 1.5/2.0/2.5 series
Tracks live Quotas (RPM, TPM, RPD), usage tracking, and Midnight Pacific Time Reset ETA.
"""

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, request, render_template, Response, stream_with_context

GCLOUD_BIN = shutil.which('gcloud') or 'gcloud'

ACCOUNTS_FILE = os.path.join(os.path.dirname(__file__), 'accounts.json')
ACTIVE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'active_config.json')
ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

# Direct path to gcloud config file — faster than invoking gcloud subprocess for polling
GCLOUD_CONFIG_PATH = os.path.join(
    os.environ.get('APPDATA', os.path.expanduser('~')),
    'gcloud', 'configurations', 'config_default'
)

app = Flask(__name__, template_folder='frontend', static_folder='frontend')


# --- Antigravity & Google AI Model Quota Registry ---

MODEL_QUOTA_SPECS = {
    "gemini-3.8-flash": {
        "name": "Gemini 3.8 Flash Medium",
        "family": "Gemini Flash",
        "badge": "Fast",
        "free": {"rpm": 60, "tpm": 2000000, "rpd": 1500, "description": "High-Speed Agentic (60 RPM / 2M TPM / 1500 RPD)"},
        "pro": {"rpm": 120, "tpm": 4000000, "rpd": 5000, "description": "Antigravity Pro Tier (120 RPM / 4M TPM / 5000 RPD)"}
    },
    "gemini-3.7-flash": {
        "name": "Gemini 3.7 Flash Medium",
        "family": "Gemini Flash",
        "badge": "Fast",
        "free": {"rpm": 60, "tpm": 2000000, "rpd": 1500, "description": "High-Speed Agentic (60 RPM / 2M TPM / 1500 RPD)"},
        "pro": {"rpm": 120, "tpm": 4000000, "rpd": 5000, "description": "Antigravity Pro Tier (120 RPM / 4M TPM / 5000 RPD)"}
    },
    "gemini-3.6-flash": {
        "name": "Gemini 3.6 Flash Medium",
        "family": "Gemini Flash",
        "badge": "Fast",
        "free": {"rpm": 60, "tpm": 2000000, "rpd": 1500, "description": "High-Speed Agentic (60 RPM / 2M TPM / 1500 RPD)"},
        "pro": {"rpm": 120, "tpm": 4000000, "rpd": 5000, "description": "Antigravity Pro Tier (120 RPM / 4M TPM / 5000 RPD)"}
    },
    "gemini-3.1-pro": {
        "name": "Gemini 3.1 Pro Low",
        "family": "Gemini Pro",
        "badge": "Reasoning",
        "free": {"rpm": 15, "tpm": 500000, "rpd": 100, "description": "Standard Reasoning Pool (15 RPM / 500k TPM / 100 RPD)"},
        "pro": {"rpm": 50, "tpm": 2000000, "rpd": 500, "description": "Deep Reasoning Pool (50 RPM / 2M TPM / 500 RPD)"}
    },
    "claude-sonnet-4.6": {
        "name": "Claude Sonnet 4.6 (Thinking)",
        "family": "Claude",
        "badge": "Thinking",
        "free": {"rpm": 10, "tpm": 250000, "rpd": 50, "description": "Thinking Model Pool (10 RPM / 250k TPM / 50 RPD)"},
        "pro": {"rpm": 30, "tpm": 1000000, "rpd": 200, "description": "Extended Thinking Pool (30 RPM / 1M TPM / 200 RPD)"}
    },
    "claude-opus-4.6": {
        "name": "Claude Opus 4.6 (Thinking)",
        "family": "Claude",
        "badge": "Thinking",
        "free": {"rpm": 5, "tpm": 100000, "rpd": 25, "description": "Ultra Reasoning Pool (5 RPM / 100k TPM / 25 RPD)"},
        "pro": {"rpm": 15, "tpm": 500000, "rpd": 100, "description": "Opus Ultra Pool (15 RPM / 500k TPM / 100 RPD)"}
    },
    "gpt-oss-120b": {
        "name": "GPT-OSS 120B (Medium)",
        "family": "Open-Weights",
        "badge": "OSS",
        "free": {"rpm": 30, "tpm": 1000000, "rpd": 500, "description": "Open Model Inference Pool (30 RPM / 1M TPM / 500 RPD)"},
        "pro": {"rpm": 60, "tpm": 2000000, "rpd": 1500, "description": "High-Throughput Open Pool (60 RPM / 2M TPM / 1500 RPD)"}
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro",
        "family": "Gemini Pro",
        "badge": "Pro",
        "free": {"rpm": 2, "tpm": 32000, "rpd": 50, "description": "Free Tier (2 RPM / 32k TPM / 50 RPD)"},
        "pro": {"rpm": 360, "tpm": 2000000, "rpd": 10000, "description": "Pay-as-you-go (360 RPM / 2M TPM)"}
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "family": "Gemini Pro",
        "badge": "Pro",
        "free": {"rpm": 2, "tpm": 32000, "rpd": 50, "description": "Free Tier (2 RPM / 32k TPM / 50 RPD)"},
        "pro": {"rpm": 360, "tpm": 2000000, "rpd": 10000, "description": "Pay-as-you-go (360 RPM / 2M TPM)"}
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "family": "Gemini Flash",
        "badge": "Fast",
        "free": {"rpm": 15, "tpm": 1000000, "rpd": 1500, "description": "Free Tier (15 RPM / 1M TPM / 1500 RPD)"},
        "pro": {"rpm": 1000, "tpm": 4000000, "rpd": 50000, "description": "Pay-as-you-go (1000 RPM / 4M TPM)"}
    },
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "family": "Gemini Flash",
        "badge": "Fast",
        "free": {"rpm": 15, "tpm": 1000000, "rpd": 1500, "description": "Free Tier (15 RPM / 1M TPM / 1500 RPD)"},
        "pro": {"rpm": 1000, "tpm": 4000000, "rpd": 50000, "description": "Pay-as-you-go (1000 RPM / 4M TPM)"}
    }
}


def get_pacific_time_reset_eta():
    """
    Computes exact Pacific Time and seconds/countdown until Midnight PT,
    when daily quotas (RPD) reset.
    """
    try:
        now_pt = datetime.now(ZoneInfo('America/Los_Angeles'))
    except Exception:
        now_pt = datetime.utcnow() - timedelta(hours=7)

    next_midnight = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_remaining = max(0, int((next_midnight - now_pt).total_seconds()))
    hours = seconds_remaining // 3600
    minutes = (seconds_remaining % 3600) // 60
    secs = seconds_remaining % 60

    return {
        "current_pt": now_pt.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
        "current_date_pt": now_pt.strftime("%Y-%m-%d"),
        "reset_at_pt": next_midnight.strftime("%Y-%m-%d 12:00:00 AM %Z"),
        "seconds_remaining": seconds_remaining,
        "countdown": f"{hours:02d}h {minutes:02d}m {secs:02d}s",
        "reset_cycle": "Daily reset at 00:00 Midnight Pacific Time (PT)"
    }


def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return []
    try:
        with open(ACCOUNTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            accounts = data if isinstance(data, list) else []
            # Check and perform daily usage reset if day rolled over
            accounts = ensure_daily_reset(accounts)
            return accounts
    except Exception:
        return []


def save_accounts(accounts):
    with open(ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(accounts, f, indent=2)


def ensure_daily_reset(accounts):
    """Automatically resets daily usage counts if current Pacific date is newer than last recorded."""
    reset_info = get_pacific_time_reset_eta()
    current_date = reset_info["current_date_pt"]
    modified = False

    for acc in accounts:
        if acc.get("last_reset_date") != current_date:
            acc["last_reset_date"] = current_date
            # Reset usage counts
            usage_map = acc.get("usage", {})
            for m_key in usage_map:
                usage_map[m_key]["used_today"] = 0
                if usage_map[m_key].get("status") == "exhausted":
                    usage_map[m_key]["status"] = "healthy"
            acc["usage"] = usage_map
            modified = True

    if modified:
        with open(ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(accounts, f, indent=2)

    return accounts


def get_account_model_usage(account, model_key):
    """Gets usage and quota info for a given model on an account."""
    tier = account.get('tier', 'free')
    tier_key = 'pro' if tier in ['pro', 'paid'] else 'free'
    specs = MODEL_QUOTA_SPECS.get(model_key, {}).get(tier_key, {
        "rpm": 15, "tpm": 500000, "rpd": 100, "description": "Standard"
    })
    
    usage_data = account.get('usage', {}).get(model_key, {})
    used_today = usage_data.get('used_today', 0)
    status = usage_data.get('status', 'healthy')

    # Calculate percentage
    rpd_limit = specs.get('rpd', 100)
    pct = min(100, int((used_today / rpd_limit) * 100)) if rpd_limit > 0 else 0

    if status != 'exhausted':
        if pct >= 90:
            status = 'warning'
        else:
            status = 'healthy'

    return {
        "model_key": model_key,
        "model_name": MODEL_QUOTA_SPECS.get(model_key, {}).get('name', model_key),
        "family": MODEL_QUOTA_SPECS.get(model_key, {}).get('family', 'General'),
        "badge": MODEL_QUOTA_SPECS.get(model_key, {}).get('badge', ''),
        "used_today": used_today,
        "rpd_limit": rpd_limit,
        "rpm_limit": specs.get('rpm', 15),
        "tpm_limit": specs.get('tpm', 500000),
        "used_pct": pct,
        "status": status,
        "description": specs.get('description', '')
    }


def get_active_gcloud_account():
    try:
        res = subprocess.run(
            [GCLOUD_BIN, 'config', 'get-value', 'account'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_active_gcloud_project():
    try:
        res = subprocess.run(
            [GCLOUD_BIN, 'config', 'get-value', 'project'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_active_api_key_id():
    if os.path.exists(ACTIVE_CONFIG_FILE):
        try:
            with open(ACTIVE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                return cfg.get('active_api_key_id')
        except Exception:
            pass
    return None


def _write_user_env_var(name: str, value: str):
    """
    Persists a user-level environment variable via the Windows registry.
    Antigravity IDE and any new terminal launched after this call will inherit it.
    """
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r'Environment',
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        winreg.CloseKey(key)
        # Notify running processes (fire-and-forget, non-blocking)
        import ctypes
        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x001A
        ctypes.windll.user32.SendNotifyMessageW(
            HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment'
        )
    except Exception:
        pass  # Non-fatal: .env file still carries the value


def _clear_user_env_var(name: str):
    """Remove a user-level env variable from Windows registry."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r'Environment',
            0, winreg.KEY_SET_VALUE
        )
        try:
            winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass
        winreg.CloseKey(key)
    except Exception:
        pass


def _sync_env_file(updates: dict):
    """
    Write/update keys in the project .env file.
    Existing keys not in `updates` are preserved.
    """
    try:
        lines = []
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    key = line.split('=', 1)[0].strip()
                    if key not in updates:
                        lines.append(line)
        for k, v in updates.items():
            if v:
                lines.append(f"{k}={v}\n")
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception:
        pass


def set_active_api_key(account_id, api_key, project_id=''):
    with open(ACTIVE_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump({"active_api_key_id": account_id}, f, indent=2)
    _sync_env_file({'GEMINI_API_KEY': api_key, 'GOOGLE_API_KEY': api_key})
    _write_user_env_var('GEMINI_API_KEY', api_key)
    _write_user_env_var('GOOGLE_API_KEY', api_key)
    if project_id:
        _sync_env_file({'GOOGLE_CLOUD_PROJECT': project_id})
        _write_user_env_var('GOOGLE_CLOUD_PROJECT', project_id)


def test_gemini_api_key(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    start_time = datetime.now()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GoogleAccountRotator/2.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            data = json.loads(response.read().decode('utf-8'))
            models_count = len(data.get('models', []))
            return {
                "status": "Healthy",
                "healthy": True,
                "http_code": 200,
                "latency_ms": latency_ms,
                "message": f"Verified ({models_count} models available)",
                "models_count": models_count
            }
    except urllib.error.HTTPError as e:
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        try:
            body = json.loads(e.read().decode('utf-8'))
            msg = body.get('error', {}).get('message', str(e))
        except Exception:
            msg = str(e)

        if e.code == 429:
            return {
                "status": "Rate-Limited (429)",
                "healthy": False,
                "http_code": 429,
                "latency_ms": latency_ms,
                "message": f"Quota Exceeded / 429: {msg[:100]}"
            }
        else:
            return {
                "status": f"HTTP {e.code}",
                "healthy": False,
                "http_code": e.code,
                "latency_ms": latency_ms,
                "message": f"API Error: {msg[:100]}"
            }
    except Exception as e:
        return {
            "status": "Connection Error",
            "healthy": False,
            "http_code": 0,
            "latency_ms": 0,
            "message": str(e)[:100]
        }


def auto_sync_gcloud_accounts():
    accounts = load_accounts()
    existing_emails = {a.get('email') for a in accounts if a.get('email')}

    try:
        res = subprocess.run(
            [GCLOUD_BIN, 'auth', 'list', '--format=json'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        gcloud_accounts = json.loads(res.stdout)
    except Exception:
        gcloud_accounts = []

    active_project = get_active_gcloud_project()
    added_any = False

    for item in gcloud_accounts:
        email = item.get('account')
        if not email or email in existing_emails:
            continue

        is_service_account = email.endswith('.iam.gserviceaccount.com')
        name = email.split('@')[0].replace('-', ' ').title()
        if is_service_account:
            name = f"Service Account ({name})"

        account_obj = {
            "id": str(uuid.uuid4())[:8],
            "account_name": name,
            "type": "service_account" if is_service_account else "gcloud",
            "email": email,
            "project_id": active_project if item.get('status') == 'ACTIVE' else "",
            "key_file": "",
            "tier": "free",
            "model": "gemini-3.8-flash",
            "usage": {},
            "added_via": "gcloud_auto_sync"
        }
        accounts.append(account_obj)
        existing_emails.add(email)
        added_any = True

    if added_any:
        save_accounts(accounts)
    return accounts


# --- API Routes ---

@app.route('/api/models')
def get_models():
    """Returns the registry of all supported AI models and their quota specifications."""
    models_list = []
    for key, val in MODEL_QUOTA_SPECS.items():
        models_list.append({
            "key": key,
            "name": val["name"],
            "family": val["family"],
            "badge": val["badge"],
            "free": val["free"],
            "pro": val["pro"]
        })
    return jsonify(models_list)


@app.route('/api/accounts')
def get_accounts():
    accounts = load_accounts()
    if not accounts:
        accounts = auto_sync_gcloud_accounts()

    active_gcloud = get_active_gcloud_account()
    active_api_key_id = get_active_api_key_id()
    active_project = get_active_gcloud_project()
    reset_info = get_pacific_time_reset_eta()

    results = []
    for acc in accounts:
        acc_type = acc.get('type', 'gcloud')
        acc_id = acc.get('id', str(uuid.uuid4())[:8])
        email = acc.get('email', '')
        api_key = acc.get('api_key', '')
        primary_model = acc.get('model', 'gemini-3.8-flash')
        tier = acc.get('tier', 'free')

        is_active = False
        if acc_type in ['gcloud', 'service_account']:
            is_active = (email == active_gcloud) if email else False
        elif acc_type == 'gemini_api_key':
            is_active = (acc_id == active_api_key_id)

        masked_identifier = email
        if acc_type == 'gemini_api_key' and api_key:
            masked_identifier = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "AIzaSy..."

        # Calculate model usage breakdown across all models
        model_usages = {}
        for m_key in MODEL_QUOTA_SPECS.keys():
            model_usages[m_key] = get_account_model_usage(acc, m_key)

        primary_usage = model_usages.get(primary_model, get_account_model_usage(acc, primary_model))

        results.append({
            "id": acc_id,
            "name": acc.get('account_name', 'Unnamed Account'),
            "type": acc_type,
            "email": email,
            "identifier": masked_identifier,
            "project_id": acc.get('project_id') or (active_project if is_active else ""),
            "key_file": acc.get('key_file', ''),
            "is_active": is_active,
            "model": primary_model,
            "model_name": MODEL_QUOTA_SPECS.get(primary_model, {}).get('name', primary_model),
            "tier": tier,
            "primary_usage": primary_usage,
            "model_usages": model_usages,
            "reset_eta": reset_info["countdown"]
        })

    return jsonify({
        "accounts": results,
        "reset_info": reset_info,
        "active_gcloud_account": active_gcloud,
        "active_gcloud_project": active_project
    })


@app.route('/api/accounts/usage', methods=['POST'])
def update_account_usage():
    """Allows incrementing usage, resetting usage, or toggling status (warning/exhausted/healthy)."""
    data = request.json or {}
    account_id = data.get('account_id')
    model_key = data.get('model', 'gemini-3.8-flash')
    action = data.get('action', 'increment') # 'increment', 'mark_exhausted', 'mark_healthy', 'reset'

    accounts = load_accounts()
    target_acc = next((a for a in accounts if a.get('id') == account_id), None)
    if not target_acc:
        return jsonify({"error": "Account not found"}), 404

    usage_map = target_acc.get('usage', {})
    if model_key not in usage_map:
        usage_map[model_key] = {"used_today": 0, "status": "healthy"}

    if action == 'increment':
        usage_map[model_key]["used_today"] = usage_map[model_key].get("used_today", 0) + 1
    elif action == 'mark_exhausted':
        usage_map[model_key]["status"] = "exhausted"
    elif action == 'mark_healthy':
        usage_map[model_key]["status"] = "healthy"
    elif action == 'reset':
        usage_map[model_key]["used_today"] = 0
        usage_map[model_key]["status"] = "healthy"

    target_acc['usage'] = usage_map
    save_accounts(accounts)

    return jsonify({
        "success": True,
        "usage": get_account_model_usage(target_acc, model_key)
    })


@app.route('/api/switch', methods=['POST'])
def switch_account():
    """
    Switches the active account and binds the identity to the Antigravity IDE
    by updating:
      1. gcloud active account + project (system-wide gcloud config)
      2. GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_CLOUD_PROJECT in the Windows
         user-level environment registry (picked up by new IDE windows/terminals)
      3. The project .env file
    """
    data = request.json or {}
    account_id = data.get('account_id')
    email = data.get('email')

    accounts = load_accounts()
    target_acc = next(
        (a for a in accounts
         if a.get('id') == account_id or (email and a.get('email') == email)),
        None
    )
    if not target_acc:
        return jsonify({"error": "Account not found"}), 404

    acc_type = target_acc.get('type', 'gcloud')
    ide_sync = []  # steps taken for IDE binding

    # ── gcloud / service account ────────────────────────────────────────────
    if acc_type in ['gcloud', 'service_account']:
        target_email = target_acc.get('email')
        key_file     = target_acc.get('key_file')
        project_id   = target_acc.get('project_id', '').strip()

        if not target_email and not key_file:
            return jsonify({"error": "No email or key file associated with this account"}), 400

        try:
            if key_file and os.path.exists(key_file):
                subprocess.run(
                    [GCLOUD_BIN, 'auth', 'activate-service-account', f'--key-file={key_file}'],
                    check=True, capture_output=True, text=True, encoding='utf-8'
                )
                ide_sync.append('activated service-account key')

            if target_email:
                subprocess.run(
                    [GCLOUD_BIN, 'config', 'set', 'account', target_email],
                    check=True, capture_output=True, text=True, encoding='utf-8'
                )
                ide_sync.append(f'gcloud active account → {target_email}')

            if project_id:
                subprocess.run(
                    [GCLOUD_BIN, 'config', 'set', 'project', project_id],
                    check=True, capture_output=True, text=True, encoding='utf-8'
                )
                ide_sync.append(f'gcloud active project → {project_id}')
                _write_user_env_var('GOOGLE_CLOUD_PROJECT', project_id)
                _sync_env_file({'GOOGLE_CLOUD_PROJECT': project_id})
                ide_sync.append('GOOGLE_CLOUD_PROJECT env var updated (new terminals)')

            # Clear any stale API key env vars from previous key-based account
            _clear_user_env_var('GEMINI_API_KEY')
            _clear_user_env_var('GOOGLE_API_KEY')
            ide_sync.append('stale API key env vars cleared')

            return jsonify({
                "success": True,
                "message": f"Switched to {target_email or 'service account'}",
                "ide_sync": ide_sync
            })

        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"gcloud error: {e.stderr.strip()}"}), 500

    # ── Gemini API Key ───────────────────────────────────────────────────────
    elif acc_type == 'gemini_api_key':
        api_key    = target_acc.get('api_key', '')
        project_id = target_acc.get('project_id', '').strip()
        if not api_key:
            return jsonify({"error": "No API key found for this account"}), 400

        set_active_api_key(target_acc.get('id'), api_key, project_id)
        ide_sync.append('GEMINI_API_KEY / GOOGLE_API_KEY written to Windows user env (new terminals)')
        ide_sync.append('.env file updated in project directory')
        if project_id:
            ide_sync.append(f'GOOGLE_CLOUD_PROJECT → {project_id}')

        return jsonify({
            "success": True,
            "message": f"Activated '{target_acc.get('account_name')}' — API key bound to IDE",
            "ide_sync": ide_sync
        })

    return jsonify({"error": f"Unknown account type: {acc_type}"}), 400


@app.route('/api/accounts/add', methods=['POST'])
def add_account():
    data = request.json or {}
    acc_name = data.get('name', '').strip()
    acc_type = data.get('type', 'gemini_api_key')
    tier = data.get('tier', 'free')
    model = data.get('model', 'gemini-3.8-flash')

    if not acc_name:
        return jsonify({"error": "Account name is required"}), 400

    account_obj = {
        "id": str(uuid.uuid4())[:8],
        "account_name": acc_name,
        "type": acc_type,
        "tier": tier,
        "model": model,
        "usage": {},
        "added_at": datetime.utcnow().isoformat()
    }

    if acc_type == 'gemini_api_key':
        api_key = data.get('api_key', '').strip()
        if not api_key:
            return jsonify({"error": "API Key is required"}), 400
        account_obj["api_key"] = api_key
        test_result = test_gemini_api_key(api_key)
        account_obj["last_health"] = test_result

    elif acc_type == 'gcloud':
        email = data.get('email', '').strip()
        if not email:
            return jsonify({"error": "Google Email is required"}), 400
        account_obj["email"] = email
        account_obj["project_id"] = data.get('project_id', '').strip()

    elif acc_type == 'service_account':
        key_file = data.get('key_file', '').strip()
        if not key_file or not os.path.exists(key_file):
            return jsonify({"error": f"Service account key file not found at '{key_file}'"}), 400
        
        account_obj["key_file"] = key_file
        try:
            with open(key_file, 'r') as f:
                kdata = json.load(f)
                account_obj["email"] = kdata.get('client_email', '')
                account_obj["project_id"] = kdata.get('project_id', '')
        except Exception as e:
            return jsonify({"error": f"Invalid key file: {e}"}), 400

    accounts = load_accounts()
    accounts.append(account_obj)
    save_accounts(accounts)

    return jsonify({"success": True, "account": account_obj})


@app.route('/api/accounts/sync-gcloud', methods=['POST'])
def sync_gcloud():
    accounts = auto_sync_gcloud_accounts()
    return jsonify({"success": True, "count": len(accounts)})


@app.route('/api/accounts/test', methods=['POST'])
def test_account():
    data = request.json or {}
    account_id = data.get('account_id')
    accounts = load_accounts()

    target_acc = next((a for a in accounts if a.get('id') == account_id), None)
    if not target_acc:
        return jsonify({"error": "Account not found"}), 404

    acc_type = target_acc.get('type')
    if acc_type == 'gemini_api_key':
        api_key = target_acc.get('api_key', '')
        res = test_gemini_api_key(api_key)
        return jsonify(res)
    elif acc_type in ['gcloud', 'service_account']:
        email = target_acc.get('email')
        try:
            sub = subprocess.run(
                [GCLOUD_BIN, 'auth', 'print-access-token', f'--account={email}'],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            return jsonify({
                "status": "Authenticated",
                "healthy": True,
                "message": "Token generated successfully via gcloud"
            })
        except subprocess.CalledProcessError as e:
            return jsonify({
                "status": "Auth Expired/Missing",
                "healthy": False,
                "message": e.stderr.strip()[:100]
            })
    return jsonify({"status": "Unknown", "healthy": False, "message": "Unknown account type"})


@app.route('/api/accounts/<account_id>', methods=['DELETE'])
def delete_account(account_id):
    accounts = load_accounts()
    initial_len = len(accounts)
    accounts = [a for a in accounts if a.get('id') != account_id]
    if len(accounts) == initial_len:
        return jsonify({"error": "Account not found"}), 404
    save_accounts(accounts)
    return jsonify({"success": True, "message": "Account removed"})


# --- SSE Live Account Watch ---

def _read_gcloud_active_account_from_file():
    """Reads the active gcloud account directly from the config file (no subprocess)."""
    try:
        with open(GCLOUD_CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                m = re.match(r'^\s*account\s*=\s*(.+)$', line)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    # Fallback: invoke gcloud subprocess
    try:
        r = subprocess.run([GCLOUD_BIN, 'config', 'get-value', 'account'],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or 'Unknown'
    except Exception:
        return 'Unknown'


def _get_gcloud_project_from_file():
    """Reads the active gcloud project directly from the config file."""
    try:
        with open(GCLOUD_CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                m = re.match(r'^\s*project\s*=\s*(.+)$', line)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    try:
        r = subprocess.run([GCLOUD_BIN, 'config', 'get-value', 'project'],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or 'Unknown'
    except Exception:
        return 'Unknown'


@app.route('/api/stream/gcloud-account')
def stream_gcloud_account():
    """
    Server-Sent Events endpoint. Streams the current active gcloud account every 2 seconds.
    The frontend subscribes to this to auto-reflect account switches from any source
    (this web app, the CLI, or gcloud IDE integrations).
    """
    def generate():
        last_account = None
        while True:
            current = _read_gcloud_active_account_from_file()
            if current != last_account:
                last_account = current
                payload = json.dumps({
                    "active_account": current,
                    "ts": int(time.time())
                })
                yield f"data: {payload}\n\n"
            else:
                # Send a heartbeat comment every 2s to keep connection alive
                yield f": heartbeat\n\n"
            time.sleep(2)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/gcloud-status')
def gcloud_status():
    """Snapshot of the current gcloud active account and project."""
    active_account = _read_gcloud_active_account_from_file()
    active_project = _get_gcloud_project_from_file()
    return jsonify({
        "active_account": active_account,
        "active_project": active_project
    })


# --- Frontend Route ---

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
