#!/usr/bin/env python
"""
Google Account Rotator CLI
Checks accounts, Google AI Pro quotas, and Midnight Pacific Time Reset ETA.
"""

import json
import os
import shutil
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

GCLOUD_BIN = shutil.which('gcloud') or 'gcloud'
ACCOUNTS_FILE = os.path.join(os.path.dirname(__file__), 'accounts.json')
ACTIVE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'active_config.json')


def get_pacific_time_reset_eta():
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
        "reset_at_pt": next_midnight.strftime("%Y-%m-%d 12:00:00 AM %Z"),
        "countdown": f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    }


def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return []
    try:
        with open(ACCOUNTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def get_active_gcloud_account():
    try:
        res = subprocess.run(
            [GCLOUD_BIN, 'config', 'get-value', 'account'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_active_api_key_id():
    if os.path.exists(ACTIVE_CONFIG_FILE):
        try:
            with open(ACTIVE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get('active_api_key_id')
        except Exception:
            pass
    return None


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

    added = False
    for item in gcloud_accounts:
        email = item.get('account')
        if not email or email in existing_emails:
            continue
        is_sa = email.endswith('.iam.gserviceaccount.com')
        accounts.append({
            "id": f"acc-{len(accounts)+1}",
            "account_name": f"Service Account ({email.split('@')[0]})" if is_sa else email.split('@')[0].title(),
            "type": "service_account" if is_sa else "gcloud",
            "email": email,
            "tier": "free",
            "model": "gemini-1.5-pro"
        })
        existing_emails.add(email)
        added = True

    if added:
        with open(ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(accounts, f, indent=2)
    return accounts


def main():
    print("=" * 68)
    print("      GOOGLE ACCOUNT ROTATOR & AI PRO QUOTA MONITOR")
    print("=" * 68)

    reset_info = get_pacific_time_reset_eta()
    print(f" Current Pacific Time: {reset_info['current_pt']}")
    print(f" Daily Quota Reset:    {reset_info['reset_at_pt']}")
    print(f" Time Until Reset ETA: \033[92m{reset_info['countdown']}\033[0m")
    print("-" * 68)

    accounts = load_accounts()
    if not accounts:
        print("No accounts in accounts.json. Auto-syncing from gcloud...")
        accounts = auto_sync_gcloud_accounts()

    if not accounts:
        print("No accounts found. Please run 'gcloud auth login' or add API keys.")
        return

    active_gcloud = get_active_gcloud_account()
    active_api_key_id = get_active_api_key_id()

    print(f"Found {len(accounts)} configured account(s):\n")

    for i, acc in enumerate(accounts, 1):
        name = acc.get('account_name', 'Unnamed')
        acc_type = acc.get('type', 'gcloud')
        email = acc.get('email', '')
        api_key = acc.get('api_key', '')
        model = acc.get('model', 'gemini-1.5-pro')
        tier = acc.get('tier', 'free')

        is_active = False
        if acc_type in ['gcloud', 'service_account']:
            is_active = (email == active_gcloud) if email else False
        elif acc_type == 'gemini_api_key':
            is_active = (acc.get('id') == active_api_key_id)

        active_badge = "\033[92m[ACTIVE]\033[0m" if is_active else "[STANDBY]"
        identifier = email if email else (f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "API_KEY")

        print(f" {i}. {name} {active_badge}")
        print(f"    Type:       {acc_type.upper()}")
        print(f"    ID/Email:   {identifier}")
        print(f"    AI Model:   {model} ({tier.upper()} Tier)")
        if tier == 'free':
            print(f"    Quotas:     2 RPM | 32,000 TPM | 50 Requests/Day")
        else:
            print(f"    Quotas:     360 RPM | 2,000,000 TPM | Unlimited RPD")
        print(f"    Reset ETA:  {reset_info['countdown']} (Midnight PT)")
        print()

    print("=" * 68)
    print("Switch Accounts:")
    print("  • Run web dashboard: python app.py  (visit http://127.0.0.1:5000)")
    print("  • CLI switch gcloud: gcloud config set account <EMAIL>")
    print("=" * 68)


if __name__ == '__main__':
    main()
