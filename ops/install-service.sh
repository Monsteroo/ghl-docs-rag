#!/bin/bash
set -euo pipefail

PLIST_SRC="$(dirname "$0")/com.vitaliimaslii.ghl-docs-rag.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.vitaliimaslii.ghl-docs-rag.plist"

cp "$PLIST_SRC" "$PLIST_DEST"
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"
echo "Installed. Check status: launchctl list | grep ghl-docs-rag"
