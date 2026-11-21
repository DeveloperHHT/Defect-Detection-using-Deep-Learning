#!/bin/sh
# Builds Docs/User-Guide.pdf from Docs/Guide/guide.html with a Chromium-based browser in headless mode.
# Set BROWSER to the browser binary if it is not found automatically.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
browser="${BROWSER:-}"
for candidate in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
                 "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
                 "/Applications/Chromium.app/Contents/MacOS/Chromium" chromium chromium-browser google-chrome; do
  [ -n "$browser" ] && break
  if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then browser="$candidate"; fi
done
[ -n "$browser" ] || { echo "No Chromium-based browser found. Set BROWSER=/path/to/browser." >&2; exit 1; }
"$browser" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$root/Docs/User-Guide.pdf" "file://$root/Docs/Guide/guide.html" 2>/dev/null
echo "written: $root/Docs/User-Guide.pdf"
