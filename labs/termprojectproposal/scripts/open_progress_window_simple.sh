#!/bin/bash
# Simple version - opens Terminal with progress monitoring

cd /Users/gilraitses/ecs630/labs/termprojectproposal

# Open new Terminal window with watch command
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd /Users/gilraitses/ecs630/labs/termprojectproposal && source venv/bin/activate && python3 scripts/show_progress.py --watch"
end tell
EOF



