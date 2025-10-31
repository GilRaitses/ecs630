#!/bin/bash
# Open progress window in a new Terminal window

cd /Users/gilraitses/ecs630/labs/termprojectproposal
source venv/bin/activate

# Open new Terminal window and run progress script
osascript -e 'tell application "Terminal" to do script "cd /Users/gilraitses/ecs630/labs/termprojectproposal && source venv/bin/activate && python3 scripts/show_progress.py --watch"'



