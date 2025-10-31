# How to Open Progress Window Outside Cursor

## Quick Method

Run this command:
```bash
bash scripts/open_progress_window_simple.sh
```

This will:
- ✅ Open a new Terminal window
- ✅ Navigate to the project directory
- ✅ Activate the virtual environment
- ✅ Start the progress monitor with auto-refresh

## Manual Method

1. **Open Terminal** (outside Cursor)
   - Press `Cmd + Space`, type "Terminal", press Enter
   - Or find Terminal in Applications > Utilities

2. **Navigate to project**:
   ```bash
   cd /Users/gilraitses/ecs630/labs/termprojectproposal
   ```

3. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Run progress window**:
   ```bash
   python3 scripts/show_progress.py --watch
   ```

## What You'll See

The progress window will show:
- Current condition (e.g., "1/45")
- Visual progress bar
- Current replication (e.g., "8/30")
- Time elapsed and remaining
- Estimated completion time
- Auto-refreshes every 5 seconds

Press **Ctrl+C** in that Terminal window to stop monitoring (simulation continues).

## Alternative: Quick Check Without Window

For a one-time check (no auto-refresh):
```bash
python3 scripts/show_progress.py
```



