#!/usr/bin/env python3
"""
Session State Manager

Tracks processes started during a trading session to enable clean shutdown
that only affects processes we started, not pre-existing processes.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-11
"""

import json
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class SessionState:
    """Manages session state for tracking started processes"""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent

        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / "data" / ".session_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save_pre_existing_processes(self) -> Dict:
        """
        Record all processes running BEFORE we start anything.
        This allows us to avoid killing pre-existing processes.
        """
        patterns = [
            "bigbrother",
            "streamlit",
            "news_ingestion",
            "token_refresh_service",
            "phase5",
        ]

        pre_existing = {}

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])

                # Check if this matches any of our patterns
                for pattern in patterns:
                    if pattern in cmdline or pattern in proc.info['name']:
                        pre_existing[proc.info['pid']] = {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline,
                            'create_time': proc.info['create_time'],
                            'pattern': pattern
                        }
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return pre_existing

    def record_session_start(self, pre_existing_pids: List[int]):
        """
        Record session start with list of pre-existing process PIDs.
        We will NOT shut these down during shutdown.
        """
        state = {
            'session_start_time': datetime.now().isoformat(),
            'pre_existing_pids': pre_existing_pids,
            'started_processes': [],  # Will be populated as we start things
        }

        self.state_file.write_text(json.dumps(state, indent=2))
        return state

    def add_started_process(self, pid: int, name: str, cmdline: str, description: str):
        """Record a process that WE started during this session"""
        state = self.load_state()

        if state:
            state['started_processes'].append({
                'pid': pid,
                'name': name,
                'cmdline': cmdline,
                'description': description,
                'start_time': datetime.now().isoformat()
            })

            self.state_file.write_text(json.dumps(state, indent=2))

    def load_state(self) -> Optional[Dict]:
        """Load current session state"""
        if not self.state_file.exists():
            return None

        try:
            return json.loads(self.state_file.read_text())
        except Exception:
            return None

    def get_processes_to_shutdown(self) -> Dict[str, List]:
        """
        Get list of processes that should be shut down.
        Returns dict with:
          - 'started_by_us': Processes we started (ALWAYS shut down)
          - 'pre_existing': Processes that were running before (NEVER shut down)
          - 'new_unknown': Processes matching patterns that we didn't start (ASK user)
        """
        state = self.load_state()

        if not state:
            # No state file - return all matching processes as 'new_unknown'
            return {
                'started_by_us': [],
                'pre_existing': [],
                'new_unknown': self._find_all_matching_processes()
            }

        pre_existing_pids = set(state.get('pre_existing_pids', []))
        started_by_us_pids = {p['pid'] for p in state.get('started_processes', [])}

        # Find all current matching processes
        all_current = self._find_all_matching_processes()

        started_by_us = []
        pre_existing = []
        new_unknown = []

        for proc in all_current:
            pid = proc['pid']

            if pid in started_by_us_pids:
                started_by_us.append(proc)
            elif pid in pre_existing_pids:
                pre_existing.append(proc)
            else:
                new_unknown.append(proc)

        return {
            'started_by_us': started_by_us,
            'pre_existing': pre_existing,
            'new_unknown': new_unknown
        }

    def _find_all_matching_processes(self) -> List[Dict]:
        """Find all processes matching our patterns"""
        patterns = [
            ("bigbrother", "Trading Engine"),
            ("streamlit", "Dashboard"),
            ("news_ingestion", "News Ingestion"),
            ("token_refresh_service", "Token Refresh Service"),
            ("python.*phase5", "Phase 5 Scripts"),
        ]

        found = []

        for pattern, description in patterns:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if pattern in cmdline or pattern in proc.info['name']:
                        # Exclude the current script
                        if 'phase5_shutdown' in cmdline or 'phase5_setup' in cmdline:
                            continue

                        found.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline[:100],
                            'description': description,
                            'pattern': pattern
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        return found

    def clear_session(self):
        """Clear session state (call after successful shutdown)"""
        if self.state_file.exists():
            self.state_file.unlink()

    def update_started_processes_status(self):
        """
        Update state file to remove processes that are no longer running.
        This helps keep state file accurate.
        """
        state = self.load_state()
        if not state:
            return

        # Check which started processes are still running
        running_pids = {p.pid for p in psutil.process_iter(['pid'])}

        state['started_processes'] = [
            p for p in state.get('started_processes', [])
            if p['pid'] in running_pids
        ]

        self.state_file.write_text(json.dumps(state, indent=2))
