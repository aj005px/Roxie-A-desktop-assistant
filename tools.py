"""
Tools the agent can call.
Read-only + one narrow write tool (Obsidian vault only).
"""

import os
import platform
import shutil
import datetime
import threading
import time

from duckduckgo_search import DDGS

VAULT_PATH = os.path.expanduser("~/Documents/ObsidianVault")

# -----------------------------
# System
# -----------------------------

def system_info() -> str:
    total, used, free = shutil.disk_usage("/")

    return (
        f"OS: {platform.system()} {platform.release()}\\n"
        f"Machine: {platform.machine()}\\n"
        f"Disk: {used // (2**30)}GB used / {total // (2**30)}GB total "
        f"({free // (2**30)}GB free)"
    )

# -----------------------------
# Files
# -----------------------------

def list_files(directory: str = ".") -> str:
    try:
        directory = os.path.expanduser(directory)
        files = os.listdir(directory)
        return "\\n".join(files) if files else "Empty directory"
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    try:
        path = os.path.expanduser(path)

        with open(path, "r") as f:
            return f.read()[:4000]

    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# Notes
# -----------------------------

def write_note(content: str, filename: str = "note.md") -> str:
    os.makedirs(VAULT_PATH, exist_ok=True)

    path = os.path.join(VAULT_PATH, filename)

    with open(path, "a") as f:
        f.write(
            f"\\n- [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}] {content}\\n"
        )

    return f"Saved to vault: {path}"

# -----------------------------
# Timer
# -----------------------------

def set_timer(seconds: float, label: str = "Timer") -> str:
    def alert():
        time.sleep(seconds)
        print(f"\\n[TIMER DONE] {label} — {seconds} seconds is up!\\n")

    threading.Thread(
        target=alert,
        daemon=True
    ).start()

    return f"Timer started: {label} for {seconds} second(s)."

# -----------------------------
# Time
# -----------------------------

def get_current_time() -> str:
    return datetime.datetime.now().strftime(
        "%A, %B %d, %Y — %I:%M %p"
    )

# -----------------------------
# Web
# -----------------------------

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        output = []

        for r in results:
            output.append(
                f"{r['title']}\\n{r['body']}\\n{r['href']}"
            )

        return "\\n\\n".join(output)

    except Exception as e:
        return f"Search error: {e}"

# -----------------------------
# Tool schema
# -----------------------------

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Get basic info about the user's system: OS, disk usage",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a given directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Save a note into the user's Obsidian vault ONLY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string"
                    },
                    "filename": {
                        "type": "string"
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Start a countdown timer in seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number"
                    },
                    "label": {
                        "type": "string"
                    }
                },
                "required": ["seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "system_info": system_info,
    "list_files": list_files,
    "read_file": read_file,
    "write_note": write_note,
    "set_timer": set_timer,
    "get_current_time": get_current_time,
    "web_search": web_search,
}
