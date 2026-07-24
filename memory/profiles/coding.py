"""Coding domain profile (Step 3)."""
from memory.profiles.registry import Profile

CODING_PROFILE = Profile(
    name="coding",
    match_processes=frozenset({
        "code.exe",          # VS Code
        "pycharm64.exe", "pycharm.exe",
        "idea64.exe", "webstorm64.exe", "goland64.exe", "clion64.exe",
        "devenv.exe",        # Visual Studio
        "sublime_text.exe", "cursor.exe", "windsurf.exe",
        "windowsterminal.exe", "wt.exe", "powershell.exe", "cmd.exe",
    }),
    match_title_keywords=(
        "visual studio code", "vscode", "pycharm", "intellij", "webstorm",
        "cursor", ".py", ".js", ".ts", ".rs", ".go", ".java", ".cpp",
        " - vim", "powershell", "terminal",
    ),
    entity_types=(
        "file", "function", "class", "method", "variable", "module",
        "library", "framework", "package", "api", "endpoint", "database",
        "table", "error", "exception", "command", "repository", "project",
        "language", "tool",
    ),
    focus=(
        "This is a CODING session. Name concrete code entities precisely: "
        "specific file names (e.g. screen.py), function/method/class names, "
        "libraries/packages, APIs/endpoints, error or exception types, shell "
        "commands, and the project/repository. NEVER use vague entities like "
        "\"code\", \"script\", \"screen\", or \"editor\". Capture what the user "
        "is building or debugging and any errors visible."
    ),
)
