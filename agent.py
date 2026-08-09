#!/usr/bin/env python3
"""
Roxie - Fast, clean terminal assistant with semantic memory (RAG).
"""

import os
import chromadb
import ollama

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from tools import TOOL_SCHEMA, TOOL_FUNCTIONS


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHARACTER_FILE = os.path.join(BASE_DIR, "character.txt")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

CHAT_MODEL = "qwen2.5:3b-instruct"
EMBED_MODEL = "nomic-embed-text"

WRITE_TOOLS = {"write_note"}

console = Console()


CAT_LOGO = """[bold cyan]
 /\\_/\\\\
( o.o )
 > ^ <
/     \\\\
(       )
 `---`
[/bold cyan][bold magenta]    R O X I E[/bold magenta]"""


# -----------------------------
# Chroma
# -----------------------------

client = chromadb.PersistentClient(path=MEMORY_DIR)

conversation_memory = client.get_or_create_collection(
    "conversations"
)


# -----------------------------
# Profile
# -----------------------------

def load_profile_text():
    if not os.path.exists(CHARACTER_FILE):
        return "No custom profile configured."

    with open(CHARACTER_FILE, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Embeddings
# -----------------------------

def embed_text(text):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )

    return response["embedding"]


# -----------------------------
# Memory
# -----------------------------

def save_conversation_turn(user_input, reply):
    entry = f"User: {user_input}\nAssistant: {reply}"

    conversation_memory.add(
        ids=[f"conv_{conversation_memory.count()}"],
        documents=[entry],
        embeddings=[embed_text(entry)]
    )


def retrieve_relevant_memory(user_input, n=5):
    count = conversation_memory.count()

    if count == 0:
        return "No relevant memories."

    query_vector = embed_text(user_input)

    results = conversation_memory.query(
        query_embeddings=[query_vector],
        n_results=min(n, count),
        include=["documents"]
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return "No relevant memories."

    return "\n".join(docs)


# -----------------------------
# Confirmation
# -----------------------------

def confirm(fn_name, fn_args):
    console.print(
        f"\n[yellow][About to run] "
        f"{fn_name}({fn_args})[/yellow]"
    )

    answer = console.input(
        "[yellow]Run it? (y/n): [/yellow]"
    ).strip().lower()

    return answer == "y"


# -----------------------------
# Exit
# -----------------------------

EXIT_PHRASES = {
    "exit",
    "quit",
    "bye",
    "goodbye",
    "see you",
    "see ya",
    "later",
    "im done",
    "i'm done"
}


# -----------------------------
# Main
# -----------------------------

def run_agent():

    profile_content = load_profile_text()

    console.print(
        Panel(
            CAT_LOGO,
            border_style="cyan",
            expand=False
        )
    )

    console.print(
        "[dim]Type 'exit' to quit.[/dim]"
    )

    console.print()

    messages = []

    while True:

        try:
            user_input = console.input(
                "[bold green]You > [/bold green]"
            ).strip()

        except (KeyboardInterrupt, EOFError):

            console.print(
                "\n[dim]Roxie: goodbye.[/dim]"
            )

            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_PHRASES:

            console.print(
                "[dim]Roxie: goodbye.[/dim]"
            )

            break

        # Retrieve relevant long-term memory
        past_memory = retrieve_relevant_memory(
            user_input,
            n=5
        )

        # Build system prompt
        dynamic_system_prompt = (

            "You are Roxie, a local terminal AI assistant. "
            "Be helpful, concise, and direct. "
            "Do not modify the operating system.\n\n"

            "Use the profile and relevant memories below "
            "when useful. Do not invent memories.\n\n"

            "If the user asks about sports matches, schedules, "
            "scores, team statistics, news, prices, or current "
            "events, use web_search.\n\n"

            f"PROFILE:\n"
            f"{profile_content[:2500]}\n\n"

            f"RELEVANT MEMORIES:\n"
            f"{past_memory}"
        )

        current_messages = (
            [
                {
                    "role": "system",
                    "content": dynamic_system_prompt
                }
            ]
            + messages[-4:]
        )

        current_messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # -----------------------------
        # Ask model
        # -----------------------------

        with console.status(
            "[bold cyan]Roxie is thinking...[/bold cyan]",
            spinner="dots"
        ):

            response = ollama.chat(
                model=CHAT_MODEL,
                messages=current_messages,
                tools=TOOL_SCHEMA,
                options={
                    "temperature": 0.3
                }
            )

            msg = response["message"]

            # -----------------------------
            # Tool calls
            # -----------------------------

            if msg.get("tool_calls"):

                current_messages.append(msg)

                for call in msg["tool_calls"]:

                    fn_name = call["function"]["name"]
                    fn_args = call["function"]["arguments"]

                    fn = TOOL_FUNCTIONS.get(fn_name)

                    # Require confirmation for write tools
                    if fn_name in WRITE_TOOLS:

                        if not confirm(
                            fn_name,
                            fn_args
                        ):

                            current_messages.append(
                                {
                                    "role": "tool",
                                    "content": (
                                        "Cancelled by user."
                                    )
                                }
                            )

                            continue

                    result = (
                        fn(**fn_args)
                        if fn
                        else f"Unknown tool: {fn_name}"
                    )

                    current_messages.append(
                        {
                            "role": "tool",
                            "content": str(result)
                        }
                    )

                # Follow-up after tools
                followup = ollama.chat(
                    model=CHAT_MODEL,
                    messages=current_messages,
                    tools=TOOL_SCHEMA,
                    options={
                        "temperature": 0.3
                    }
                )

                reply = followup["message"]["content"]

            else:
                reply = msg["content"]

        # -----------------------------
        # Conversation history
        # -----------------------------

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": reply
            }
        )

        # Save semantic memory
        save_conversation_turn(
            user_input,
            reply
        )

        # -----------------------------
        # Display
        # -----------------------------

        console.print(
            "\n[bold magenta]Roxie:[/bold magenta]"
        )

        console.print(
            Markdown(reply)
        )

        console.print()


if __name__ == "__main__":
    run_agent()
