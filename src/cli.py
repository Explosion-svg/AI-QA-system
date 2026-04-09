"""
cli.py —— 命令行问答工具
=========================
复用主服务容器，提供一个轻量 CLI 入口。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.container import get_container
from src.infra.config import DEFAULT_MODEL, DEFAULT_PROVIDER, KNOWLEDGE_DIR, PROVIDERS
from src.infra.logger import setup_logger
from src.memory.history_manager import HistoryManager

setup_logger()

app = typer.Typer(help="RAG 问答系统 CLI")
console = Console()


def _show_help() -> None:
    table = Table(title="内置命令", show_header=True, header_style="bold magenta")
    table.add_column("命令", style="cyan", no_wrap=True)
    table.add_column("说明")
    commands = [
        ("/switch <provider> <model>", "切换模型提供商和模型"),
        ("/rag on|off", "开启或关闭 RAG"),
        ("/rag build [路径...]", "从知识库目录或指定路径重建索引"),
        ("/rag add <路径...>", "增量加入文档"),
        ("/history", "查看当前会话历史"),
        ("/status", "查看当前状态"),
        ("/new", "开始新会话"),
        ("/clear", "清空当前会话历史"),
        ("/models", "列出模型"),
        ("/help", "显示帮助"),
        ("/exit", "退出"),
    ]
    for command, description in commands:
        table.add_row(command, description)
    console.print(table)


def _collect_supported_paths(paths: list[str], container) -> list[str]:
    document_loader = container.document_loader()
    file_paths: list[str] = []

    if not paths:
        root = Path(KNOWLEDGE_DIR)
        if root.exists():
            for candidate in root.iterdir():
                if candidate.is_file() and document_loader.is_supported(candidate.name):
                    file_paths.append(str(candidate))
        return file_paths

    for item in paths:
        candidate = Path(item)
        if candidate.is_file() and document_loader.is_supported(candidate.name):
            file_paths.append(str(candidate))
        elif candidate.is_dir():
            for sub_path in candidate.rglob("*"):
                if sub_path.is_file() and document_loader.is_supported(sub_path.name):
                    file_paths.append(str(sub_path))
    return file_paths


@app.command()
def chat(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    rag: bool = typer.Option(True, "--rag/--no-rag"),
    session: Optional[str] = typer.Option(None, "--session", "-s"),
):
    """启动交互式命令行问答。"""
    console.print(
        Panel(
            "RAG 知识库问答 CLI\n输入 /help 查看命令",
            title="AI QA",
            expand=False,
        )
    )

    container = get_container()
    asyncio.run(container.startup())
    chat_service = container.chat_service()
    rag_engine = container.rag_engine()
    history_manager = container.history_manager()

    session_id = session or HistoryManager.new_session_id()
    use_rag = rag

    try:
        _show_help()
        while True:
            user_input = console.input("[bold green]你[/bold green]: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split()
                command = parts[0].lower()

                if command == "/exit":
                    break
                if command == "/help":
                    _show_help()
                    continue
                if command == "/clear":
                    history_manager.delete(session_id)
                    console.print("[dim]当前会话历史已清空[/dim]")
                    continue
                if command == "/new":
                    session_id = HistoryManager.new_session_id()
                    console.print(f"[dim]已切换新会话: {session_id}[/dim]")
                    continue
                if command == "/history":
                    history = history_manager.load(session_id)
                    if not history:
                        console.print("[dim]暂无历史消息[/dim]")
                    else:
                        for message in history[-10:]:
                            label = "你" if message["role"] == "user" else "AI"
                            console.print(f"[cyan]{label}[/cyan]: {message['content'][:120]}")
                    continue
                if command == "/switch" and len(parts) >= 3:
                    provider, model = parts[1], parts[2]
                    console.print(f"[dim]已切换到 {provider} / {model}[/dim]")
                    continue
                if command == "/models":
                    for provider_name, config in PROVIDERS.items():
                        console.print(
                            f"{config['icon']} [cyan]{provider_name}[/cyan]: "
                            + ", ".join(config["models"])
                        )
                    continue
                if command == "/status":
                    console.print(
                        Panel(
                            f"provider: {provider}\n"
                            f"model: {model}\n"
                            f"session: {session_id}\n"
                            f"rag: {'on' if use_rag else 'off'}\n"
                            f"rag_ready: {rag_engine.is_ready()}\n"
                            f"sources: {len(rag_engine.list_sources()) if rag_engine.is_ready() else 0}",
                            title="当前状态",
                        )
                    )
                    continue
                if command == "/rag":
                    sub_command = parts[1].lower() if len(parts) > 1 else ""
                    if sub_command == "on":
                        use_rag = True
                        console.print("[green]RAG 已开启[/green]")
                    elif sub_command == "off":
                        use_rag = False
                        console.print("[yellow]RAG 已关闭[/yellow]")
                    elif sub_command in {"build", "add"}:
                        file_paths = _collect_supported_paths(parts[2:], container)
                        if not file_paths:
                            console.print("[yellow]未找到可索引文档[/yellow]")
                            continue
                        if sub_command == "build":
                            rag_engine.clear_index()
                        chunk_count = rag_engine.build_index(file_paths)
                        use_rag = True
                        console.print(f"[green]索引完成，新增 {chunk_count} 个 chunks[/green]")
                    else:
                        console.print("[yellow]用法: /rag on|off|build [路径...]|add <路径...>[/yellow]")
                    continue

                console.print("[yellow]未知命令，输入 /help 查看帮助[/yellow]")
                continue

            answer, sources = asyncio.run(
                chat_service.chat(
                    message=user_input,
                    session_id=session_id,
                    use_rag=use_rag,
                    provider=provider,
                    model=model,
                )
            )
            console.print(f"[bold blue]AI[/bold blue]: {answer}")
            if sources:
                console.print("[dim]来源: " + "、".join(sources) + "[/dim]")
    finally:
        asyncio.run(container.shutdown())


if __name__ == "__main__":
    app()
