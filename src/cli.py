"""
cli.py —— 命令行问答工具
==========================
这是「终端版」的入口文件，不需要浏览器，直接在命令行里和 AI 对话。

使用的库：
- typer：让函数参数自动变成命令行参数（--provider、--model 等）
- rich：让终端输出变得好看（彩色文字、表格、面板）

运行示例：
  python cli.py chat
  python cli.py chat --provider deepseek --model deepseek-chat
  python cli.py chat --provider ollama --model qwen2
  python cli.py chat --rag
"""

import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.infra.config import PROVIDERS, DEFAULT_PROVIDER, DEFAULT_MODEL, MAX_HISTORY, setup_logging
import logging

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)

from src.infra.llm_client import LLMClient
from src.memory.history_manager import HistoryManager
from src.rag import RAGEngine

# typer 应用实例
app = typer.Typer(help="AI 问答系统 CLI — 支持 OpenAI / DeepSeek / Qwen / Ollama")
console = Console()


def show_banner():
    """启动时显示欢迎横幅。"""
    console.print(Panel(
        "[bold cyan]AI 问答系统[/bold cyan]\n"
        "[dim]支持 OpenAI / DeepSeek / Qwen / Ollama | RAG 知识库 | 流式输出[/dim]",
        expand=False,
    ))


def print_help():
    """用表格打印所有内置命令。"""
    table = Table(title="内置命令", show_header=True, header_style="bold magenta")
    table.add_column("命令", style="cyan", no_wrap=True)
    table.add_column("说明")
    commands = [
        ("/switch <provider> <model>", "切换服务商和模型，如: /switch deepseek deepseek-chat"),
        ("/rag on",                   "开启 RAG 知识库检索"),
        ("/rag off",                  "关闭 RAG 知识库检索"),
        ("/rag build",                "重新构建知识库向量索引"),
        ("/rag add <文件路径>",        "把指定文件增量加入知识库"),
        ("/history",                  "查看当前对话历史"),
        ("/sessions",                 "列出所有已保存的历史会话"),
        ("/load <session_id>",        "加载一个历史会话继续对话"),
        ("/new",                      "开启新会话（自动保存当前会话）"),
        ("/clear",                    "清空当前对话历史"),
        ("/models",                   "列出所有服务商及其可用模型"),
        ("/status",                   "查看当前配置"),
        ("/help",                     "显示此帮助信息"),
        ("/exit",                     "保存对话并退出程序"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    console.print(table)


@app.command()
def chat(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p",
                                  help="服务商: openai / deepseek / qwen / ollama"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m",
                               help="模型名称"),
    rag: bool = typer.Option(False, "--rag", "-r",
                              help="启动时开启 RAG 知识库"),
    session: Optional[str] = typer.Option(None, "--session", "-s",
                                           help="加载指定历史会话 ID"),
    system_prompt: str = typer.Option(
        "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
        "--system", help="自定义系统提示词"
    ),
):
    """
    启动交互式对话（主命令）。
    直接运行 python cli.py chat 即可开始，所有参数均可选。
    """
    show_banner()
    print_help()

    # 初始化各模块
    client = LLMClient(provider=provider, model=model)
    history_mgr = HistoryManager()
    rag_engine = None
    use_rag = rag

    # 会话 ID 和历史消息
    session_id = session or HistoryManager.new_session_id()
    messages = history_mgr.load(session_id) if session else []

    # API Key 检查
    if not client.is_available():
        console.print(
            f"[yellow]警告: {provider} 未配置有效的 API Key，"
            f"请检查 .env 文件。[/yellow]"
        )

    # 启动时若指定 --rag，加载知识库
    if use_rag:
        rag_engine = RAGEngine()
        console.print("[dim]正在加载知识库索引...[/dim]")
        loaded = rag_engine.load_index()
        if not loaded:
            console.print("[yellow]未找到已有索引，请先放入文档并用 /rag build 构建。[/yellow]")

    # 显示配置摘要
    icon = PROVIDERS.get(provider, {}).get("icon", "")
    console.print(
        f"\n[bold]当前配置:[/bold] {icon} {provider} / {model}  "
        f"| 会话: [cyan]{session_id}[/cyan]  "
        f"| RAG: {'[green]开启[/green]' if use_rag else '[dim]关闭[/dim]'}\n"
    )

    # ==============================================================
    # 主对话循环
    # ==============================================================
    while True:
        try:
            user_input = console.input("[bold green]你[/bold green]: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]正在保存并退出...[/dim]")
            history_mgr.save(session_id, messages, {"provider": provider, "model": model})
            break

        if not user_input:
            continue

        # ---- 命令处理 ----
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/exit":
                history_mgr.save(session_id, messages, {"provider": provider, "model": model})
                console.print("[dim]会话已保存，再见！[/dim]")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/clear":
                messages.clear()
                console.print("[dim]对话历史已清空[/dim]")

            elif cmd == "/new":
                history_mgr.save(session_id, messages, {"provider": provider, "model": model})
                session_id = HistoryManager.new_session_id()
                messages = []
                console.print(f"[dim]新会话已开始: {session_id}[/dim]")

            elif cmd == "/history":
                if not messages:
                    console.print("[dim]暂无对话历史[/dim]")
                else:
                    for msg in messages[-10:]:
                        label = "[green]你[/green]" if msg["role"] == "user" else "[blue]AI[/blue]"
                        preview = msg["content"][:120].replace("\n", " ")
                        console.print(f"{label}: {preview}")

            elif cmd == "/sessions":
                sessions = history_mgr.list_sessions()
                if not sessions:
                    console.print("[dim]暂无历史会话[/dim]")
                else:
                    t = Table(title="历史会话")
                    t.add_column("Session ID", style="cyan")
                    for s in sessions:
                        t.add_row(s)
                    console.print(t)

            elif cmd == "/load" and len(parts) > 1:
                history_mgr.save(session_id, messages, {"provider": provider, "model": model})
                session_id = parts[1]
                messages = history_mgr.load(session_id)
                console.print(f"[dim]已加载会话 {session_id}，共 {len(messages)} 条消息[/dim]")

            elif cmd == "/switch" and len(parts) >= 3:
                provider, model = parts[1], parts[2]
                client.switch(provider, model)
                icon = PROVIDERS.get(provider, {}).get("icon", "")
                console.print(f"[dim]已切换至 {icon} {provider} / {model}[/dim]")

            elif cmd == "/models":
                for pname, pcfg in PROVIDERS.items():
                    console.print(f"{pcfg['icon']} [cyan]{pname}[/cyan]: {', '.join(pcfg['models'])}")

            elif cmd == "/status":
                console.print(Panel(
                    f"服务商:   {provider}\n模型:     {model}\n"
                    f"会话 ID:  {session_id}\nRAG:      {'开启' if use_rag else '关闭'}\n"
                    f"历史消息: {len(messages)} 条",
                    title="当前配置",
                ))

            elif cmd == "/rag":
                sub = parts[1] if len(parts) > 1 else ""
                if sub == "on":
                    use_rag = True
                    rag_engine = rag_engine or RAGEngine()
                    if not rag_engine.load_index():
                        console.print("[yellow]未找到索引，请先用 /rag build 构建。[/yellow]")
                    else:
                        console.print("[green]RAG 已开启[/green]")
                elif sub == "off":
                    use_rag = False
                    console.print("[dim]RAG 已关闭[/dim]")
                elif sub == "build":
                    rag_engine = rag_engine or RAGEngine()
                    rag_engine.clear_index()
                    count = rag_engine.build_index()
                    use_rag = True
                    console.print(f"[green]知识库构建完成，共 {count} 个文档块[/green]")
                elif sub == "add" and len(parts) > 2:
                    rag_engine = rag_engine or RAGEngine()
                    n = rag_engine.build_index(paths=parts[2:])
                    console.print(f"[green]已新增 {n} 个文档块[/green]")
                else:
                    console.print("用法: /rag on | off | build | add <路径>")
            else:
                console.print("[yellow]未知命令，输入 /help 查看帮助[/yellow]")
            continue

        # ---- RAG 检索 ----
        rag_context = ""
        if use_rag and rag_engine and rag_engine.is_ready():
            rag_context = rag_engine.get_context(user_input)
            if rag_context:
                console.print("[dim][RAG] 已检索到参考资料[/dim]")

        # ---- 构建发送消息 ----
        final_prompt = user_input
        if rag_context:
            final_prompt = (
                f"请根据以下参考资料回答问题，如资料不足可结合自身知识。\n\n"
                f"参考资料:\n{rag_context}\n\n问题: {user_input}"
            )

        # 滚动窗口，防止超 token
        if len(messages) >= MAX_HISTORY * 2:
            messages = messages[-(MAX_HISTORY * 2):]

        # 原始问题存历史
        messages.append({"role": "user", "content": user_input})

        # ---- 流式输出 ----
        console.print("[bold blue]AI[/bold blue]: ", end="")
        reply_parts = []
        try:
            send_msgs = messages[:-1] + [{"role": "user", "content": final_prompt}]
            for chunk in client.chat_stream(send_msgs, system_prompt=system_prompt):
                console.print(chunk, end="", highlight=False)
                reply_parts.append(chunk)
        except Exception as e:
            console.print(f"\n[red]请求出错: {e}[/red]")
            messages.pop()
            continue

        reply = "".join(reply_parts)
        console.print()
        messages.append({"role": "assistant", "content": reply})

        # 自动保存
        history_mgr.save(session_id, messages, {"provider": provider, "model": model})


if __name__ == "__main__":
    app()
