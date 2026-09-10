from __future__ import annotations

import html
import json
import os
import base64
from pathlib import Path
from typing import Any

import gradio as gr

from api_client import ResearchAPIError, api_url, ingest_pdf, list_papers, query_research_api

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
LOCAL_PAPERS = ROOT.parent / "backend" / "data" / "artifacts" / "papers.json"
PAGES = ["home", "research", "library", "upload"]


def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def normalize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for i, item in enumerate(items):
        authors = item.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        authors = [a.get("name", "") if isinstance(a, dict) else str(a) for a in authors]
        result.append({
            "paper_id": str(item.get("paper_id") or item.get("paperId") or item.get("id") or f"paper-{i}"),
            "title": str(item.get("title") or "Untitled paper"),
            "authors": authors,
            "year": item.get("year") or "—",
            "url": item.get("url") or item.get("pdf_url") or item.get("source_url") or "",
        })
    return sorted(result, key=lambda p: p["title"].lower())


def load_papers() -> list[dict[str, Any]]:
    try:
        return normalize(list_papers())
    except ResearchAPIError:
        try:
            return normalize(json.loads(LOCAL_PAPERS.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return []


PAPERS = load_papers()


def asset(name: str) -> str:
    path = ASSETS / name
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def count_label() -> str:
    return f"{len(PAPERS)} indexed paper{'s' if len(PAPERS) != 1 else ''}"


def choices() -> list[tuple[str, str]]:
    return [(p["title"], p["paper_id"]) for p in PAPERS]


def sidebar_html() -> str:
    if not PAPERS:
        return "<div class='side-empty'>Your indexed papers will appear here.</div>"
    rows = "".join(f"<div class='side-paper'><i></i><span>{esc(p['title'])}</span></div>" for p in PAPERS[:6])
    more = f"<div class='side-more'>+ {len(PAPERS)-6} more in Library</div>" if len(PAPERS) > 6 else ""
    return rows + more


def cards(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<div class='empty'><b>No papers match that search.</b><span>Try a different title, author, or topic.</span></div>"
    out = []
    for p in items:
        authors = ", ".join(p["authors"][:2]) or "Academic research corpus"
        url = api_url(str(p["url"]))
        link = f"<a href='{esc(url)}' target='_blank' rel='noopener'>Open source ↗</a>" if url else "<span>Source link unavailable</span>"
        out.append(f"<article class='paper-card'><div class='paper-top'><b>RESEARCH PAPER</b><span>{esc(p['year'])}</span></div><h3>{esc(p['title'])}</h3><p>{esc(authors)}</p><div class='paper-foot'><span>Indexed in corpus</span>{link}</div></article>")
    return "<div class='paper-grid'>" + "".join(out) + "</div>"


def nav(page: str):
    return tuple(gr.update(visible=name == page) for name in PAGES)


def library_search(query: str):
    q = (query or "").lower().strip()
    visible = [p for p in PAPERS if q in p["title"].lower() or q in " ".join(p["authors"]).lower()]
    return cards(visible), f"Showing {len(visible)} of {len(PAPERS)} papers"


def ask(question: str, history: list[dict[str, str]], scope: str, paper_id: str):
    if not question or not question.strip():
        raise gr.Error("Write a research question before sending.")
    if scope == "paper" and not paper_id:
        raise gr.Error("Select a paper for paper-specific questions.")
    try:
        answer, sources = query_research_api(question.strip(), scope, paper_id if scope == "paper" else None)
    except ResearchAPIError as exc:
        raise gr.Error(str(exc)) from exc
    history = list(history or []) + [{"role": "user", "content": question.strip()}, {"role": "assistant", "content": answer}]
    source_lines, trace_rows = [], []
    for s in sources:
        tag, title = esc(s.get("source") or "S"), esc(s.get("title") or "Untitled paper")
        meta = " · ".join(str(x) for x in [s.get("year"), f"p.{s.get('page')}" if s.get("page") else None, s.get("section")] if x)
        url = api_url(str(s.get("url") or s.get("pdf_url") or ""))
        link = f" · [Open paper ↗]({esc(url)})" if url else ""
        source_lines.append(f"- **[{tag}] {title}** — {esc(meta)}{link}")
        trace_rows.append(f"<div class='evidence'><b>{tag}</b><span><strong>{title}</strong><small>{esc(s.get('section') or 'Retrieved evidence')} · page {esc(s.get('page') or '—')}</small></span></div>")
    trace = "".join(trace_rows) or "<div class='trace-empty'>No evidence was retrieved.</div>"
    return history, "\n".join(source_lines) or "No evidence was returned for this question.", trace, ""


def clear_chat():
    return [], "Sources will appear here after you ask a question.", "<div class='trace-empty'>Evidence will appear after your first question.</div>", ""


def ingest(path: str | None):
    if not path:
        raise gr.Error("Choose a PDF before uploading.")
    try:
        result = ingest_pdf(path)
    except ResearchAPIError as exc:
        return f"<div class='status error'><b>Upload could not be completed</b><span>{esc(exc)}</span></div>"
    if result.get("already_indexed"):
        msg = result.get("message") or "This paper already exists in your library."
        return f"<div class='status duplicate'><b>Already in your library</b><span>{esc(msg)}</span></div>"
    msg = result.get("message") or "The paper was indexed and is ready for research questions."
    if result.get("ok", True):
        return f"<div class='status success'><b>Paper approved and indexed</b><span>{esc(msg)}</span></div>"
    return f"<div class='status error'><b>Ingestion needs attention</b><span>{esc(msg)}</span></div>"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');
:root{--ink:#18211f;--muted:#6b7773;--line:#dfe5df;--soft:#f5f7f3;--paper:#fff;--green:#235d4a;--deep:#184939;--mint:#e6f1eb;--gold:#b98845;--shadow:0 20px 60px rgba(31,49,41,.08)}*{box-sizing:border-box}body{margin:0;background:#f9faf8;color:var(--ink);font-family:'DM Sans',sans-serif}footer{display:none!important}.gradio-container{max-width:none!important;padding:0!important;background:#f9faf8!important}.app{min-height:100vh;display:flex}.rail{width:260px;background:var(--deep);color:#eef5ef;padding:28px 18px;display:flex;flex-direction:column;position:fixed;inset:0 auto 0 0;z-index:5}.brand{padding:5px 12px 34px;border-bottom:1px solid rgba(255,255,255,.15)}.wordmark{font-size:16px;font-weight:700;letter-spacing:-.03em}.wordmark span{color:#b6d8c4}.sub{font:11px 'DM Mono';color:#a8c4b4;margin-top:8px;letter-spacing:.06em}.nav{padding:28px 0}.nav-label,.eyebrow{font:10px 'DM Mono';text-transform:uppercase;letter-spacing:.14em;color:#8dae9e}.nav-label{padding:0 12px 12px}.nav-btn{border:0!important;background:transparent!important;color:#c2d8cc!important;width:100%;text-align:left!important;padding:12px!important;border-radius:8px!important;font-size:13px!important;margin:2px 0}.nav-btn:hover{background:rgba(255,255,255,.1)!important;color:white!important}.rail-bottom{margin-top:auto;padding:14px 12px 0;border-top:1px solid rgba(255,255,255,.15)}.rail-status{font-size:11px;color:#a8c4b4;display:flex;gap:8px;align-items:center}.dot{width:7px;height:7px;border-radius:50%;background:#83ca9b}.side-list{margin-top:15px}.side-paper{font-size:11px;color:#c8d9d0;display:flex;gap:8px;line-height:1.35;margin:12px 0}.side-paper i{width:5px;height:5px;border-radius:50%;background:#d9b476;margin-top:5px;flex:0 0 auto}.side-more{font:10px 'DM Mono';color:#84a698;margin-top:15px}.main{margin-left:260px;width:calc(100% - 260px);padding:0 6vw 60px}.topbar{height:88px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}.crumb,.count{font:10px 'DM Mono';color:var(--muted);text-transform:uppercase;letter-spacing:.12em}.page{max-width:1140px;margin:0 auto}.hero{display:grid;grid-template-columns:1.08fr .92fr;gap:7vw;align-items:center;padding:80px 0 92px}.hero h1{font:500 68px/1.02 Newsreader,serif;letter-spacing:-.045em;margin:18px 0 22px;max-width:650px}.hero h1 em{color:var(--green);font-style:normal}.hero p{font-size:16px;line-height:1.7;color:var(--muted);max-width:520px}.hero img{width:100%;height:430px;object-fit:cover;filter:saturate(.75)}.note{position:absolute;bottom:-18px;left:-24px;background:white;padding:14px 16px;box-shadow:var(--shadow);font:10px 'DM Mono';color:var(--muted);border-left:3px solid var(--gold)}.media{position:relative}.actions{display:flex;gap:12px;margin-top:32px}.btn{border-radius:7px!important;padding:13px 18px!important;font-weight:600!important;font-size:12px!important;border:1px solid var(--green)!important}.primary{background:var(--green)!important;color:white!important}.quiet{background:transparent!important;color:var(--green)!important;border-color:var(--line)!important}.features{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:25px 0 70px;border-top:1px solid var(--line)}.feature{padding:18px 12px 8px 0}.feature b,.paper-top b{font:10px 'DM Mono';color:var(--gold);letter-spacing:.1em}.feature h3{font-size:16px;margin:17px 0 7px}.feature p{font-size:12px;line-height:1.6;color:var(--muted);margin:0}.intro{padding:58px 0 38px}.intro h1{font:500 52px/1.05 Newsreader,serif;letter-spacing:-.04em;margin:13px 0}.intro p{color:var(--muted);max-width:650px;line-height:1.65;font-size:14px}.research-grid,.upload-grid{display:grid;grid-template-columns:1fr 320px;gap:22px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:22px;box-shadow:0 8px 30px rgba(25,44,35,.03)}.chatbot{min-height:470px!important;background:#fff!important;border:0!important}.chatbot .message{font-size:13px!important;line-height:1.65!important}.composer textarea{border:1px solid var(--line)!important;border-radius:7px!important;background:#fbfcfa!important;padding:15px!important}.scope-row{display:flex;align-items:end;gap:12px;margin-top:13px}.scope-radio label{font-size:11px!important}.scope-select{flex:1}.scope-select label{font-size:10px!important;color:var(--muted)!important}.sources h3{font:14px Newsreader,serif;margin:0 0 14px}.sources p,.sources li{font-size:11px;line-height:1.55;color:var(--muted)}.trace-title{font:10px 'DM Mono';text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:20px 0 12px;padding-top:16px;border-top:1px solid var(--line)}.evidence{display:flex;gap:9px;border-bottom:1px solid var(--line);padding:10px 0}.evidence>b{color:var(--green);font:11px 'DM Mono'}.evidence strong{font-size:11px;font-weight:600;display:block}.evidence small{font-size:10px;color:var(--muted)}.trace-empty{font-size:11px;color:var(--muted);line-height:1.5}.toolbar{display:flex;gap:12px;align-items:center;margin:8px 0 24px}.search input{border:1px solid var(--line)!important;background:#fff!important;border-radius:7px!important;padding:12px!important;font-size:12px!important}.meta{font:10px 'DM Mono';color:var(--muted);margin-left:auto}.paper-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.paper-card{background:#fff;border:1px solid var(--line);padding:20px;min-height:230px;display:flex;flex-direction:column;transition:.2s}.paper-card:hover{transform:translateY(-3px);box-shadow:var(--shadow);border-color:#b6cdbd}.paper-top{display:flex;justify-content:space-between}.paper-top span{font:11px 'DM Mono';color:var(--muted)}.paper-card h3{font:500 19px/1.25 Newsreader,serif;margin:26px 0 10px}.paper-card p{font-size:11px;color:var(--muted);line-height:1.45}.paper-foot{display:flex;justify-content:space-between;gap:8px;border-top:1px solid var(--line);margin-top:auto;padding-top:14px;font:10px 'DM Mono';color:var(--muted)}.paper-foot a{color:var(--green);text-decoration:none}.empty{border:1px dashed var(--line);padding:55px;text-align:center;display:flex;flex-direction:column;gap:8px;color:var(--muted);font-size:12px}.upload-drop{border:1px dashed #b8c9bd!important;background:#fbfcfa!important;border-radius:8px!important;min-height:240px}.upload-btn{width:100%;margin-top:12px}.status{display:flex;flex-direction:column;gap:4px;padding:16px;border-radius:7px;margin-top:15px;font-size:12px}.status span{color:var(--muted)}.success{background:#eaf5ed;border:1px solid #c6e2cc;color:#1d6a3b}.duplicate{background:#fbf3e5;border:1px solid #ead4a8;color:#8a6428}.error{background:#f9eaea;border:1px solid #ecc7c7;color:#8c3a3a}.steps{padding:0;margin:10px 0 0;list-style:none}.steps li{display:flex;gap:14px;border-bottom:1px solid var(--line);padding:17px 0;font-size:12px}.step-no{font:11px 'DM Mono';color:var(--gold)}.steps span{color:var(--muted);font-size:11px;display:block;margin-top:4px}.upload-note{margin-top:22px;padding:18px;background:var(--mint);font-size:11px;line-height:1.6;color:var(--deep)}
@media(max-width:900px){.rail{width:210px}.main{margin-left:210px;width:calc(100% - 210px);padding:0 4vw 40px}.hero{grid-template-columns:1fr;padding:55px 0}.hero h1{font-size:58px}.research-grid,.upload-grid{grid-template-columns:1fr}.paper-grid{grid-template-columns:repeat(2,1fr)}}
@media(min-width:901px){body.rail-collapsed .rail{width:78px;padding-left:12px;padding-right:12px}.rail-collapsed .main{margin-left:78px;width:calc(100% - 78px)}body.rail-collapsed .brand{padding-left:5px;padding-right:5px}.rail-collapsed .wordmark,.rail-collapsed .sub,.rail-collapsed .nav-label,.rail-collapsed .rail-status{font-size:0}.rail-collapsed .nav-btn{font-size:0!important;text-align:center!important;padding-left:0!important;padding-right:0!important}.rail-collapsed .nav-btn:first-of-type:before{content:'⌂';font-size:18px}.rail-collapsed .nav-btn:nth-of-type(2):before{content:'✦';font-size:16px}.rail-collapsed .nav-btn:nth-of-type(3):before{content:'▤';font-size:16px}.rail-collapsed .nav-btn:nth-of-type(4):before{content:'＋';font-size:18px}.rail-collapsed .brand:before{content:'a';display:grid;place-items:center;width:34px;height:34px;border:1px solid rgba(255,255,255,.25);border-radius:10px;font-weight:700;font-size:16px}.rail-collapsed .dot{display:none}}
@media(max-width:620px){.app{display:block}.rail{position:static;width:100%;padding:18px}.nav{display:flex;gap:4px;overflow:auto;padding:18px 0 0}.nav-label,.rail-bottom{display:none}.nav-btn{white-space:nowrap;width:auto;font-size:11px!important}.main{margin:0;width:100%;padding:0 18px 35px}.topbar{height:68px}.hero h1{font-size:47px}.hero img{height:290px}.features,.paper-grid{grid-template-columns:1fr}.intro{padding:38px 0 20px}.intro h1{font-size:42px}.toolbar{display:block}.meta{display:block;margin:12px 0}.actions{flex-wrap:wrap}}
"""


EXTRA_CSS = """
.chat-head{display:flex;align-items:center;justify-content:space-between;padding-bottom:15px;border-bottom:1px solid var(--line);margin-bottom:6px}.chat-head b{display:block;font-size:13px}.chat-head span{display:block;color:var(--muted);font-size:10px;margin-top:4px}.chat-model{font:9px 'DM Mono'!important;color:var(--green)!important;letter-spacing:.1em}.chatbot{border-radius:0!important}.composer textarea{font-size:13px!important;line-height:1.55!important;min-height:80px!important}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Academico Helper", css=CSS + EXTRA_CSS, theme=gr.themes.Base()) as demo:
        with gr.Row(elem_classes="app"):
            with gr.Column(elem_classes="rail", scale=0, min_width=220):
                gr.HTML("<div class='brand'><div class='wordmark'>academico <span>/ helper</span></div><div class='sub'>RESEARCH WORKSPACE</div></div><div class='nav'><div class='nav-label'>Workspace</div></div>")
                nav_home = gr.Button("Overview", elem_classes="nav-btn")
                nav_research = gr.Button("Ask the corpus", elem_classes="nav-btn")
                nav_library = gr.Button("Paper library", elem_classes="nav-btn")
                nav_upload = gr.Button("Add a paper", elem_classes="nav-btn")
                with gr.Column(elem_classes="rail-bottom"):
                    gr.HTML("<div class='rail-status'><span class='dot'></span> Retrieval system online</div>")
            with gr.Column(elem_classes="main", scale=1):
                with gr.Row(elem_classes="topbar"):
                    gr.HTML("<span class='crumb'>Academico / Workspace</span>")
                    gr.HTML(f"<span class='count'>{esc(count_label())}</span>")
                with gr.Column(elem_classes="page", visible=True) as home:
                    gr.HTML(f"<section class='hero'><div><div class='eyebrow'>A better way to read the literature</div><h1>Move from <em>searching</em> to understanding.</h1><p>Academico Helper gives your research questions a grounded starting point. Ask across your indexed papers, follow the evidence, and keep your reading workflow in one focused place.</p><div class='actions'></div></div><div class='media'><img src='{asset('research-desk.jpg')}' alt='Research notes and papers on a desk'><div class='note'>A calm interface for serious inquiry</div></div></section><section class='features'><div class='feature'><b>01 / ASK</b><h3>Question the corpus</h3><p>Get concise, source-aware answers instead of another page of search results.</p></div><div class='feature'><b>02 / TRACE</b><h3>See the evidence</h3><p>Every response stays connected to the papers and passages that informed it.</p></div><div class='feature'><b>03 / BUILD</b><h3>Grow your library</h3><p>Add your own PDFs and keep a research collection that gets more useful over time.</p></div></section>")
                    with gr.Row():
                        start = gr.Button("Start a research session  →", elem_classes="btn primary")
                        explore = gr.Button("Browse the library", elem_classes="btn quiet")
                with gr.Column(elem_classes="page", visible=False) as research:
                    gr.HTML("<div class='intro'><div class='eyebrow'>Research session</div><h1>Ask better questions.</h1><p>Grounded answers from the papers in your workspace. Start broad, then narrow the conversation to one paper when you need a closer reading.</p></div>")
                    with gr.Row(elem_classes="research-grid"):
                        with gr.Column(elem_classes="panel"):
                            gr.HTML("<div class='chat-head'><div><b>Research conversation</b><span>Grounded in your indexed papers</span></div><span class='chat-model'>RAG / GROUNDED</span></div>")
                            chatbot = gr.Chatbot(type="messages", show_label=False, elem_classes="chatbot")
                            question = gr.Textbox(show_label=False, placeholder="Ask anything about your papers…", lines=3, elem_classes="composer")
                            with gr.Row(elem_classes="scope-row"):
                                scope = gr.Radio([("Entire library", "all"), ("One paper", "paper")], value="all", show_label=False, elem_classes="scope-radio")
                                paper_select = gr.Dropdown(choices=choices(), label="Focus paper", interactive=True, elem_classes="scope-select")
                            with gr.Row():
                                ask_btn = gr.Button("Send question  →", elem_classes="btn primary")
                                clear_btn = gr.Button("Clear", elem_classes="btn quiet")
                        with gr.Column(elem_classes="panel sources"):
                            gr.HTML("<h3>Evidence trail</h3><p>Your retrieved sources will be collected here with the answer. Use them to return to the original paper whenever you need to verify a claim.</p><div class='trace-title'>Retrieved passages</div>")
                            sources = gr.Markdown("Sources will appear here after you ask a question.")
                            trace = gr.HTML("<div class='trace-empty'>Evidence will appear after your first question.</div>")
                with gr.Column(elem_classes="page", visible=False) as library:
                    gr.HTML("<div class='intro'><div class='eyebrow'>Corpus library</div><h1>A considered collection.</h1><p>Explore the papers currently available to Academico Helper. Search by title or author, then bring any paper into a focused research session.</p></div>")
                    with gr.Row(elem_classes="toolbar"):
                        library_search_box = gr.Textbox(show_label=False, placeholder="Search title or author…", elem_classes="search")
                        library_meta = gr.HTML(f"<span>{esc(count_label())}</span>", elem_classes="meta")
                    library_cards = gr.HTML(cards(PAPERS))
                    gr.HTML(f"<div class='panel' style='margin-top:22px;display:flex;gap:22px;align-items:center'><img src='{asset('library-shelves.jpg')}' alt='Library shelves' style='width:170px;height:110px;object-fit:cover'><div><div class='eyebrow'>Keep building</div><h3>Your next useful paper belongs here.</h3><p style='color:var(--muted);font-size:12px'>Upload a PDF to add it to the searchable knowledge base.</p></div></div>")
                with gr.Column(elem_classes="page", visible=False) as upload:
                    gr.HTML("<div class='intro'><div class='eyebrow'>Library ingestion</div><h1>Bring your own evidence.</h1><p>Upload a PDF and we will validate, extract, chunk, embed, and index it. Duplicate protection keeps your corpus clean.</p></div>")
                    with gr.Row(elem_classes="upload-grid"):
                        with gr.Column(elem_classes="panel"):
                            upload_file = gr.File(label="Research paper PDF", file_types=[".pdf"], type="filepath", elem_classes="upload-drop")
                            upload_btn = gr.Button("Upload and index paper  →", elem_classes="btn primary upload-btn")
                            upload_status = gr.HTML("<div class='status'><b>Ready for a paper</b><span>PDF files are checked before they enter your library.</span></div>")
                        with gr.Column(elem_classes="panel"):
                            gr.HTML("<div class='eyebrow'>What happens next</div><ol class='steps'><li><b class='step-no'>01</b><div><strong>Validate</strong><span>Confirm the file is a readable PDF.</span></div></li><li><b class='step-no'>02</b><div><strong>Extract</strong><span>Read metadata and document text.</span></div></li><li><b class='step-no'>03</b><div><strong>Index</strong><span>Make passages available to retrieval.</span></div></li><li><b class='step-no'>04</b><div><strong>Protect</strong><span>Detect existing papers before adding.</span></div></li></ol><div class='upload-note'>Already in the library? You will see a clear duplicate message instead of creating a second record.</div>")
        pages = [home, research, library, upload]
        for button, page in [(nav_home, "home"), (nav_research, "research"), (nav_library, "library"), (nav_upload, "upload"), (start, "research"), (explore, "library")]:
            button.click(lambda p=page: nav(p), outputs=pages)
        question.submit(ask, inputs=[question, chatbot, scope, paper_select], outputs=[chatbot, sources, trace, question])
        ask_btn.click(ask, inputs=[question, chatbot, scope, paper_select], outputs=[chatbot, sources, trace, question])
        clear_btn.click(clear_chat, outputs=[chatbot, sources, trace, question])
        scope.change(lambda value: gr.update(visible=value == "paper"), inputs=scope, outputs=paper_select)
        library_search_box.input(library_search, inputs=library_search_box, outputs=[library_cards, library_meta])
        upload_btn.click(ingest, inputs=upload_file, outputs=[upload_status])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")), allowed_paths=[str(ASSETS)])
