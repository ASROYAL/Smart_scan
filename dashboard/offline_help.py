"""Offline, source-backed documentation search. No model or network calls."""

import re
from difflib import get_close_matches
from pathlib import Path

STOP_WORDS = {
    "how", "do", "i", "the", "a", "is", "to", "what", "my", "can", "it",
    "why", "does", "not", "and", "of", "for", "this", "with", "please", "help",
}
ALIASES = {
    "modulenotfounderror": {"missing", "module", "import"},
    "permissionerror": {"permission", "error"},
    "venv": {"environment", "activation"},
    "moved": {"moving", "directory", "stale"},
    "install": {"installation", "setup"},
    "launch": {"dashboard", "setup"},
    "crash": {"error", "traceback"},
    "offline": {"privacy", "local"},
    "upload": {"recording", "format"},
    "json": {"metadata", "sidecar"},
    "dbm": {"calibration", "power"},
    "labels": {"annotations", "unavailable"},
}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def refine_query(document: str, query: str) -> tuple[str, list[str]]:
    """Bounded refinement: aliases, then conservative spelling correction.

    No retrieved text becomes instructions. Expansion is limited to two passes
    and a fixed support vocabulary; arbitrary model-generated queries are absent.
    """
    original = tokens(query[:300])
    expanded = set(original)
    trace = []
    for term in sorted(original):
        if term in ALIASES:
            expanded.update(ALIASES[term])
            trace.append(f"Recognized {term}: {', '.join(sorted(ALIASES[term]))}")
    vocabulary = tokens(document) | set(ALIASES)
    for term in sorted(original - vocabulary):
        if len(term) < 5:
            continue
        match = get_close_matches(term, sorted(vocabulary), n=1, cutoff=0.86)
        if match:
            expanded.add(match[0])
            expanded.update(ALIASES.get(match[0], set()))
            trace.append(f"Possible spelling match: {term} → {match[0]}")
    return " ".join(sorted(expanded)), trace


def search_help(document: str, query: str) -> list[tuple[str, str, int]]:
    """Return matching sections and their one-based source heading lines."""
    terms = tokens(query)
    if not terms:
        return []
    sections = []
    title, body, line_number = "", [], 1
    for number, line in enumerate(document.splitlines(), 1):
        if line.startswith("## "):
            if title:
                sections.append((title, "\n".join(body).strip(), line_number))
            title, body, line_number = line[3:], [], number
        elif title:
            body.append(line)
    if title:
        sections.append((title, "\n".join(body).strip(), line_number))

    def score(section):
        heading = set(re.findall(r"[a-z0-9]+", section[0].lower()))
        words = set(re.findall(r"[a-z0-9]+", section[1].lower()))
        return 3 * len(terms & heading) + len(terms & words)

    return sorted((s for s in sections if score(s)), key=score, reverse=True)[:4]


def answer_help(document: str, query: str) -> dict:
    """Retrieve exact evidence, then retry with a bounded expanded query."""
    query = query[:300]
    matches = search_help(document, query)
    direct_terms = tokens(query)
    covered = tokens(" ".join(s[0] + " " + s[1] for s in matches))
    trace = ["Searched the guide using the original question."]
    if not matches or not direct_terms.issubset(covered):
        expanded, changes = refine_query(document, query)
        if changes:
            matches = search_help(document, expanded)
            trace.extend(changes)
            trace.append("Searched again with the recognized support terms.")
    missing = sorted(direct_terms - tokens(" ".join(s[0] + " " + s[1] for s in matches)))
    return {"matches": matches, "trace": trace, "unmatched_terms": missing}


def render_help(document_path: Path) -> None:
    """Render help without executing any experiment."""
    import streamlit as st

    st.title("Offline help")
    st.caption("Search the local support guide. Answers are documentation excerpts, "
               "not generated claims. No cloud service or account is needed.")
    try:
        document = document_path.read_text(encoding="utf-8")
    except OSError:
        st.error("The local support guide could not be read. Restore docs/offline-help.md.")
        return
    query = st.text_input("What do you need help with?", max_chars=300,
                          placeholder="For example: permission error, recording format, units")
    if query.strip():
        result = answer_help(document, query)
        matches = result["matches"]
        if not matches:
            st.info("No matching section was found. Try a specific term or read the full guide below.")
        else:
            st.caption("These passages may help. Search relevance is not a verified diagnosis.")
        if result["unmatched_terms"]:
            st.caption("Terms not present in the retrieved passages: "
                       + ", ".join(result["unmatched_terms"]))
        with st.expander("How this answer was found"):
            for step in result["trace"]:
                st.write(step)
        for title, body, line in matches:
            with st.expander(title, expanded=True):
                st.markdown(body)
                st.caption(f"Source: docs/offline-help.md, line {line}")
        if matches:
            report = "# Support search\n\nQuestion: " + query + "\n\n"
            for title, body, line in matches:
                report += f"## {title}\n\n{body}\n\nSource: docs/offline-help.md:{line}\n\n"
            st.download_button("Save this answer with sources", report,
                               "support-answer.md", "text/markdown")
    with st.expander("Read the complete support guide"):
        st.markdown(document)
    st.download_button("Download support guide", document, "offline-help.md", "text/markdown")
