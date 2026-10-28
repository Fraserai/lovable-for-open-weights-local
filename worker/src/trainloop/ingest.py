"""Turn uploaded documents into text chunks the example generator can work from."""

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".html", ".md", ".txt"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def find_documents(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES))
        elif p.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(p)
        else:
            raise ValueError(f"Unsupported file type: {p}")
    return found


def to_markdown(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    from docling.document_converter import DocumentConverter  # heavy import, load on demand

    return DocumentConverter().convert(path).document.export_to_markdown()


def chunk_markdown(markdown: str, source: str, max_chars: int = 6000, min_chars: int = 200) -> list[Chunk]:
    """Split on headings, then pack paragraphs up to max_chars. Tiny fragments are dropped."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        buf = ""
        for para in paragraphs:
            while len(para) > max_chars:  # a single oversized paragraph (e.g. a big table)
                if buf:
                    chunks.append(Chunk(source, heading, buf))
                    buf = ""
                chunks.append(Chunk(source, heading, para[:max_chars]))
                para = para[max_chars:]
            if buf and len(buf) + len(para) + 2 > max_chars:
                chunks.append(Chunk(source, heading, buf))
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append(Chunk(source, heading, buf))

    return [c for c in chunks if len(c.text) >= min_chars]
