import os
import re
import zipfile
from datetime import datetime


DOCX_OUT = "Project_Code_Documentation.docx"


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def w_p(text: str, style: str | None = None) -> str:
    # Normal paragraph.
    text = xml_escape(text)
    ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
    return f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"


def w_code_line(text: str) -> str:
    # One code line, monospace, preserves spaces.
    text = xml_escape(text.rstrip("\n"))
    return (
        "<w:p>"
        "<w:pPr><w:spacing w:before=\"0\" w:after=\"0\"/></w:pPr>"
        "<w:r>"
        "<w:rPr>"
        "<w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\" w:cs=\"Consolas\"/>"
        "<w:sz w:val=\"20\"/><w:szCs w:val=\"20\"/>"
        "</w:rPr>"
        f"<w:t xml:space=\"preserve\">{text}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def section_heading(text: str, level: int) -> str:
    # Use built-in heading styles.
    level = max(1, min(3, level))
    return w_p(text, style=f"Heading{level}")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def list_source_files(root: str) -> list[str]:
    keep_ext = {".py", ".md"}
    skip_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "datasets",
        "models",
        "outputs",
    }
    skip_files = {
        "Copy_of_Welcome_To_Colab.ipynb",
    }
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if name in skip_files:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in keep_ext:
                continue
            out.append(os.path.join(dirpath, name))
    out.sort(key=lambda p: p.lower())
    return out


def file_index(files: list[str], root: str) -> list[str]:
    # Short explanation for the major files/folders first, then list everything included.
    lines = []
    lines.append("Key files:")
    lines.append("- main.py: CLI entrypoint (train/predict)")
    lines.append("- app.py: Streamlit UI (demo + live evaluation)")
    lines.append("- src/: model + pipeline code (loader, ANN, GNN, ensemble, training)")
    lines.append("- docs/PROJECT_DOCUMENTATION.md: full project reference")
    lines.append("- docs/METHODOLOGY_AND_FORMULAS.md: methodology + formulas")
    lines.append("")
    lines.append("Included source files in this document:")
    for p in files:
        rel = os.path.relpath(p, root)
        lines.append(f"- {rel}")
    return lines


def build_document_xml(root: str) -> str:
    parts = []
    parts.append(section_heading("Money Laundering Detection Project", 1))
    parts.append(w_p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    parts.append(w_p("This document contains:"))
    parts.append(w_p("- Dataset/preprocessing chapter draft (Chapter 4)"))
    parts.append(w_p("- File index (what each file does)"))
    parts.append(w_p("- Full project source code (Python + Markdown)"))

    # Chapter 4 draft: use existing file if present, otherwise skip.
    chapter_file = os.path.join(root, "docs", "DATASETS_AND_PREPROCESSING.md")
    if os.path.exists(chapter_file):
        parts.append(section_heading("Chapter 4: Datasets, Preprocessing and Experimental Setup", 1))
        for line in read_text(chapter_file).splitlines():
            parts.append(w_p(line))

    files = list_source_files(root)

    parts.append(section_heading("File Index", 1))
    for line in file_index(files, root):
        parts.append(w_p(line))

    parts.append(section_heading("Source Code", 1))
    for path in files:
        rel = os.path.relpath(path, root)
        parts.append(section_heading(rel, 2))
        text = read_text(path)
        # Normalize line endings for display.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for ln in text.split("\n"):
            parts.append(w_code_line(ln))
        parts.append(w_p(""))  # spacer

    body = "".join(parts)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<w:body>{body}<w:sectPr/></w:body>"
        "</w:document>"
    )


def write_docx(root: str, out_path: str) -> None:
    document_xml = build_document_xml(root)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>
"""

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)


def main():
    root = os.path.abspath(os.getcwd())
    out_path = os.path.join(root, DOCX_OUT)
    write_docx(root, out_path)
    print(out_path)


if __name__ == "__main__":
    main()

