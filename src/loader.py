from pathlib import Path

import pypdf
import pdfplumber
import fitz


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    return text


def load_with_pypdf(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return clean_text("\n".join(pages))


def load_with_pdfplumber(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return clean_text("\n".join(pages))


def load_with_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text:
            pages.append(text)
    doc.close()
    return clean_text("\n".join(pages))


def load_pdf(pdf_path: str, loader: str = "pymupdf") -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if loader == "pymupdf":
        return load_with_pymupdf(pdf_path)
    elif loader == "pdfplumber":
        return load_with_pdfplumber(pdf_path)
    elif loader == "pypdf":
        return load_with_pypdf(pdf_path)
    else:
        raise ValueError(f"Unknown loader: {loader}")


# Keep alias for backwards compatibility with run_eval.py
def load_pdf_text(pdf_path: str, loader_type: str = "pymupdf") -> str:
    return load_pdf(pdf_path, loader=loader_type)


if __name__ == "__main__":
    path = "data/Identifying and Scaling AI Use Cases.pdf"
    for loader in ["pypdf", "pdfplumber", "pymupdf"]:
        text = load_pdf(path, loader=loader)
        print(f"\n--- {loader} (first 300 chars) ---")
        print(text[:300])


