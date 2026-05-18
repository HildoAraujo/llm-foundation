from pathlib import Path

import pypdf 
import pdfplumber
import pymupdf
import fitz


def clean_text(text: str) -> str:
    
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    return text


def load_with_pymupdf(pdf_path: str) -> str:

    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text()

        if text:
            pages.append(text)

    doc.close()
    return clean_text("\n".join(pages))

def load_with_pdfplumber(pdf_path: str) -> str:

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:
            text = page.extract_text()

            if text: 
                pages.append(text)
    return clean_text ("\n".join(pages))    

def load_pdf_text(pdf_path: str, loader_type: str = "pymupdf") -> str:

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if loader_type == "pymupdf":
        return load_with_pymupdf(pdf_path)

    elif loader_type == "pdfplumber":
        return load_with_pdfplumber(pdf_path)

    else: 
        raise ValueError(
            f"Unsupported loader_type: {loader_type} "
        )                  
         

