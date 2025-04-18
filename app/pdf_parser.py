# Uses PyMuPDF to extract and parse full-text from PDFs
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Verilen PDF dosyasındaki tüm metni birleştirerek döndürür.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text.append(page.get_text())

    doc.close()
    return "\n".join(full_text)


# Test (manuel)
if __name__ == "__main__":
    test_pdf_path = "example.pdf"
    try:
        text = extract_text_from_pdf(test_pdf_path)
        print(text[:1000])  # ilk 1000 karakteri göster
    except Exception as e:
        print(f"Hata: {e}")
