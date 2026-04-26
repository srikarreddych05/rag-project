import fitz  
import os

def parse_pdf(filepath):
    """Extracts text from PDF while preserving basic structure."""
    doc = fitz.open(filepath)
    filename = os.path.basename(filepath)
    pages_data = []
    
    for i, page in enumerate(doc):
        # "blocks" helps handle multi-column layouts better than just "text"
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position, then horizontal to maintain reading flow
        blocks.sort(key=lambda b: (b[1], b[0])) 
        
        page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
        
        pages_data.append({
            "text": page_text,
            "metadata": {
                "page": i + 1,
                "source": filename
            }
        })
    return pages_data

if __name__ == "__main__":
    # Test script on one file
    test_file = "data/your_first_paper.pdf"
    if os.path.exists(test_file):
        data = parse_pdf(test_file)
        print(f"Parsed {len(data)} pages from {test_file}")
        print(f"Sample text from page 1: {data[0]['text'][:200]}...")