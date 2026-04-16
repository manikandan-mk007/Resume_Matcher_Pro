Resume Classification and Matching System

How to run:

1. Create virtual environment
   python -m venv venv

2. Activate environment
   Windows:
   venv\Scripts\activate

3. Install packages
   pip install -r requirements.txt

4. Run project
   streamlit run app.py

Features:
- Upload multiple resumes
- Supports PDF and DOCX
- Enter job description
- Semantic matching using pretrained NLP model
- Resume ranking by similarity score
- Basic resume category classification
- CSV export

Recommended model:
- sentence-transformers/all-MiniLM-L6-v2

Possible future improvements:
- Gmail integration
- OCR for scanned resumes
- Better skill extraction
- ATS integration
- Email notification