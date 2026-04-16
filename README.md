
# 📄 Resume Classification and Matching System

##  How to Run

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
streamlit run app.py
```


##  Features

*  Upload multiple resumes
*  Supports **PDF** and **DOCX** formats
*  Enter job description
*  Semantic matching using pretrained NLP model
*  Resume ranking based on similarity score
*  Basic resume category classification
*  Export results as CSV


##  Recommended Model

* `sentence-transformers/all-MiniLM-L6-v2`


##  Future Improvements

*  Gmail integration
*  OCR support for scanned resumes
*  Advanced skill extraction
*  ATS (Applicant Tracking System) integration
*  Email notification system


