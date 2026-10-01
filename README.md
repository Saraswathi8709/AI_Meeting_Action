# 📝 Enterprise AI Meeting Action-Item Extractor

An intelligent Natural Language Processing (NLP) data extraction pipeline designed to transform unformatted multi-speaker meeting transcripts into verified, structured corporate action tasks.

## 🚀 Key Features
- **Token Segmentation Engine:** Automatically parses raw conversation strings into distinct Speaker and Speech token blocks.
- **Entity Regex Locator:** Automatically extracts deadline target dates formatted as `YYYY-MM-DD`.
- **Duplicate & Validation Safeguards:** Programmatically filters out duplicate spoken entries and flags missing task owners.
- **Accuracy Performance Matrix:** Includes a dedicated metric panel calculating Accuracy, Precision, Recall, and F1-Scores.
- **Production Upload Console:** Built using a two-column Streamlit interface supporting direct text parsing and file ingestion.

## 🛠️ Architecture & Tech Stack
- **Development Language:** Python 3.10+
- **Interface Component:** Streamlit Framework
- **Data Engineering:** Pandas & NumPy DataFrames
- **Statistical Analytics:** Scikit-Learn Validation Metrics

## ⚙️ How to Setup and Run Local Server

1. **Clone or open your project repository directory:**
   ```bash
   cd my_first_ai
   ```

2. **Install the required system dependency packages:**
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```

3. **Launch the web application module server:**
   ```bash
   streamlit run extractor.py
   ```

## 📊 Processing Pipeline View
The system ingests text files or manually pasted transcripts, applies NLP token cleaning steps to strip conversational noise, checks assertions via validation rules, and maps the structured data rows directly into an interactive exportable spreadsheet grid.
