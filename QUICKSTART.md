# Quick Start Guide

## Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 16+** with npm
3. **Ollama** running locally with `llama3:8b` model

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Start the Backend Server

Open a terminal and run:

**Windows:**
```bash
python api_server.py
```

**Linux/Mac:**
```bash
python3 api_server.py
```

The API server will start on `http://localhost:8000`

### 4. Start the Frontend

Open a new terminal and run:

```bash
npm run dev
```

The React app will start on `http://localhost:3000`

## Using the Application

1. **Open your browser** and navigate to `http://localhost:3000`

2. **Upload Documents:**
   - Click "Manage Documents" in the sidebar
   - Click "Upload Document" button
   - Select a PDF, DOCX, TXT, or image file
   - Wait for indexing to complete

3. **Ask Questions:**
   - Click "Ask Questions" in the sidebar
   - Type your question in the input field
   - Press Enter or click the send button
   - Get answers based on your uploaded documents

4. **Switch Projects:**
   - Click the project dropdown in the sidebar
   - Select a different project to manage separate document sets

5. **Toggle Theme:**
   - Click the theme toggle button (moon/sun icon) in the sidebar
   - Switch between light and dark modes

## Troubleshooting

### Backend won't start
- Make sure Python dependencies are installed
- Check that port 8000 is not already in use
- Verify Ollama is running: `ollama serve`

### Frontend won't start
- Make sure Node.js dependencies are installed: `npm install`
- Check that port 3000 is not already in use

### Documents not uploading
- Check backend console for errors
- Verify file format is supported (PDF, DOCX, TXT, PNG, JPG, JPEG)
- Ensure Ollama is running with the correct model

### No answers from RAG
- Make sure documents are successfully indexed (check "Manage Documents")
- Verify Ollama is running: `http://localhost:11434`
- Check backend logs for errors

