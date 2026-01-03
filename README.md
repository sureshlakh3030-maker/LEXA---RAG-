# RAG System - React Frontend

A modern, responsive React frontend for the RAG (Retrieval-Augmented Generation) system with dark/light theme support and persistent chat history.

## Features

- 🎨 **Modern UI** - Clean interface matching the design with white panels, light grey background, and blue accents
- 🌓 **Dark/Light Theme** - Seamless theme switching with persistent preference
- 📱 **Fully Responsive** - Works on desktop, tablet, and mobile devices
- 💬 **Chat Interface** - Interactive Q&A with conversation history
- 💾 **Persistent Chat History** - All conversations are saved to SQLite database and persist across sessions
- 📄 **Document Management** - Upload and manage documents (PDF, DOCX, TXT, Images)
- 🔄 **Multi-Project Support** - Switch between different document projects
- ⚡ **FastAPI Backend** - RESTful API server connecting React to Python RAG system

## Project Structure

```
RAG/
├── src/
│   ├── components/
│   │   ├── Layout.jsx
│   │   ├── Sidebar.jsx
│   │   ├── ChatInterface.jsx
│   │   └── DocumentManagement.jsx
│   ├── contexts/
│   │   └── ThemeContext.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── api_server.py          # FastAPI backend
├── database.py            # SQLite database for chat persistence
├── package.json
├── vite.config.js
└── requirements.txt
```

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Start the FastAPI Backend Server

```bash
python api_server.py
```

The API server will run on `http://localhost:8000`
The database (`rag_chats.db`) will be automatically created on first run.

### 4. Start the React Development Server

```bash
npm run dev
```

The React app will run on `http://localhost:3000`

## Usage

1. **Upload Documents**: Go to "Manage Documents" and upload PDF, DOCX, TXT, or image files
2. **Ask Questions**: Switch to "Ask Questions" to chat with your documents
3. **Chat Persistence**: All conversations are automatically saved and restored when you switch tabs or projects
4. **Switch Projects**: Use the project dropdown in the sidebar to manage multiple document sets
5. **Toggle Theme**: Click the theme toggle button in the sidebar to switch between light and dark modes

## Database

The application uses SQLite (`rag_chats.db`) to store:
- **Conversations**: Each project can have multiple conversation sessions
- **Messages**: All messages are stored with timestamps and roles (user/assistant)
- **Automatic Persistence**: Conversations are saved automatically when messages are sent

## API Endpoints

### Chat Endpoints
- `POST /api/ask` - Ask a question (saves to database automatically)
- `GET /api/projects/{project_id}/active-conversation` - Get active conversation for a project
- `GET /api/projects/{project_id}/conversations` - List all conversations for a project
- `GET /api/conversations/{conversation_id}/messages` - Get messages for a conversation
- `POST /api/conversations/{conversation_id}/clear` - Clear all messages from a conversation
- `DELETE /api/conversations/{conversation_id}` - Delete a conversation

### Document Endpoints
- `POST /api/upload` - Upload a document
- `GET /api/projects/{project_id}/documents` - List documents in a project
- `GET /api/projects/{project_id}/stats` - Get project statistics

### Project Endpoints
- `GET /api/projects` - List all projects
- `POST /api/projects/switch` - Switch active project

## Color Theme

### Light Theme
- Background: `#f5f5f5` (Light grey)
- Panels: `#ffffff` (White)
- Accent: `#4a90e2` (Blue)
- Text: `#1a1a1a` (Dark grey)

### Dark Theme
- Background: `#1a1a1a` (Dark grey)
- Panels: `#2d2d2d` (Dark panel)
- Accent: `#5aa3f0` (Light blue)
- Text: `#ffffff` (White)

## Responsive Breakpoints

- Desktop: > 768px (Full sidebar and layout)
- Tablet: 481px - 768px (Compact sidebar, horizontal layout)
- Mobile: ≤ 480px (Minimal sidebar, vertical layout)

## Development

- React 18.2.0
- Vite 5.0.8
- FastAPI 0.104.1
- SQLite 3 (built-in Python)
- React Icons for iconography

## Notes

- Make sure Ollama is running on `http://localhost:11434` with the `llama3:8b` model
- The backend connects to the existing Python RAG system (`ask_questions.py` and `pdf_to_vectors.py`)
- Documents are stored in the `rag_data/{project_id}/` directory structure
- Chat conversations are stored in `rag_chats.db` SQLite database
- Chat history is automatically restored when switching between tabs or projects
