# CalHacks Chatbot - Electron App

A desktop chatbot application that connects to the CalHacks vLLM endpoint.

## Setup

1. Install dependencies:
```bash
npm install
```

## Running the App

Start the application:
```bash
npm start
```

For development with logging enabled:
```bash
npm run dev
```

## Features

- Clean, modern chat interface
- Connects to vLLM endpoint at `https://calhacks-monitor-vllm.ngrok.pizza`
- Small, dialog-style window (400x600px)
- Automatic model selection (uses latest model)
- Loading indicators and error handling

## Architecture

- `main.js` - Electron main process, handles API calls
- `preload.js` - Secure IPC bridge between main and renderer
- `renderer.js` - UI logic and event handling
- `index.html` - Chat interface HTML
- `styles.css` - Modern, gradient-themed styling

