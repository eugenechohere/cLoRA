const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const axios = require('axios');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 400,
    height: 600,
    resizable: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#1a1a1a',
    titleBarStyle: 'hiddenInset',
    frame: true,
    title: 'cLoRa'
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in development
  // mainWindow.webContents.openDevTools();
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Handle fetching LoRA data
ipcMain.handle('fetch-lora-data', async () => {
  try {
    const response = await axios.get('https://calhacks-monitor-backend.ngrok.pizza/get_data', {
      params: {
        samples_per_batch: 20
      }
    });
    return {
      success: true,
      data: response.data
    };
  } catch (error) {
    console.error('Error fetching LoRA data:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// Handle chat message from renderer
ipcMain.handle('send-message', async (event, message) => {
  try {
    // Get the latest model
    const modelsResponse = await axios.get('https://calhacks-monitor-vllm.ngrok.pizza/v1/models');
    const models = modelsResponse.data.data[modelsResponse.data.data.length - 1].id;

    // Send chat completion request
    const response = await axios.post(
      'https://calhacks-monitor-vllm.ngrok.pizza/v1/chat/completions',
      {
        model: models,
        messages: [
          { role: 'user', content: message }
        ],
        extra_body: { 
          chat_template_kwargs: { 
            enable_thinking: false 
          } 
        },
        temperature: 0.05,
        top_p: 0.95,
        max_tokens: 1000
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer EMPTY'
        }
      }
    );

    return {
      success: true,
      message: response.data.choices[0].message.content
    };
  } catch (error) {
    console.error('Error sending message:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

