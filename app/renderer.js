const chatContainer = document.getElementById('chatContainer');
const messageForm = document.getElementById('messageForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const newChatButton = document.getElementById('newChatButton');
const viewDataButton = document.getElementById('viewDataButton');
const backToChatButton = document.getElementById('backToChatButton');
const refreshDataButton = document.getElementById('refreshDataButton');
const chatView = document.getElementById('chatView');
const dataView = document.getElementById('dataView');
const sheetTabs = document.getElementById('sheetTabs');
const tableContainer = document.getElementById('tableContainer');

let isProcessing = false;
let loraData = null;
let currentSheet = 0;

// Add message to chat
function addMessage(content, type) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  
  // Remove <think></think> tags and any stray </think> tags from display
  let displayContent = content.replace(/<think>[\s\S]*?<\/think>/gi, '');
  displayContent = displayContent.replace(/<\/think>/gi, '');
  // Clean up excessive whitespace and trim
  displayContent = displayContent.replace(/\s+/g, ' ').trim();
  
  messageDiv.textContent = displayContent;
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return messageDiv;
}

// Add loading indicator
function addLoadingMessage() {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message loading';
  messageDiv.innerHTML = 'Thinking<span class="loading-dots"></span>';
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return messageDiv;
}

// Handle form submission
messageForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const message = messageInput.value.trim();
  if (!message || isProcessing) return;
  
  // Clear input
  messageInput.value = '';
  
  // Add user message
  addMessage(message, 'user');
  
  // Disable input while processing
  isProcessing = true;
  sendButton.disabled = true;
  messageInput.disabled = true;
  
  // Add loading indicator
  const loadingMessage = addLoadingMessage();
  
  try {
    // Send message to main process
    const response = await window.electronAPI.sendMessage(message);
    
    // Remove loading indicator
    loadingMessage.remove();
    
    if (response.success) {
      addMessage(response.message, 'assistant');
    } else {
      addMessage(`Error: ${response.error}`, 'error');
    }
  } catch (error) {
    loadingMessage.remove();
    addMessage(`Failed to send message: ${error.message}`, 'error');
  } finally {
    // Re-enable input
    isProcessing = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
  }
});

// Handle new chat button
newChatButton.addEventListener('click', () => {
  // Clear all messages except welcome message
  chatContainer.innerHTML = `
    <div class="welcome-message">
      <p>👋 Hi! Ask me anything about your activities.</p>
    </div>
  `;
  
  // Clear and focus input
  messageInput.value = '';
  messageInput.focus();
});

// Switch to data view
viewDataButton.addEventListener('click', async () => {
  chatView.style.display = 'none';
  dataView.style.display = 'flex';
  
  // Auto-load data if not already loaded
  if (!loraData) {
    tableContainer.innerHTML = '<div class="loading-state"><p>Loading data...</p></div>';
    
    try {
      const response = await window.electronAPI.fetchLoraData();
      
      if (response.success) {
        loraData = response.data;
        renderSheetTabs();
        renderTable(0);
      } else {
        tableContainer.innerHTML = `<div class="error-state"><p>Error: ${response.error}</p></div>`;
      }
    } catch (error) {
      tableContainer.innerHTML = `<div class="error-state"><p>Failed to load data: ${error.message}</p></div>`;
    }
  }
});

// Switch back to chat view
backToChatButton.addEventListener('click', () => {
  dataView.style.display = 'none';
  chatView.style.display = 'flex';
  messageInput.focus();
});

// Refresh LoRA data
refreshDataButton.addEventListener('click', async () => {
  tableContainer.innerHTML = '<div class="loading-state"><p>Loading data...</p></div>';
  
  try {
    const response = await window.electronAPI.fetchLoraData();
    
    if (response.success) {
      loraData = response.data;
      renderSheetTabs();
      renderTable(0);
    } else {
      tableContainer.innerHTML = `<div class="error-state"><p>Error: ${response.error}</p></div>`;
    }
  } catch (error) {
    tableContainer.innerHTML = `<div class="error-state"><p>Failed to load data: ${error.message}</p></div>`;
  }
});

// Render sheet tabs
function renderSheetTabs() {
  if (!loraData || !Array.isArray(loraData)) {
    sheetTabs.innerHTML = '';
    return;
  }
  
  sheetTabs.innerHTML = '';
  loraData.forEach((sheet, index) => {
    const tab = document.createElement('button');
    tab.className = `sheet-tab ${index === currentSheet ? 'active' : ''}`;
    tab.textContent = `Batch ${index + 1}`;
    tab.addEventListener('click', () => {
      currentSheet = index;
      renderTable(index);
      // Update active tab
      document.querySelectorAll('.sheet-tab').forEach((t, i) => {
        t.className = `sheet-tab ${i === index ? 'active' : ''}`;
      });
    });
    sheetTabs.appendChild(tab);
  });
}

// Render table for a specific sheet
function renderTable(sheetIndex) {
  if (!loraData || !Array.isArray(loraData) || !loraData[sheetIndex]) {
    tableContainer.innerHTML = '<div class="error-state"><p>No data available for this sheet</p></div>';
    return;
  }
  
  const sheetData = loraData[sheetIndex];
  
  if (!Array.isArray(sheetData) || sheetData.length === 0) {
    tableContainer.innerHTML = '<div class="loading-state"><p>No data in this sheet</p></div>';
    return;
  }
  
  const table = document.createElement('table');
  table.className = 'data-table';
  
  // Create header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  const questionHeader = document.createElement('th');
  questionHeader.textContent = 'Question';
  const answerHeader = document.createElement('th');
  answerHeader.textContent = 'Answer';
  headerRow.appendChild(questionHeader);
  headerRow.appendChild(answerHeader);
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Create body
  const tbody = document.createElement('tbody');
  sheetData.forEach(item => {
    if (item && typeof item === 'object' && 'question' in item && 'answer' in item) {
      const row = document.createElement('tr');
      
      const questionCell = document.createElement('td');
      questionCell.textContent = item.question || '';
      questionCell.addEventListener('click', handleCellClick);
      row.appendChild(questionCell);
      
      const answerCell = document.createElement('td');
      answerCell.textContent = item.answer || '';
      answerCell.addEventListener('click', handleCellClick);
      row.appendChild(answerCell);
      
      tbody.appendChild(row);
    }
  });
  table.appendChild(tbody);
  
  tableContainer.innerHTML = '';
  tableContainer.appendChild(table);
}

// Handle cell selection with drag
let selectedCells = new Set();
let isSelecting = false;
let selectionStart = null;

function handleCellClick(event) {
  // Only handle single click if not dragging
  if (!isSelecting) {
    const cell = event.target;
    clearSelection();
    selectCell(cell);
  }
}

function handleMouseDown(event) {
  const cell = event.target;
  if (!cell || cell.tagName !== 'TD') return;
  
  isSelecting = true;
  selectionStart = cell;
  tableContainer.classList.add('selecting');
  
  clearSelection();
  selectCell(cell);
  
  event.preventDefault();
}

function handleMouseMove(event) {
  if (!isSelecting || !selectionStart) return;
  
  const cell = event.target;
  if (!cell || cell.tagName !== 'TD') return;
  
  // Get the table and find row/col indices
  const table = cell.closest('table');
  if (!table) return;
  
  const startRow = selectionStart.parentElement.rowIndex;
  const startCol = selectionStart.cellIndex;
  const endRow = cell.parentElement.rowIndex;
  const endCol = cell.cellIndex;
  
  // Calculate selection range
  const minRow = Math.min(startRow, endRow);
  const maxRow = Math.max(startRow, endRow);
  const minCol = Math.min(startCol, endCol);
  const maxCol = Math.max(startCol, endCol);
  
  // Clear and reselect
  clearSelection();
  
  // Select all cells in range
  const rows = table.querySelectorAll('tbody tr');
  rows.forEach((row, rowIdx) => {
    if (rowIdx >= minRow - 1 && rowIdx <= maxRow - 1) { // -1 because tbody doesn't include thead
      const cells = row.querySelectorAll('td');
      cells.forEach((c, colIdx) => {
        if (colIdx >= minCol && colIdx <= maxCol) {
          selectCell(c, {
            isTop: rowIdx === minRow - 1,
            isBottom: rowIdx === maxRow - 1,
            isLeft: colIdx === minCol,
            isRight: colIdx === maxCol
          });
        }
      });
    }
  });
}

function handleMouseUp() {
  if (isSelecting) {
    isSelecting = false;
    selectionStart = null;
    tableContainer.classList.remove('selecting');
  }
}

function selectCell(cell, borders = {}) {
  if (!cell) return;
  
  selectedCells.add(cell);
  cell.classList.add('selected');
  
  // Add border classes for selection outline
  if (borders.isTop) cell.classList.add('selection-top');
  if (borders.isBottom) cell.classList.add('selection-bottom');
  if (borders.isLeft) cell.classList.add('selection-left');
  if (borders.isRight) cell.classList.add('selection-right');
}

function clearSelection() {
  selectedCells.forEach(cell => {
    cell.classList.remove('selected', 'selection-top', 'selection-bottom', 'selection-left', 'selection-right');
  });
  selectedCells.clear();
}

// Attach mouse events to table container
tableContainer.addEventListener('mousedown', (event) => {
  if (event.target.tagName === 'TD') {
    handleMouseDown(event);
  }
});

tableContainer.addEventListener('mousemove', handleMouseMove);
tableContainer.addEventListener('mouseup', handleMouseUp);
document.addEventListener('mouseup', handleMouseUp);

// Clear selection when clicking outside
document.addEventListener('click', (event) => {
  if (selectedCells.size > 0 && !event.target.closest('.data-table td')) {
    clearSelection();
  }
});

// Focus input on load
messageInput.focus();

