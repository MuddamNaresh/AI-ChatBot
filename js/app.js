/**
 * AuraGen AI - Frontend Application Core
 * Handles chat interactions, streaming SSE, PDF uploads, session persistence,
 * settings, and interactive RAG citations.
 */

// Application State
const state = {
    currentSessionId: null,
    sessions: [],
    documents: [],
    activeDocument: null,
    chatMode: 'rag', // 'rag' or 'general'
    isStreaming: false,
    settings: {
        model: 'gemini-3.5-flash-lite',
        temperature: 0.7,
        top_k: 4,
        apiKey: ''
    }
};

// DOM Element References
const elements = {
    // Sidebar
    sidebar: document.getElementById('sidebar'),
    btnToggleSidebar: document.getElementById('btn-toggle-sidebar'),
    btnMobileMenu: document.getElementById('btn-mobile-menu'),
    btnNewChat: document.getElementById('btn-new-chat'),
    modeRag: document.getElementById('mode-rag'),
    modeGeneral: document.getElementById('mode-general'),
    btnLoadSample: document.getElementById('btn-load-sample'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('pdf-file-input'),
    uploadProgress: document.getElementById('upload-progress'),
    documentList: document.getElementById('document-list'),
    sessionList: document.getElementById('session-list'),
    apiStatusDot: document.getElementById('api-status-dot'),
    apiStatusText: document.getElementById('api-status-text'),
    btnOpenSettings: document.getElementById('btn-open-settings'),

    // Main Chat
    currentSessionTitle: document.getElementById('current-session-title'),
    activeDocBadge: document.getElementById('active-doc-badge'),
    chatMessages: document.getElementById('chat-messages'),
    welcomeScreen: document.getElementById('welcome-screen'),
    userInput: document.getElementById('user-input'),
    btnSend: document.getElementById('btn-send'),
    modeIndicator: document.getElementById('mode-indicator'),
    btnClearChat: document.getElementById('btn-clear-chat'),
    btnPreviewRag: document.getElementById('btn-preview-rag'),

    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    btnCloseSettings: document.getElementById('btn-close-settings'),
    btnCancelSettings: document.getElementById('btn-cancel-settings'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    settingApiKey: document.getElementById('setting-api-key'),
    btnToggleKeyVisibility: document.getElementById('btn-toggle-key-visibility'),
    btnVerifyKey: document.getElementById('btn-verify-key'),
    verifyStatus: document.getElementById('verify-status'),
    settingModel: document.getElementById('setting-model'),
    settingTemp: document.getElementById('setting-temperature'),
    tempVal: document.getElementById('temp-val'),
    settingTopK: document.getElementById('setting-top-k'),
    topKVal: document.getElementById('top-k-val'),

    // Inspector Modal
    inspectorModal: document.getElementById('inspector-modal'),
    btnCloseInspector: document.getElementById('btn-close-inspector'),
    inspectorQuery: document.getElementById('inspector-query'),
    btnRunInspect: document.getElementById('btn-run-inspect'),
    inspectorResults: document.getElementById('inspector-results'),

    // 13 ML Questions Modal
    btnMlQuestions: document.getElementById('btn-ml-questions'),
    mlModal: document.getElementById('ml-modal'),
    btnCloseMlModal: document.getElementById('btn-close-ml-modal')
};

// ==========================================
// 1. Initialization
// ==========================================

document.addEventListener('DOMContentLoaded', async () => {
    initEventListeners();
    await checkHealth();
    await loadDocuments();
    await loadSessions();
});

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        
        if (data.api_key_configured) {
            elements.apiStatusDot.className = 'status-dot online';
            elements.apiStatusText.textContent = `Gemini Online (${data.masked_key})`;
        } else {
            elements.apiStatusDot.className = 'status-dot';
            elements.apiStatusText.textContent = 'API Key Needed (⚙️)';
        }
    } catch (e) {
        elements.apiStatusDot.className = 'status-dot';
        elements.apiStatusText.textContent = 'Backend Offline';
    }
}

// ==========================================
// 2. Session Management
// ==========================================

async function loadSessions() {
    try {
        const res = await fetch('/api/sessions');
        state.sessions = await res.json();
        renderSessionList();

        if (state.sessions.length > 0) {
            // Select the most recent session
            await selectSession(state.sessions[0].id);
        } else {
            // Create a fresh session
            await createNewSession();
        }
    } catch (err) {
        console.error('Error loading sessions:', err);
    }
}

async function createNewSession(title = 'New Conversation') {
    try {
        const res = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: title,
                active_document: state.activeDocument
            })
        });
        const session = await res.json();
        state.sessions.unshift(session);
        renderSessionList();
        await selectSession(session.id);
    } catch (err) {
        console.error('Error creating new session:', err);
    }
}

async function selectSession(sessionId) {
    state.currentSessionId = sessionId;
    renderSessionList();

    try {
        const res = await fetch(`/api/sessions/${sessionId}`);
        const data = await res.json();
        
        elements.currentSessionTitle.textContent = data.session.title || 'Conversation';
        
        // Update active document if session had one
        if (data.session.active_document) {
            setActiveDocument(data.session.active_document);
        } else {
            updateActiveDocBadge();
        }

        renderMessages(data.messages || []);
    } catch (err) {
        console.error('Error selecting session:', err);
    }
}

async function deleteSession(e, sessionId) {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this conversation?')) return;

    try {
        await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        state.sessions = state.sessions.filter(s => s.id !== sessionId);
        renderSessionList();

        if (state.currentSessionId === sessionId) {
            if (state.sessions.length > 0) {
                await selectSession(state.sessions[0].id);
            } else {
                await createNewSession();
            }
        }
    } catch (err) {
        console.error('Error deleting session:', err);
    }
}

function renderSessionList() {
    elements.sessionList.innerHTML = '';
    
    if (state.sessions.length === 0) {
        elements.sessionList.innerHTML = '<div class="empty-state-sm">No previous chats</div>';
        return;
    }

    state.sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = `session-item ${session.id === state.currentSessionId ? 'active' : ''}`;
        item.onclick = () => selectSession(session.id);

        const title = document.createElement('span');
        title.className = 'session-title';
        title.textContent = session.title || 'New Chat';

        const delBtn = document.createElement('button');
        delBtn.className = 'session-delete';
        delBtn.innerHTML = '✕';
        delBtn.title = 'Delete chat';
        delBtn.onclick = (e) => deleteSession(e, session.id);

        item.appendChild(title);
        item.appendChild(delBtn);
        elements.sessionList.appendChild(item);
    });
}

// ==========================================
// 3. Document Management & RAG
// ==========================================

async function loadDocuments() {
    try {
        const res = await fetch('/api/documents');
        state.documents = await res.json();
        renderDocumentList();

        // If active document isn't set, select the first available
        if (!state.activeDocument && state.documents.length > 0) {
            setActiveDocument(state.documents[0].filename);
        } else {
            updateActiveDocBadge();
        }
    } catch (err) {
        console.error('Error loading documents:', err);
    }
}

function setActiveDocument(filename) {
    state.activeDocument = filename;
    updateActiveDocBadge();
    renderDocumentList();
    
    // Save to active session
    if (state.currentSessionId) {
        fetch(`/api/sessions/${state.currentSessionId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ active_document: filename })
        }).catch(err => console.error(err));
    }
}

function updateActiveDocBadge() {
    if (state.activeDocument) {
        elements.activeDocBadge.className = 'badge active-doc';
        elements.activeDocBadge.textContent = `Doc: ${state.activeDocument}`;
    } else {
        elements.activeDocBadge.className = 'badge';
        elements.activeDocBadge.textContent = 'No Document Attached';
    }
}

function renderDocumentList() {
    elements.documentList.innerHTML = '';
    
    if (state.documents.length === 0) {
        elements.documentList.innerHTML = '<div class="empty-state-sm">No PDF uploaded yet</div>';
        return;
    }

    state.documents.forEach(doc => {
        const item = document.createElement('div');
        item.className = `doc-item ${doc.filename === state.activeDocument ? 'selected' : ''}`;
        item.onclick = () => setActiveDocument(doc.filename);

        const info = document.createElement('div');
        info.className = 'doc-info';

        const name = document.createElement('span');
        name.className = 'doc-name';
        name.textContent = doc.filename;
        name.title = doc.filename;

        const meta = document.createElement('span');
        meta.className = 'doc-meta';
        meta.textContent = `${doc.total_pages} pgs • ${doc.total_chunks} chunks`;

        info.appendChild(name);
        info.appendChild(meta);

        const actions = document.createElement('div');
        actions.className = 'doc-actions';

        const delBtn = document.createElement('button');
        delBtn.className = 'btn-icon';
        delBtn.innerHTML = '🗑';
        delBtn.title = 'Remove document';
        delBtn.onclick = (e) => deleteDocument(e, doc.id);

        actions.appendChild(delBtn);

        item.appendChild(info);
        item.appendChild(actions);
        elements.documentList.appendChild(item);
    });
}

async function uploadFile(file) {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
        alert('Please upload a valid PDF document (.pdf).');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    elements.dropzone.querySelector('.dropzone-content').style.display = 'none';
    elements.uploadProgress.style.display = 'flex';
    document.getElementById('upload-status-text').textContent = `Indexing ${file.name}...`;

    try {
        const res = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        await loadDocuments();
        setActiveDocument(file.name);
    } catch (err) {
        alert(`Error indexing PDF: ${err.message}`);
    } finally {
        elements.dropzone.querySelector('.dropzone-content').style.display = 'flex';
        elements.uploadProgress.style.display = 'none';
        elements.fileInput.value = '';
    }
}

async function loadSampleDocument() {
    elements.dropzone.querySelector('.dropzone-content').style.display = 'none';
    elements.uploadProgress.style.display = 'flex';
    document.getElementById('upload-status-text').textContent = 'Loading Sample Handbook...';

    try {
        const res = await fetch('/api/documents/load-sample', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to load sample');

        await loadDocuments();
        setActiveDocument('Data_Science_Internship_Handbook.pdf');
    } catch (err) {
        alert(`Error loading sample: ${err.message}`);
    } finally {
        elements.dropzone.querySelector('.dropzone-content').style.display = 'flex';
        elements.uploadProgress.style.display = 'none';
    }
}

async function deleteDocument(e, docId) {
    e.stopPropagation();
    if (!confirm('Delete this document and its vector embeddings?')) return;

    try {
        await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
        if (state.activeDocument === state.documents.find(d => d.id === docId)?.filename) {
            state.activeDocument = null;
        }
        await loadDocuments();
    } catch (err) {
        console.error('Error deleting document:', err);
    }
}

// ==========================================
// 4. Chat Rendering & Streaming
// ==========================================

function renderMessages(messages) {
    elements.chatMessages.innerHTML = '';
    
    if (messages.length === 0) {
        elements.chatMessages.appendChild(elements.welcomeScreen);
        elements.welcomeScreen.style.display = 'block';
        return;
    }

    elements.welcomeScreen.style.display = 'none';
    messages.forEach(msg => {
        appendMessage(msg.role, msg.content, msg.sources, false);
    });
    scrollToBottom();
}

function appendMessage(role, content, sources = null, animate = true) {
    elements.welcomeScreen.style.display = 'none';

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = `avatar ${role === 'user' ? 'user' : 'ai'}`;
    avatar.innerHTML = role === 'user' ? '👤' : '✨';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = parseMarkdown(content);

    contentDiv.appendChild(bubble);

    // If assistant message includes source citations, append the citation card
    if (role === 'assistant' && sources && sources.length > 0) {
        const sourcesCard = createSourcesCard(sources);
        contentDiv.appendChild(sourcesCard);
    }

    if (role === 'user') {
        row.appendChild(contentDiv);
        row.appendChild(avatar);
    } else {
        row.appendChild(avatar);
        row.appendChild(contentDiv);
    }

    elements.chatMessages.appendChild(row);
    scrollToBottom();
    return bubble;
}

function createSourcesCard(sources) {
    const card = document.createElement('div');
    card.className = 'sources-card';

    const header = document.createElement('div');
    header.className = 'sources-header';
    header.innerHTML = `<span>📚 Retrieved Document Sources (${sources.length})</span> <span class="toggle-icon">▼</span>`;

    const list = document.createElement('div');
    list.className = 'sources-list';

    sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';

        const meta = document.createElement('div');
        meta.className = 'source-meta';
        meta.innerHTML = `
            <strong>${src.filename}</strong>
            <span class="badge-page">Page ${src.page}</span>
            <span class="badge-score">Similarity: ${Math.round(src.score * 100)}%</span>
        `;

        const text = document.createElement('div');
        text.className = 'source-text';
        text.textContent = `"${src.text_preview}"`;

        item.appendChild(meta);
        item.appendChild(text);
        list.appendChild(item);
    });

    // Toggle expand/collapse
    header.onclick = () => {
        const isCollapsed = list.style.display === 'none';
        list.style.display = isCollapsed ? 'flex' : 'none';
        header.querySelector('.toggle-icon').textContent = isCollapsed ? '▼' : '▶';
    };

    card.appendChild(header);
    card.appendChild(list);
    return card;
}

async function sendMessage() {
    const text = elements.userInput.value.trim();
    if (!text || state.isStreaming) return;

    if (!state.currentSessionId) {
        await createNewSession();
    }

    // Append user message immediately
    appendMessage('user', text);
    elements.userInput.value = '';
    adjustTextareaHeight(elements.userInput);
    
    state.isStreaming = true;
    elements.btnSend.disabled = true;

    // Create placeholder assistant bubble for streaming
    const assistantBubble = appendMessage('assistant', '');
    assistantBubble.innerHTML = '<span class="typing-indicator">Thinking...</span>';

    try {
        const payload = {
            session_id: state.currentSessionId,
            message: text,
            mode: state.chatMode,
            document_name: state.chatMode === 'rag' ? state.activeDocument : null,
            model: state.settings.model,
            temperature: state.settings.temperature,
            top_k: state.settings.top_k
        };

        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            // Fallback to non-streaming endpoint
            const fallbackRes = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await fallbackRes.json();
            assistantBubble.innerHTML = parseMarkdown(data.message.content);
            if (data.sources && data.sources.length > 0) {
                assistantBubble.parentElement.appendChild(createSourcesCard(data.sources));
            }
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let accumulated = '';
        let sources = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const parsed = JSON.parse(line.substring(6));
                        if (parsed.token) {
                            accumulated += parsed.token;
                            assistantBubble.innerHTML = parseMarkdown(accumulated);
                            scrollToBottom();
                        }
                        if (parsed.done && parsed.sources) {
                            sources = parsed.sources;
                        }
                    } catch (e) {
                        // Incomplete JSON chunk, will resolve on next loop
                    }
                }
            }
        }

        // Attach source citation card if available
        if (sources && sources.length > 0) {
            assistantBubble.parentElement.appendChild(createSourcesCard(sources));
        }

        // Refresh session list to update title if it changed
        const activeItem = state.sessions.find(s => s.id === state.currentSessionId);
        if (activeItem && activeItem.title === 'New Conversation') {
            await loadSessions();
        }

    } catch (err) {
        console.error('Chat generation error:', err);
        assistantBubble.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    } finally {
        state.isStreaming = false;
        elements.btnSend.disabled = false;
        elements.userInput.focus();
    }
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
}

// ==========================================
// 5. Lightweight Markdown Parser
// ==========================================

function parseMarkdown(text) {
    if (!text) return '';
    
    let html = text;

    // Escape HTML tags to prevent XSS
    html = html.replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Fenced Code blocks with language badge
    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><div class="code-header">${lang || 'code'}</div><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italics
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Bullet lists
    html = html.replace(/^\s*[-•*]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

    // Simple Tables
    html = html.replace(/\|(.+)\|/g, (match) => {
        const cells = match.split('|').filter(c => c.trim().length > 0);
        const isHeaderSep = cells.every(c => /^:?-+:?$/.test(c.trim()));
        if (isHeaderSep) return '';
        const tag = 'td';
        return '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
    });
    html = html.replace(/(<tr>.*<\/tr>)+/g, '<table>$0</table>');

    // Line breaks to paragraphs
    html = html.split('\n\n').map(p => {
        if (p.startsWith('<pre>') || p.startsWith('<ul>') || p.startsWith('<table>') || p.startsWith('<h')) {
            return p;
        }
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
}

// ==========================================
// 6. Event Listeners & Modals
// ==========================================

function initEventListeners() {
    // Mode toggling
    elements.modeRag.onclick = () => setMode('rag');
    elements.modeGeneral.onclick = () => setMode('general');

    // Send actions
    elements.btnSend.onclick = sendMessage;
    elements.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    elements.userInput.addEventListener('input', () => adjustTextareaHeight(elements.userInput));

    // New Chat
    elements.btnNewChat.onclick = () => createNewSession();

    // Clear Chat
    elements.btnClearChat.onclick = async () => {
        if (!confirm('Clear all messages in this conversation?')) return;
        if (state.currentSessionId) {
            await fetch(`/api/sessions/${state.currentSessionId}`, { method: 'DELETE' });
            await createNewSession();
        }
    };

    // Sample prompts & 13 ML Questions chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.onclick = () => {
            elements.userInput.value = chip.dataset.prompt;
            setMode('rag');
            sendMessage();
        };
    });

    // 13 ML Questions Modal triggers
    if (elements.btnMlQuestions) {
        elements.btnMlQuestions.onclick = () => {
            elements.mlModal.style.display = 'flex';
        };
    }
    if (elements.btnCloseMlModal) {
        elements.btnCloseMlModal.onclick = () => {
            elements.mlModal.style.display = 'none';
        };
    }
    document.querySelectorAll('.ml-modal-btn').forEach(btn => {
        btn.onclick = () => {
            elements.mlModal.style.display = 'none';
            elements.userInput.value = btn.dataset.prompt;
            setMode('rag');
            sendMessage();
        };
    });

    // Dropzone upload
    elements.dropzone.onclick = () => elements.fileInput.click();
    elements.fileInput.onchange = (e) => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    };

    elements.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropzone.classList.add('dragover');
    });
    elements.dropzone.addEventListener('dragleave', () => elements.dropzone.classList.remove('dragover'));
    elements.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });

    // Load Sample Button
    elements.btnLoadSample.onclick = loadSampleDocument;

    // Sidebar Mobile Toggle
    if (elements.btnToggleSidebar) {
        elements.btnToggleSidebar.onclick = () => elements.sidebar.classList.toggle('open');
    }
    if (elements.btnMobileMenu) {
        elements.btnMobileMenu.onclick = () => elements.sidebar.classList.toggle('open');
    }

    // Settings Modal
    elements.btnOpenSettings.onclick = openSettings;
    elements.btnCloseSettings.onclick = closeSettings;
    elements.btnCancelSettings.onclick = closeSettings;
    elements.btnSaveSettings.onclick = saveSettings;

    elements.btnToggleKeyVisibility.onclick = () => {
        const isPass = elements.settingApiKey.type === 'password';
        elements.settingApiKey.type = isPass ? 'text' : 'password';
        elements.btnToggleKeyVisibility.textContent = isPass ? '🔒' : '👁';
    };

    elements.btnVerifyKey.onclick = verifyApiKey;
    elements.settingTemp.oninput = (e) => elements.tempVal.textContent = e.target.value;
    elements.settingTopK.oninput = (e) => elements.topKVal.textContent = e.target.value;

    // Inspector Modal
    elements.btnPreviewRag.onclick = () => {
        elements.inspectorModal.style.display = 'flex';
        elements.inspectorQuery.focus();
    };
    elements.btnCloseInspector.onclick = () => elements.inspectorModal.style.display = 'none';
    elements.btnRunInspect.onclick = runRagInspection;
    elements.inspectorQuery.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runRagInspection();
    });
}

function setMode(mode) {
    state.chatMode = mode;
    elements.modeRag.classList.toggle('active', mode === 'rag');
    elements.modeGeneral.classList.toggle('active', mode === 'general');
    elements.modeIndicator.textContent = mode === 'rag' 
        ? `Mode: Document RAG (${state.activeDocument || 'All Documents'})` 
        : 'Mode: General AI Assistant';
}

function openSettings() {
    elements.settingsModal.style.display = 'flex';
    elements.settingModel.value = state.settings.model;
    elements.settingTemp.value = state.settings.temperature;
    elements.tempVal.textContent = state.settings.temperature;
    elements.settingTopK.value = state.settings.top_k;
    elements.topKVal.textContent = state.settings.top_k;
    elements.verifyStatus.textContent = '';
}

function closeSettings() {
    elements.settingsModal.style.display = 'none';
}

async function verifyApiKey() {
    const key = elements.settingApiKey.value.trim();
    if (!key) {
        elements.verifyStatus.className = 'verify-status error';
        elements.verifyStatus.textContent = 'Please enter a key.';
        return;
    }

    elements.verifyStatus.className = 'verify-status';
    elements.verifyStatus.textContent = 'Testing connection...';

    try {
        const res = await fetch('/api/settings/verify-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: key })
        });
        const data = await res.json();
        
        if (data.valid) {
            elements.verifyStatus.className = 'verify-status success';
            elements.verifyStatus.textContent = '✓ Key Verified!';
        } else {
            elements.verifyStatus.className = 'verify-status error';
            elements.verifyStatus.textContent = `✗ ${data.message}`;
        }
    } catch (e) {
        elements.verifyStatus.className = 'verify-status error';
        elements.verifyStatus.textContent = 'Error contacting server';
    }
}

async function saveSettings() {
    const key = elements.settingApiKey.value.trim();
    state.settings.model = elements.settingModel.value;
    state.settings.temperature = parseFloat(elements.settingTemp.value);
    state.settings.top_k = parseInt(elements.settingTopK.value, 10);

    if (key) {
        try {
            const res = await fetch('/api/settings/save-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: key })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Failed to save key');
            await checkHealth();
        } catch (e) {
            alert(`Error saving API Key: ${e.message}`);
            return;
        }
    }

    closeSettings();
}

async function runRagInspection() {
    const query = elements.inspectorQuery.value.trim();
    if (!query) return;

    elements.inspectorResults.innerHTML = '<div class="empty-state-sm">Searching vector space...</div>';

    try {
        const res = await fetch('/api/rag/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                document_name: state.activeDocument,
                top_k: state.settings.top_k
            })
        });

        const data = await res.json();
        elements.inspectorResults.innerHTML = '';

        if (!data.results || data.results.length === 0) {
            elements.inspectorResults.innerHTML = '<div class="empty-state-sm">No chunks found for this query.</div>';
            return;
        }

        data.results.forEach((r, idx) => {
            const card = document.createElement('div');
            card.className = 'source-item';
            card.innerHTML = `
                <div class="source-meta">
                    <strong>#${idx + 1} ${r.filename}</strong>
                    <span class="badge-page">Page ${r.page}</span>
                    <span class="badge-score">Similarity: ${Math.round(r.score * 100)}%</span>
                    <span class="doc-meta">[Method: ${r.method}]</span>
                </div>
                <div class="source-text">${r.text}</div>
            `;
            elements.inspectorResults.appendChild(card);
        });

    } catch (err) {
        elements.inspectorResults.innerHTML = `<div style="color: var(--danger)">Error: ${err.message}</div>`;
    }
}
