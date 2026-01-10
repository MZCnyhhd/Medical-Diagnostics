import streamlit as st
import json

def render_chat_component(api_key, base_url, model, system_prompt):
    """
    Renders a client-side chat component that communicates directly with the LLM API.
    This avoids Streamlit page refreshes during chat.
    
    Note: Using .replace() instead of .format() to avoid conflicts with CSS/JS braces.
    """
    
    # Escape the system prompt for JS string safely
    system_prompt_json = json.dumps(system_prompt)
    
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat Component</title>
        <style>
            body {
                font-family: "Source Sans Pro", sans-serif;
                margin: 0;
                padding: 0;
                background-color: #ffffff;
                display: flex;
                flex-direction: column;
                height: 100vh; /* Fill the iframe */
                overflow: hidden;
            }
            
            #chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 10px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                scrollbar-width: thin;
                background-color: #ffffff;
            }
            
            .message {
                max-width: 85%;
                padding: 10px 14px;
                border-radius: 12px;
                line-height: 1.5;
                font-size: 14px;
                position: relative;
                word-wrap: break-word;
            }
            
            .user-message {
                align-self: flex-end;
                background-color: #3b82f6; /* Medical Blue */
                color: white;
                border-bottom-right-radius: 2px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            
            .assistant-message {
                align-self: flex-start;
                background-color: #f1f5f9; /* Slate-100 */
                color: #1e293b; /* Slate-800 */
                border-bottom-left-radius: 2px;
                border: 1px solid #e2e8f0;
            }
            
            .input-area {
                padding: 10px;
                background-color: white;
                border-top: 1px solid #e2e8f0;
                display: flex;
                gap: 8px;
                align-items: center;
            }
            
            #user-input {
                flex: 1;
                padding: 10px 14px;
                border: 1px solid #cbd5e1;
                border-radius: 20px;
                outline: none;
                font-family: inherit;
                font-size: 14px;
                transition: all 0.2s;
                background-color: #f8fafc;
            }
            
            #user-input:focus {
                border-color: #3b82f6;
                background-color: white;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
            }
            
            #send-btn {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 50%;
                width: 38px;
                height: 38px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                flex-shrink: 0;
            }
            
            #send-btn:hover {
                background-color: #2563eb;
                transform: translateY(-1px);
            }
            
            #send-btn:disabled {
                background-color: #cbd5e1;
                cursor: not-allowed;
                transform: none;
            }
            
            #clear-btn {
                background-color: transparent;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
                border-radius: 50%;
                width: 32px;
                height: 32px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                flex-shrink: 0;
            }
            
            #clear-btn:hover {
                background-color: #fee2e2;
                border-color: #ef4444;
                color: #ef4444;
            }

            /* Markdown styles */
            .assistant-message p { margin: 0 0 8px 0; }
            .assistant-message p:last-child { margin: 0; }
            .assistant-message code { background-color: #e2e8f0; padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }
            .assistant-message pre { background-color: #1e293b; color: #e2e8f0; padding: 10px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
            .assistant-message pre code { background-color: transparent; color: inherit; padding: 0; }
            .assistant-message ul, .assistant-message ol { margin: 0 0 8px 0; padding-left: 20px; }
            
            /* Loading indicator */
            .typing-indicator { display: flex; gap: 4px; padding: 4px 8px; align-items: center; }
            .dot { width: 6px; height: 6px; background-color: #94a3b8; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
            .dot:nth-child(1) { animation-delay: -0.32s; }
            .dot:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

            .modal-overlay {
                position: fixed;
                inset: 0;
                display: none;
                align-items: center;
                justify-content: center;
                background: rgba(15, 23, 42, 0.45);
                z-index: 2147483647;
            }

            .modal-overlay.open {
                display: flex;
            }

            .modal-dialog {
                width: 360px;
                max-width: calc(100vw - 32px);
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.18), 0 10px 10px -5px rgba(0, 0, 0, 0.08);
                padding: 14px 14px 12px 14px;
            }

            .modal-title {
                font-size: 14px;
                font-weight: 600;
                color: #0f172a;
                margin: 0 0 6px 0;
            }

            .modal-text {
                font-size: 13px;
                color: #334155;
                margin: 0 0 12px 0;
                line-height: 1.5;
            }

            .modal-actions {
                display: flex;
                justify-content: flex-end;
                gap: 8px;
            }

            .modal-btn {
                border: 1px solid #e2e8f0;
                background: #ffffff;
                color: #0f172a;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                cursor: pointer;
                transition: all 0.2s;
            }

            .modal-btn:hover {
                background: #f8fafc;
            }

            .modal-btn-primary {
                background: #3b82f6;
                border-color: #3b82f6;
                color: #ffffff;
            }

            .modal-btn-primary:hover {
                background: #2563eb;
                border-color: #2563eb;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div class="message assistant-message">
                你好！我是您的医疗 AI 助手。您可以针对诊断结果向我提问。
            </div>
        </div>
        <div class="input-area">
            <button id="clear-btn" title="清空聊天记录">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
            <input type="text" id="user-input" placeholder="请输入您的问题..." autocomplete="off">
            <button id="send-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
        </div>

        <div id="clear-confirm-overlay" class="modal-overlay" role="dialog" aria-modal="true" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-title">提示</div>
                <div class="modal-text">确定要清空聊天记录吗？</div>
                <div class="modal-actions">
                    <button id="clear-confirm-cancel" class="modal-btn" type="button">取消</button>
                    <button id="clear-confirm-ok" class="modal-btn modal-btn-primary" type="button">确定</button>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            // Constants replaced by Python
            const API_KEY = "__API_KEY__";
            const BASE_URL = "__BASE_URL__";
            const MODEL = "__MODEL__";
            const SYSTEM_PROMPT = __SYSTEM_PROMPT_JSON__;
            const STORAGE_KEY = "medical_chat_history";
            
            const chatContainer = document.getElementById('chat-container');
            const userInput = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const clearBtn = document.getElementById('clear-btn');
            const clearConfirmOverlay = document.getElementById('clear-confirm-overlay');
            const clearConfirmOk = document.getElementById('clear-confirm-ok');
            const clearConfirmCancel = document.getElementById('clear-confirm-cancel');
            
            let messages = [];
            let clearConfirmResolver = null;
            let clearConfirmPreviousFocus = null;

            function openClearConfirm() {
                if (clearConfirmResolver) return Promise.resolve(false);
                clearConfirmPreviousFocus = document.activeElement;
                clearConfirmOverlay.classList.add('open');
                clearConfirmOverlay.setAttribute('aria-hidden', 'false');
                setTimeout(() => clearConfirmOk.focus(), 0);
                return new Promise((resolve) => {
                    clearConfirmResolver = resolve;
                });
            }

            function closeClearConfirm(result) {
                if (!clearConfirmResolver) return;
                const resolve = clearConfirmResolver;
                clearConfirmResolver = null;
                clearConfirmOverlay.classList.remove('open');
                clearConfirmOverlay.setAttribute('aria-hidden', 'true');
                if (clearConfirmPreviousFocus && typeof clearConfirmPreviousFocus.focus === 'function') {
                    clearConfirmPreviousFocus.focus();
                }
                clearConfirmPreviousFocus = null;
                resolve(result);
            }

            // Initialize
            function init() {
                loadChatHistory();
                restoreChatUI();
                
                // Bind events
                sendBtn.addEventListener('click', sendMessage);
                userInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendMessage();
                });
                
                clearConfirmOk.addEventListener('click', () => closeClearConfirm(true));
                clearConfirmCancel.addEventListener('click', () => closeClearConfirm(false));
                clearConfirmOverlay.addEventListener('click', (e) => {
                    if (e.target === clearConfirmOverlay) closeClearConfirm(false);
                });
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape') closeClearConfirm(false);
                });

                clearBtn.addEventListener('click', async () => {
                    const ok = await openClearConfirm();
                    if (!ok) return;
                    messages = [{ role: "system", content: SYSTEM_PROMPT }];
                    saveChatHistory();
                    restoreChatUI();
                });
            }

            function loadChatHistory() {
                try {
                    const saved = localStorage.getItem(STORAGE_KEY);
                    if (saved) {
                        messages = JSON.parse(saved);
                        // Ensure system prompt is up to date
                        if (messages.length > 0 && messages[0].role === "system") {
                            messages[0].content = SYSTEM_PROMPT;
                        }
                        return;
                    }
                } catch (e) {
                    console.error("Failed to load history", e);
                }
                messages = [{ role: "system", content: SYSTEM_PROMPT }];
            }
            
            function saveChatHistory() {
                try {
                    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
                } catch (e) {
                    console.error("Failed to save history", e);
                }
            }

            function appendMessage(role, content) {
                const div = document.createElement('div');
                div.className = `message ${role}-message`;
                if (role === 'assistant') {
                    div.innerHTML = marked.parse(content);
                } else {
                    div.textContent = content;
                }
                chatContainer.appendChild(div);
                scrollToBottom();
                return div;
            }
            
            function scrollToBottom() {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function restoreChatUI() {
                chatContainer.innerHTML = '';
                // Welcome message (always show if empty or just system prompt)
                if (messages.length <= 1) {
                appendMessage('assistant', '你好！我是您的医疗 AI 助手。您可以针对诊断结果向我提问。');
                } else {
                for (const msg of messages) {
                    if (msg.role !== 'system') {
                        appendMessage(msg.role, msg.content);
                    }
                }
                }
                scrollToBottom();
            }

            async function sendMessage() {
                const text = userInput.value.trim();
                if (!text) return;
                
                // Check API Key
                if (!API_KEY) {
                    alert("API Key is missing. Please configure it in settings.");
                    return;
                }
                
                // Add user message
                appendMessage('user', text);
                messages.push({ role: "user", content: text });
                saveChatHistory();
                
                userInput.value = '';
                userInput.disabled = true;
                sendBtn.disabled = true;
                
                // Create placeholder for assistant
                const assistantDiv = document.createElement('div');
                assistantDiv.className = 'message assistant-message';
                assistantDiv.innerHTML = '<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
                chatContainer.appendChild(assistantDiv);
                scrollToBottom();
                
                let fullResponse = "";
                
                try {
                    console.log("Sending request to:", BASE_URL);
                    const response = await fetch(BASE_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${API_KEY}`
                        },
                        body: JSON.stringify({
                            model: MODEL,
                            messages: messages,
                            stream: true
                        })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error("API Error:", response.status, errorText);
                        throw new Error(`API Error: ${response.statusText} (${response.status})`);
                    }
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");
                    
                    // Clear loading indicator
                    assistantDiv.innerHTML = "";
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const jsonStr = line.slice(6);
                                if (jsonStr === '[DONE]') continue;
                                
                                try {
                                    const json = JSON.parse(jsonStr);
                                    const content = json.choices[0].delta.content || "";
                                    fullResponse += content;
                                    assistantDiv.innerHTML = marked.parse(fullResponse);
                                    scrollToBottom();
                                } catch (e) {
                                    // Ignore parse errors for partial chunks
                                }
                            }
                        }
                    }
                    
                    messages.push({ role: "assistant", content: fullResponse });
                    saveChatHistory();
                    
                } catch (error) {
                    console.error("Send message error:", error);
                    assistantDiv.textContent = "Error: " + error.message;
                    assistantDiv.style.color = "#ef4444";
                    // If simple fetch error, maybe network issue
                    if (error.message.includes("Failed to fetch")) {
                         assistantDiv.textContent += " (Network Error or CORS issue)";
                    }
                } finally {
                    userInput.disabled = false;
                    sendBtn.disabled = false;
                    userInput.focus();
                }
            }

            // Start
            init();
        </script>
    </body>
    </html>
    """
    
    # Perform safe replacements
    html_code = html_code.replace("__API_KEY__", api_key)
    html_code = html_code.replace("__BASE_URL__", base_url)
    html_code = html_code.replace("__MODEL__", model)
    html_code = html_code.replace("__SYSTEM_PROMPT_JSON__", system_prompt_json)
    
    # Render with Streamlit
    # height=500 is usually enough for the internal content, but we forced stPopoverBody height in CSS
    # so we should make this fill the container.
    st.components.v1.html(html_code, height=500)
