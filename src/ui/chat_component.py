import streamlit as st
import json

def render_chat_component(api_key, base_url, model, system_prompt):
    """
    Renders a client-side chat component that communicates directly with the LLM API.
    This avoids Streamlit page refreshes during chat.
    """
    
    # Escape the system prompt for JS string
    system_prompt_json = json.dumps(system_prompt)
    
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat Component</title>
        <style>
            body {{
                font-family: "Source Sans Pro", sans-serif;
                margin: 0;
                padding: 0;
                background-color: transparent;
                display: flex;
                flex-direction: column;
                height: 100vh;
                overflow: hidden;
            }}
            
            #chat-container {{
                flex: 1;
                overflow-y: auto;
                padding: 10px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                scrollbar-width: thin;
            }}
            
            .message {{
                max-width: 80%;
                padding: 10px 15px;
                border-radius: 15px;
                line-height: 1.5;
                font-size: 14px;
                position: relative;
                word-wrap: break-word;
            }}
            
            .user-message {{
                align-self: flex-end;
                background-color: #ff9f43; /* Orange accent */
                color: white;
                border-bottom-right-radius: 2px;
            }}
            
            .assistant-message {{
                align-self: flex-start;
                background-color: #f0f2f6;
                color: #31333F;
                border-bottom-left-radius: 2px;
                border: 1px solid #e0e0e0;
            }}
            
            .input-area {{
                padding: 10px;
                background-color: white;
                border-top: 1px solid #ddd;
                display: flex;
                gap: 10px;
            }}
            
            #user-input {{
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 20px;
                outline: none;
                font-family: inherit;
            }}
            
            #user-input:focus {{
                border-color: #ff9f43;
            }}
            
            #send-btn {{
                background-color: #ff9f43;
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background-color 0.2s;
            }}
            
            #send-btn:hover {{
                background-color: #f39c12;
            }}
            
            #send-btn:disabled {{
                background-color: #ccc;
                cursor: not-allowed;
            }}

            /* Markdown styles (simplified) */
            .assistant-message p {{ margin: 0 0 10px 0; }}
            .assistant-message p:last-child {{ margin: 0; }}
            .assistant-message code {{ background-color: #e0e0e0; padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
            .assistant-message pre {{ background-color: #262730; color: #fff; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .assistant-message pre code {{ background-color: transparent; color: inherit; }}
            
            /* Loading indicator */
            .typing-indicator {{
                display: flex;
                gap: 5px;
                padding: 5px 10px;
            }}
            .dot {{
                width: 6px;
                height: 6px;
                background-color: #888;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out both;
            }}
            .dot:nth-child(1) {{ animation-delay: -0.32s; }}
            .dot:nth-child(2) {{ animation-delay: -0.16s; }}
            
            @keyframes bounce {{
                0%, 80%, 100% {{ transform: scale(0); }}
                40% {{ transform: scale(1); }}
            }}
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div class="message assistant-message">
                你好！我是您的医疗 AI 助手。您可以针对诊断结果向我提问。
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="请输入您的问题..." autocomplete="off">
            <button id="send-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            const apiKey = "{api_key}";
            const baseUrl = "{base_url}";
            const model = "{model}";
            const systemPrompt = {system_prompt_json};
            
            const chatContainer = document.getElementById('chat-container');
            const userInput = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            
            let messages = [
                {{ role: "system", content: systemPrompt }}
            ];

            function appendMessage(role, content) {{
                const div = document.createElement('div');
                div.className = `message ${{role}}-message`;
                if (role === 'assistant') {{
                    div.innerHTML = marked.parse(content);
                }} else {{
                    div.textContent = content;
                }}
                chatContainer.appendChild(div);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return div;
            }}

            async function sendMessage() {{
                const text = userInput.value.trim();
                if (!text) return;
                
                // Add user message
                appendMessage('user', text);
                messages.push({{ role: "user", content: text }});
                userInput.value = '';
                userInput.disabled = true;
                sendBtn.disabled = true;
                
                // Create placeholder for assistant message
                const assistantDiv = document.createElement('div');
                assistantDiv.className = 'message assistant-message';
                assistantDiv.innerHTML = '<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
                chatContainer.appendChild(assistantDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                let fullResponse = "";
                
                try {{
                    const response = await fetch(baseUrl, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${{apiKey}}`
                        }},
                        body: JSON.stringify({{
                            model: model,
                            messages: messages,
                            stream: true
                        }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`API Error: ${{response.statusText}}`);
                    }}
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");
                    
                    // Clear loading indicator
                    assistantDiv.innerHTML = "";
                    
                    while (true) {{
                        const {{ done, value }} = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value, {{ stream: true }});
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {{
                            if (line.startsWith('data: ')) {{
                                const jsonStr = line.slice(6);
                                if (jsonStr === '[DONE]') continue;
                                
                                try {{
                                    const json = JSON.parse(jsonStr);
                                    const content = json.choices[0].delta.content || "";
                                    fullResponse += content;
                                    
                                    // Simple markdown rendering for streaming (optimization: render periodically or full at end)
                                    // For smoother streaming, we just append text first, then render markdown at the end?
                                    // Or render markdown on every chunk (expensive but correct)?
                                    // Let's try rendering markdown on every chunk for now.
                                    assistantDiv.innerHTML = marked.parse(fullResponse);
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                    
                                }} catch (e) {{
                                    console.error("Error parsing JSON", e);
                                }}
                            }}
                        }}
                    }}
                    
                    messages.push({{ role: "assistant", content: fullResponse }});
                    
                }} catch (error) {{
                    assistantDiv.textContent = "Error: " + error.message;
                    assistantDiv.style.color = "red";
                }} finally {{
                    userInput.disabled = false;
                    sendBtn.disabled = false;
                    userInput.focus();
                }}
            }}

            sendBtn.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', (e) => {{
                if (e.key === 'Enter') sendMessage();
            }});
        </script>
    </body>
    </html>
    """
    
    st.components.v1.html(html_code, height=500)
