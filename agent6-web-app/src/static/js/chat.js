// Chat Interface JavaScript - Exactly replicates run_conversation CLI flow

console.log('CHAT.JS VERSION 20251105_v10');

class ChatInterface {
    constructor() {
// Expose helper on window for resize handlers inside the ChatInterface
if (typeof window !== 'undefined') window.updateArtifactsPanelLayout = updateArtifactsPanelLayout;
        console.log('ChatInterface constructor called');
        this.initializeElements();
        this.setupEventListeners();
        
        
        // Task polling management for real-time streaming
        this.activePoll = null;
        this.displayedMessageIds = new Set();  // Track which messages we've already displayed
        
        // Pick up any pending state set by other scripts (e.g., main.js)
        if (window.pendingConversationState) {
            try {
                this.conversationState = window.pendingConversationState;
                console.log('Applied pending conversation state:', this.conversationState);
                // Clear the pending flag
                delete window.pendingConversationState;
            } catch (e) {
                console.warn('Failed to apply pendingConversationState', e);
            }
        }
        if (!this.conversationState) this.conversationState = 'start';

        console.log('ChatInterface initialized successfully');
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        
        console.log('Elements initialized:');
        
       
    }

    setupEventListeners() {
        // Setup send button for main chat input
        const sendButton = document.querySelector('#chatInputContainer button');
        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendMessage());
        }

        // Setup send button for custom description
        const customSendButton = document.querySelector('#customInput button');
        if (customSendButton) {
            customSendButton.addEventListener('click', () => {
                submitCustomDescription();
            });
        }

        if (this.chatInput) {
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // Setup custom description textarea
        const customDescription = document.getElementById('customDescription');
        if (customDescription) {
            customDescription.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    e.preventDefault();
                    submitCustomDescription();
                }
            });
        }
        
        // Setup resize handle for panel resizing
        this.setupPanelResize();
        
        // Setup resize handle for system logs
        this.setupLogsResize();
        
        // Initialize grid padding based on initial system logs height
        this.updateGridPadding();
    }
    
    updateGridPadding() {
        const systemLogs = document.getElementById('systemLogs');
        const animationGrid = document.getElementById('animationGrid');

        if (systemLogs && animationGrid) {
            const logsHeight = systemLogs.offsetHeight || 250; // Default to 250px if not set
            animationGrid.style.paddingBottom = `${logsHeight + 20}px`; // +20px buffer
        }
    }
    
    setupLogsResize() {
        const logsResizeHandle = document.getElementById('logsResizeHandle');
        const systemLogs = document.getElementById('systemLogs');
        
        if (!logsResizeHandle || !systemLogs) {
            console.warn('System logs resize elements not found');
            return;
        }
        
        let isResizing = false;
        let startY = 0;
        let startHeight = 0;
        
        logsResizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = systemLogs.offsetHeight;
            logsResizeHandle.classList.add('dragging');
            document.body.style.cursor = 'row-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
            e.stopPropagation();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            // Note: Moving mouse UP increases height (negative deltaY)
            const deltaY = startY - e.clientY;
            const newHeight = startHeight + deltaY;
            
            // Enforce min/max constraints
            const minHeight = 50;
            const maxHeight = window.innerHeight * 0.6; // Max 60% of viewport
            
            if (newHeight >= minHeight && newHeight <= maxHeight) {
                systemLogs.style.height = `${newHeight}px`;
                // Remove transition during drag for smooth resizing
                systemLogs.style.transition = 'none';
                
                // IMPORTANT: Update  grid bottom padding to match logs height
                const animationGrid = document.getElementById('animationGrid');
                if (animationGrid) {
                    // Add 20px buffer for visual spacing
                    animationGrid.style.paddingBottom = `${newHeight + 20}px`;
                }
                // Continuously update artifacts panel layout while dragging
                if (window.updateArtifactsPanelLayout) window.updateArtifactsPanelLayout();
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                logsResizeHandle.classList.remove('dragging');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                // Restore transition
                systemLogs.style.transition = 'height 0.3s ease';
                // Final layout update after resize completes
                if (window.updateArtifactsPanelLayout) window.updateArtifactsPanelLayout();
            }
        });
    }
    
    setupPanelResize() {
        const resizeHandle = document.getElementById('resizeHandle');
        const chatPanel = document.querySelector('.chat-panel');
        const container = document.querySelector('.container');
        const systemLogs = document.getElementById('systemLogs');
        
        if (!resizeHandle || !chatPanel || !container) {
            console.warn('Resize elements not found');
            return;
        }
        
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;
        
        resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startWidth = chatPanel.offsetWidth;
            resizeHandle.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const deltaX = e.clientX - startX;
            const newWidth = startWidth + deltaX;
            const containerWidth = container.offsetWidth;
            
            // Enforce min/max constraints
            const minWidth = 300;
            const maxWidth = containerWidth * 0.7; // 70% max
            
            if (newWidth >= minWidth && newWidth <= maxWidth) {
                const widthPercent = (newWidth / containerWidth) * 100;
                chatPanel.style.width = `${widthPercent}%`;
                
                // Sync system logs width with animation panel (remaining space)
                if (systemLogs) {
                    const animationPanelPercent = 100 - widthPercent;
                    systemLogs.style.width = `${animationPanelPercent}%`;
                }
                // Continuously update artifacts panel layout while dragging
                if (window.updateArtifactsPanelLayout) window.updateArtifactsPanelLayout();
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                resizeHandle.classList.remove('dragging');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                // After resizing completes, update artifacts layout
                if (window.updateArtifactsPanelLayout) window.updateArtifactsPanelLayout();
            }
        });
    }

    startConversation() {
        // Hide static UI elements and start dynamic conversation
        const datasetSelection = document.getElementById('datasetSelection');
        const phenomenonSelection = document.getElementById('phenomenonSelection');
        const customInput = document.getElementById('customInput');
        const chatInputContainer = document.getElementById('chatInputContainer');

        if (datasetSelection) datasetSelection.style.display = 'none';
        if (phenomenonSelection) phenomenonSelection.style.display = 'none';
        if (customInput) customInput.style.display = 'none';
        if (chatInputContainer) chatInputContainer.style.display = 'block';

        // Exactly like run_conversation start
        this.sendApiMessage('', 'start');
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addMessage(message, 'user');
        this.chatInput.value = '';

        // Send based on current conversation state
        // Treat the main chat input as the custom-description input when the UI
        // is in phenomenon selection or awaiting a custom description. This
        // makes typing into the main input behave like choosing 'Custom Description'
        // (choice '0') and matches the curl/API behavior.
        if (this.conversationState === 'awaiting_guidance') {
            await this.sendApiMessage(message, 'provide_guidance');
        } else if (this.conversationState === 'conversation_loop') {
            // If we're in the conversation loop, users are usually expected to
            // reply with a short token: 'y' (yes), 'n' (no), 'g' (guide), or
            // 'quit'. However users sometimes type a full custom description
            // here (e.g. a new phenomenon). Treat long/free-text input as a
            // custom description (choice '0') unless the message is one of the
            // short conversation-control tokens.
            const short = message.trim().toLowerCase();
                if (['y', 'n', 'g', 'quit'].includes(short)) {
                await this.sendApiMessage(message, 'continue_conversation');
            } else {
                // Treat as a custom phenomenon description
                // send the phenomenon as the message; no 'choice' field required
                // Use continue_conversation to keep the protocol simple and
                // avoid the deprecated legacy action.
                await this.sendApiMessage(message, 'continue_conversation');
            }
        } else if (this.conversationState === 'custom_description' || this.conversationState === 'phenomenon_selection') {
            // Directly post the custom phenomenon as the message
            // Use continue_conversation instead of the deprecated legacy action
            await this.sendApiMessage(message, 'continue_conversation');
        } else {
            // Fallback: if conversation state is unexpected, still attempt to send
            // as a custom description so UI input doesn't get ignored.
            await this.sendApiMessage(message, 'continue_conversation');
        }
    }

    async sendApiMessage(message, action, extraData = {}) {
        let thinkingId = null;
        try {
            this.updateStatus('Processing...', 'processing');
            
            // Add thinking indicator for queries that will take time
            if (action === 'continue_conversation') {
                thinkingId = 'thinking-' + Date.now();
                this.addThinkingIndicator(thinkingId);
            }

            const requestData = {
                message: message,
                action: action,
                ...extraData
            };
            

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            console.log('Received response data:', data);
            
            // Check if this is a task-based response (for continue_conversation)
            if (data.type === 'task_started' && data.task_id) {
                console.log('🚀 Task started:', data.task_id);
                // Remove initial thinking indicator, we'll show real-time updates now
                if (thinkingId) {
                    this.removeThinkingIndicator(thinkingId);
                }
                // Start polling for task updates
                await this.pollTaskStatus(data.task_id);
                return;
            }
            
            // Remove thinking indicator for non-task responses
            if (thinkingId) {
                this.removeThinkingIndicator(thinkingId);
            }

            if (data.status === 'error') {
                this.addMessage(`Error: ${data.message}`, 'bot');
                this.updateStatus('Error', 'error');
                return;
            }

            // Handle different response types exactly like run_conversation
            this.handleApiResponse(data);

        } catch (error) {
            console.error('Error:', error);
            // Remove thinking indicator on error
            if (thinkingId) {
                this.removeThinkingIndicator(thinkingId);
            }
            this.addMessage('Sorry, there was an error processing your request.', 'bot');
            this.updateStatus('Error', 'error');
        }
    }
    
    async pollTaskStatus(taskId) {
        console.log('📊 Starting to poll task:', taskId);
        let pollCount = 0;
        const maxPolls = 1200;  // 10 minutes max (1200 * 500ms) - increased for long-running insight generation
        
        this.activePoll = setInterval(async () => {
            pollCount++;
            try {
                const response = await fetch(`/api/chat/status/${taskId}`);
                if (!response.ok) {
                    console.error('Poll failed with status:', response.status);
                    clearInterval(this.activePoll);
                    this.addMessage('Error: Failed to get task status', 'bot');
                    return;
                }
                
                const status = await response.json();
                console.log(`📡 Poll #${pollCount}:`, status.status, `- ${status.messages.length} messages`);
                
                // Process new messages
                if (status.messages && status.messages.length > 0) {
                    status.messages.forEach(msg => {
                        const msgId = `${taskId}-${msg.timestamp}`;
                        if (!this.displayedMessageIds.has(msgId)) {
                            this.handleProgressMessage(msg);
                            this.displayedMessageIds.add(msgId);
                        }
                    });
                }
                
                // Check if task is completed
                if (status.status === 'completed') {
                    console.log('✅ Task completed!', status.result);
                    clearInterval(this.activePoll);
                    this.activePoll = null;
                    
                    // Handle final result
                    if (status.result) {
                        this.handleApiResponse(status.result);
                    }
                    this.updateStatus('Ready', 'success');
                    
                } else if (status.status === 'error') {
                    console.error('❌ Task failed:', status.error);
                    clearInterval(this.activePoll);
                    this.activePoll = null;
                    this.addMessage(`Error: ${status.error}`, 'bot');
                    this.updateStatus('Error', 'error');
                }
                
                // NO TIMEOUT - removed to allow unlimited processing time for complex queries
                
            } catch (error) {
                console.error('Poll error:', error);
                clearInterval(this.activePoll);
                this.activePoll = null;
                this.addMessage('Error polling for updates', 'bot');
            }
        }, 500);  // Poll every 500ms
    }
    
    handleProgressMessage(msg) {
        console.log('📬 Processing progress message:', msg.type);
        
        switch (msg.type) {
            case 'intent_parsed':
                // Intent goes to chat for user visibility
                this.addMessage('🔍 Intent Analysis:', 'bot');
                const intentStr = JSON.stringify(msg.data, null, 2);
                this.addMessage(intentStr, 'bot');
                break;
                
            case 'insight_extraction_started':
                // Show in chat that insight generation started
                this.addMessage('💡 Generating Insight...', 'bot');
                break;
                
            case 'iteration_update':
            case 'llm_response':
            case 'query_execution':
            case 'query_failed':
                // These are system-level logs, already in system logs panel
                // No need to show in chat
                break;
                
            case 'insight_generated':
                // Only show actual insight in chat
                if (msg.data && msg.data.insight) {
                    this.addMessage(msg.data.insight, 'bot');
                }
                if (msg.data && msg.data.data_summary) {
                    this.addMessage('📊 Data Summary:', 'bot');
                    const summaryStr = typeof msg.data.data_summary === 'string' 
                        ? msg.data.data_summary 
                        : JSON.stringify(msg.data.data_summary, null, 2);
                    this.addMessage(summaryStr, 'bot');
                }
                // If progress message includes artifact references, render them
                try {
                    if (msg.data && (msg.data.plot_files || msg.data.query_code_file || msg.data.plot_code_file)) {
                        console.log('Progress message contains artifacts; rendering artifacts panel');
                        this.renderArtifacts(msg.data);
                    }
                } catch (e) {
                    console.warn('Failed to render artifacts from progress message', e);
                }
                break;
                
            default:
                console.log('Unknown progress message type:', msg.type);
        }
    }

    handleApiResponse(data) {
        console.log('====================================');
        console.log('HANDLING API RESPONSE');
        console.log('Response type:', data.type);
        console.log('Response keys:', Object.keys(data));
        console.log('Has assistant_messages:', 'assistant_messages' in data);
        if (data.assistant_messages) {
            console.log('assistant_messages length:', data.assistant_messages.length);
            console.log('assistant_messages:', JSON.stringify(data.assistant_messages, null, 2));
        }
        console.log('====================================');

        try {
            // Display LLM agent outputs (intent parsing, insight generation) in chat
            if (data.assistant_messages && Array.isArray(data.assistant_messages)) {
                console.log('🔥 Processing assistant_messages:', data.assistant_messages.length);
                data.assistant_messages.forEach((msg, index) => {
                    console.log(`🔥 Processing message ${index + 1}:`, msg.type);
                    if (msg.type === 'intent_parsing' && msg.data) {
                        // Show intent parsing output as formatted JSON
                        console.log('🔍 Adding intent parsing message');
                        this.addMessage('🔍 Intent Analysis:', 'bot');
                        const intentStr = JSON.stringify(msg.data, null, 2);
                        this.addMessage(intentStr, 'bot');
                        console.log('✅ Intent message added');
                    } else if (msg.type === 'insight_generation' && msg.data) {
                        // Show insight generation output
                        console.log('💡 Adding insight generation message');
                        this.addMessage('💡 Generating Insight...', 'bot');
                        if (msg.data.insight) {
                            console.log('Adding insight text:', msg.data.insight.substring(0, 50));
                            this.addMessage(msg.data.insight, 'bot');
                        }
                        if (msg.data.data_summary) {
                            console.log('Adding data summary');
                            this.addMessage('📊 Data Summary:', 'bot');
                            const summaryStr = typeof msg.data.data_summary === 'string' 
                                ? msg.data.data_summary 
                                : JSON.stringify(msg.data.data_summary, null, 2);
                            this.addMessage(summaryStr, 'bot');
                        }
                        console.log('✅ Insight messages added');
                    } else {
                        console.warn('⚠️ Unknown assistant message type:', msg.type);
                    }
                });
                console.log('✅ All assistant_messages processed');
            } else {
                console.log('⚠️ No assistant_messages in response or not an array');
            }
            
            switch (data.type) {
                case 'dataset_selection':
                    // Show the exact same options as run_conversation
                    this.addTypingMessage(data.message, 'bot');
                    this.conversationState = 'phenomenon_selection';
                    break;

                case 'dataset_summary':
                    // Server returned a dataset summary (e.g., after select_dataset)
                    console.log('Processing dataset_summary response');
                    try {
                        // Short headline or status message
                        if (data.message) this.addMessage(data.message, 'bot');

                        // The summary may be a string or a structured object
                        const summary = data.summary || data.result || data.summary_text;
                        if (summary) {
                            if (typeof summary === 'string') {
                                this.addMessage(summary, 'bot');
                            } else if (typeof summary === 'object') {
                                // Try common fields first
                                if (summary.text) {
                                    this.addMessage(summary.text, 'bot');
                                } else if (summary.summary) {
                                    this.addMessage(summary.summary, 'bot');
                                } else {
                                    // Fallback to pretty JSON
                                    this.addMessage(JSON.stringify(summary, null, 2), 'bot');
                                }
                            } else {
                                this.addMessage(String(summary), 'bot');
                            }
                        }

                        // Advance conversation state so follow-ups work normally
                        this.conversationState = 'continue_conversation';
                        this.updateStatus('Summary Ready', 'success');
                    } catch (err) {
                        console.error('Error rendering dataset_summary:', err);
                        this.addMessage(data.message || 'Dataset summary received', 'bot');
                    }
                    break;

                case 'particular_response':
                    // User asked a specific question - show the answer
                    console.log('Processing particular_response (specific question)');
                    if (data.message) this.addMessage(data.message, 'bot');
                    if (data.answer) this.addMessage(data.answer, 'bot');
                    if (data.insight) this.addMessage(data.insight, 'bot');
                    if (data.data_summary) {
                        const summaryStr = typeof data.data_summary === 'string' 
                            ? data.data_summary 
                            : JSON.stringify(data.data_summary, null, 2);
                        this.addMessage('📊 ' + summaryStr, 'bot');
                    }
                    if (data.plot_files && data.plot_files.length > 0) {
                        this.addMessage(`📈 Generated ${data.plot_files.length} plot(s)`, 'bot');
                    }
                    this.updateStatus('Answer Provided', 'success');
                    break;

                case 'exploration_response':
                    // User wants general exploration - show insights
                    console.log('Processing exploration_response (general exploration)');
                    if (data.message) this.addMessage(data.message, 'bot');
                    if (data.insights) {
                        if (typeof data.insights === 'string') {
                            this.addMessage(data.insights, 'bot');
                        } else if (Array.isArray(data.insights)) {
                            data.insights.forEach(insight => this.addMessage(`• ${insight}`, 'bot'));
                        } else {
                            this.addMessage(JSON.stringify(data.insights, null, 2), 'bot');
                        }
                    }
                    if (data.insight) this.addMessage(data.insight, 'bot');
                    if (data.data_summary) {
                        const summaryStr = typeof data.data_summary === 'string' 
                            ? data.data_summary 
                            : JSON.stringify(data.data_summary, null, 2);
                        this.addMessage('📊 ' + summaryStr, 'bot');
                    }
                    if (data.plot_files && data.plot_files.length > 0) {
                        this.addMessage(`📈 Generated ${data.plot_files.length} plot(s)`, 'bot');
                    }
                    this.updateStatus('Insights Ready', 'success');
                    break;

                case 'help_response':
                    // User requested help
                    console.log('Processing help_response');
                    if (data.message) this.addMessage(data.message, 'bot');
                    this.updateStatus('Help Provided', 'success');
                    break;

                case 'clarification':
                    // Query was unrelated or needs clarification
                    console.log('Processing clarification response');
                    this.addMessage(data.message || 'Could you rephrase that in terms of data exploration?', 'bot');
                    this.updateStatus('Awaiting Input', 'info');
                    break;

                case 'agent_response':
                    // Generic agent response (fallback)
                    console.log('Processing generic agent_response');
                    const result = data.result || {};
                    this.addMessage(result.message || data.message || 'Processing complete', 'bot');
                    this.updateStatus('Ready', 'success');
                    break;

                case 'conversation_end':
                    // User said 'n' or 'quit' (exactly like run_conversation)
                    this.addMessage(data.message, 'bot');
                    this.conversationState = 'start';
                    this.updateStatus('Session Complete', 'success');
                    // Optionally restart after a delay
                    setTimeout(() => {
                        this.addMessage('Would you like to start a new animation? Click here to begin.', 'bot');
                        this.addCustomContent('<button class="option-btn" onclick="chatInterface.startConversation()">Start New Animation</button>');
                    }, 3000);
                    break;

                case 'guidance_request':
                    // User said 'g' - asking for guidance (exactly like run_conversation)
                    this.addMessage(data.message, 'bot');
                    this.conversationState = 'awaiting_guidance';
                    break;

                case 'clarification_needed':
                    // Invalid response (exactly like run_conversation error handling)
                    this.addMessage(data.message, 'bot');
                    break;

                case 'new_approach_suggested':
                    // LLM suggested new approach
                    this.addMessage(data.message, 'bot');
                    break;

                default:
                        this.addMessage(data.message || 'Unknown response type', 'bot');
                }

                // Render artifacts panel if response contains artifact references
                try {
                    if ((data.plot_files && data.plot_files.length > 0) || data.query_code_file || data.plot_code_file) {
                        console.log('Rendering artifacts for response');
                        this.renderArtifacts(data);
                    }
                } catch (e) {
                    console.warn('Failed to render artifacts:', e);
                }
        } catch (error) {
            console.error('Error in handleApiResponse:', error);
            console.error('Full error details:', error.message, error.stack);
            this.addMessage('Sorry, there was an error processing your request.', 'bot');
            this.updateStatus('Error', 'error');
        }
    }

    selectPhenomenon(id, name) {
        // Hide static phenomenon selection UI
        const phenomenonSelection = document.getElementById('phenomenonSelection');
        if (phenomenonSelection) phenomenonSelection.style.display = 'none';

        // Exactly like run_conversation choice handling
        if (id === '0') {
            this.addMessage(`Selected: ${name}`, 'user');
            this.addTypingMessage('Please describe the oceanographic phenomenon you want to visualize:', 'bot');
            this.conversationState = 'custom_description';

            // Show custom input UI
            const customInput = document.getElementById('customInput');
            if (customInput) customInput.style.display = 'block';
        } else {
            this.addMessage(`Selected: ${name}`, 'user');
            // Send the phenomenon id/name as the message so backend receives
            // a textual description and does not rely on numeric 'choice'.
            const desc = name || id;
            // Use continue_conversation to replace the deprecated action
            this.sendApiMessage(desc, 'continue_conversation');
        }
    }

    addMessage(content, type) {
        try {
            console.log(`Adding ${type} message with content length:`, content ? content.length : 0);

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;

            const avatar = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

            // Handle multi-line content with proper formatting and escape HTML
            const formattedContent = content
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;')
                .replace(/\n/g, '<br>');


            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <p>${formattedContent}</p>
                </div>
            `;

            this.chatMessages.appendChild(messageDiv);
            this.scrollToBottom();
            console.log(`Successfully added ${type} message`);
        } catch (error) {
            console.error('Error in addMessage:', error);
            console.error('Content that caused error:', content);

            // Fallback: try to add a simple error message
            try {
                const errorDiv = document.createElement('div');
                errorDiv.className = `message ${type}-message`;
                errorDiv.innerHTML = `
                    <div class="message-avatar"><i class="fas fa-robot"></i></div>
                    <div class="message-content">
                        <p>[Message content could not be displayed due to formatting issues]</p>
                    </div>
                `;
                this.chatMessages.appendChild(errorDiv);
                this.scrollToBottom();
            } catch (fallbackError) {
                console.error('Even fallback message failed:', fallbackError);
            }

            throw error; // Re-throw so the caller knows it failed
        }
    }

    addThinkingIndicator(id) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.id = id;
        
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <p><span class="thinking-dots">🤔 Analyzing your request<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></span></p>
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add CSS animation for dots
        if (!document.getElementById('thinking-animation-style')) {
            const style = document.createElement('style');
            style.id = 'thinking-animation-style';
            style.textContent = `
                .thinking-dots .dot {
                    animation: blink 1.4s infinite;
                }
                .thinking-dots .dot:nth-child(2) {
                    animation-delay: 0.2s;
                }
                .thinking-dots .dot:nth-child(3) {
                    animation-delay: 0.4s;
                }
                @keyframes blink {
                    0%, 20% { opacity: 0.2; }
                    50% { opacity: 1; }
                    100% { opacity: 0.2; }
                }
            `;
            document.head.appendChild(style);
        }
        
        return id;
    }

    removeThinkingIndicator(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    addTypingMessage(content, type = 'bot', typingSpeed = 30, onComplete = null) {
        try {
            console.log(`Adding ${type} typing message with content length:`, content ? content.length : 0);

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;

            const avatar = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

            // Create empty message container
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <p class="typing-content"></p>
                </div>
            `;

            this.chatMessages.appendChild(messageDiv);
            const typingElement = messageDiv.querySelector('.typing-content');

            // Add typing cursor
            const cursor = document.createElement('span');
            cursor.className = 'typing-cursor';
            cursor.textContent = '|';
            typingElement.appendChild(cursor);

            this.scrollToBottom();

            // Process content - no need to HTML escape since we're using text nodes
            const processedContent = content;

            // Split content into words for typing effect
            const lines = processedContent.split('\n');
            let currentLineIndex = 0;
            let currentCharIndex = 0;

            const typeText = () => {
                if (currentLineIndex < lines.length) {
                    const currentLine = lines[currentLineIndex];

                    if (currentCharIndex < currentLine.length) {
                        // Add next character
                        const char = currentLine[currentCharIndex];
                        const textNode = document.createTextNode(char);
                        typingElement.insertBefore(textNode, cursor);
                        currentCharIndex++;

                        // Auto-scroll as content appears
                        this.scrollToBottom();

                        // Continue typing
                        setTimeout(typeText, typingSpeed);
                    } else {
                        // End of line, add line break and move to next line
                        if (currentLineIndex < lines.length - 1) {
                            const brElement = document.createElement('br');
                            typingElement.insertBefore(brElement, cursor);
                        }
                        currentLineIndex++;
                        currentCharIndex = 0;

                        // Continue with next line after a brief pause
                        setTimeout(typeText, typingSpeed * 2);
                    }
                } else {
                    // Typing complete, remove cursor
                    cursor.remove();
                    console.log(`Successfully completed typing ${type} message`);

                    // Call completion callback if provided
                    if (onComplete && typeof onComplete === 'function') {
                        setTimeout(onComplete, 100); // Small delay for visual smoothness
                    }
                }
            };

            // Start typing after a brief delay
            setTimeout(typeText, 200);

        } catch (error) {
            console.error('Error in addTypingMessage:', error);
            // Fallback to regular message
            this.addMessage(content, type);
            // Still call completion callback if provided
            if (onComplete && typeof onComplete === 'function') {
                setTimeout(onComplete, 100);
            }
        }
    }

    addCustomContent(html) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';

        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">${html}</div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    updateStatus(message, type) {
        // Lightweight, resilient status updater used across the UI.
        try {
            console.log('Status update:', message, type);
            if (this.animationStatus) {
                // Prefer a child span for text, fall back to the element itself
                const statusElement = this.animationStatus.querySelector('span:last-child') || this.animationStatus;
                const indicator = this.animationStatus.querySelector('.status-indicator');

                if (statusElement) {
                    // If statusElement is the container, set its textContent appropriately
                    if (statusElement === this.animationStatus) {
                        this.animationStatus.textContent = message;
                    } else {
                        statusElement.textContent = message;
                    }
                }

                if (indicator) {
                    indicator.className = `status-indicator ${type || ''}`.trim();
                }
            }
        } catch (e) {
            console.warn('updateStatus failed:', e);
        }
    }
    
    // --- Artifacts rendering helpers ---
    renderArtifacts(data) {
        try {
            const panel = document.getElementById('artifactsPanel');
            if (!panel) return;

            // Plots
            const plotsContainer = document.getElementById('plotsContainer');
            if (!plotsContainer) return;
            plotsContainer.innerHTML = '';
            const plotFiles = Array.isArray(data.plot_files) ? data.plot_files : [];
            if (plotFiles.length === 0) {
                plotsContainer.innerHTML = '<em>No plots available.</em>';
            } else {
                plotFiles.forEach((p, idx) => {
                    try {
                        const url = this.pathToArtifactUrl(p);
                        const wrap = document.createElement('div');
                        wrap.className = 'artifact-plot';
                        const img = document.createElement('img');
                        img.src = url;
                        img.alt = `Plot ${idx + 1}`;
                        img.style.maxWidth = '100%';
                        img.style.height = 'auto';
                        img.style.display = 'block';
                        img.style.marginBottom = '6px';
                        // Debugging hooks: log load/error and visually mark failures
                        img.onload = () => {
                            console.log('Artifact image loaded:', url, 'size:', img.naturalWidth, img.naturalHeight);
                            // ensure visible
                            img.style.visibility = 'visible';
                        };
                        img.onerror = (ev) => {
                            console.error('Artifact image failed to load:', url, ev);
                            // show visible red border to help debugging
                            img.style.border = '4px solid rgba(255,0,0,0.9)';
                            img.style.visibility = 'visible';
                        };

                        const dl = document.createElement('a');
                        dl.href = url;
                        dl.className = 'artifact-download';
                        dl.textContent = 'Download';
                        dl.style.display = 'inline-block';
                        dl.style.marginBottom = '10px';

                        wrap.appendChild(img);
                        wrap.appendChild(dl);
                        plotsContainer.appendChild(wrap);
                    } catch (e) {
                        console.warn('Failed to render plot item', e, p);
                    }
                });
            }

            // Plot code
            const plotCodeContainer = document.getElementById('plotCodeContainer');
            if (plotCodeContainer) {
                if (data.plot_code_file) {
                    const url = this.pathToArtifactUrl(data.plot_code_file);
                    this.fetchAndShowCode(url, plotCodeContainer, 'plot_code.py');
                } else {
                    plotCodeContainer.innerHTML = '<em>No plot code available.</em>';
                }
            }

            // Query code
            const queryCodeContainer = document.getElementById('queryCodeContainer');
            if (queryCodeContainer) {
                if (data.query_code_file) {
                    const url = this.pathToArtifactUrl(data.query_code_file);
                    this.fetchAndShowCode(url, queryCodeContainer, 'query_code.py');
                } else {
                    queryCodeContainer.innerHTML = '<em>No query code available.</em>';
                }
            }

        } catch (e) {
            console.warn('renderArtifacts error', e);
        }
    }

    pathToArtifactUrl(p) {
        try {
            if (!p) return p;
            if (typeof p !== 'string') p = String(p);
            // If already an API path or URL, return as-is
            if (p.startsWith('/api/') || p.startsWith('/static') || p.startsWith('http')) return p;
            // Find 'ai_data' segment and build /api/artifacts/<relpath>
            const idx = p.indexOf('ai_data');
            if (idx !== -1) {
                const rel = p.substring(idx + 'ai_data'.length + 1).replace(/\\\\/g, '/').replace(/\\/g, '/');
                // Ensure we return a same-origin absolute URL so browser requests don't hit cross-host issues
                const origin = (window && window.location && window.location.origin) ? window.location.origin : '';
                return `${origin}/api/artifacts/${rel}`.replace(/%2F/g, '/');
            }
            // Fallback: return original path (may work if already relative)
            return p;
        } catch (e) {
            return p;
        }
    }

    async fetchAndShowCode(url, containerEl, fallbackName) {
        try {
            containerEl.innerHTML = 'Loading code...';
            const resp = await fetch(url);
            if (!resp.ok) {
                containerEl.innerHTML = `<em>Unable to fetch code (${resp.status})</em>`;
                return;
            }
            const text = await resp.text();
            const safe = text.replace(/&/g, '&amp;').replace(/</g, '&lt;');
            containerEl.innerHTML = `<pre class="artifact-code"><code>${safe}</code></pre><a href="${url}" download="${fallbackName}">Download</a>`;
        } catch (e) {
            console.warn('fetchAndShowCode error', e);
            containerEl.innerHTML = `<em>Error loading code</em>`;
        }
    }

    downloadAllPlots() {
        try {
            const plots = Array.from(document.querySelectorAll('#plotsContainer a.artifact-download'));
            plots.forEach(a => {
                const link = document.createElement('a');
                link.href = a.href;
                link.download = '';
                document.body.appendChild(link);
                link.click();
                link.remove();
            });
        } catch (e) {
            console.warn('downloadAllPlots failed', e);
        }
    }

}

// Global functions called from HTML
function selectDataset(type) {
    console.log('selectDataset called with type:', type);
    if (type === 'available') {
        // Start the conversation when user selects available dataset
        console.log('Starting conversation...');
        chatInterface.startConversation();
    } else if (type === 'upload') {
      console.log('Starting conversation...');
        chatInterface.startConversation();
    }
}


function submitCustomDescription() {
    const description = document.getElementById('customDescription').value.trim();
    if (description && chatInterface) {
    chatInterface.addMessage(description, 'user');
    // Send the custom description as the phenomenon message. Do not send
    // the legacy numeric 'choice' field — backend expects free-text.
    // Use continue_conversation instead of the deprecated legacy action
    chatInterface.sendApiMessage(description, 'continue_conversation');
        document.getElementById('customDescription').value = '';

        // Hide custom input UI
        const customInput = document.getElementById('customInput');
        if (customInput) customInput.style.display = 'none';
    }
}

function sendMessage() {
    if (chatInterface) {
        chatInterface.sendMessage();
    }
}

// Initialize the chat interface when the page loads
let chatInterface;
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing ChatInterface...');
    chatInterface = new ChatInterface();
    console.log('ChatInterface assigned to global variable:', chatInterface);

    // Initialize system logs
    initializeSystemLogs();
    // No delegation required: workflow uses explicit dataset selection -> summary -> typed conversation.
    // Give the layout helper a short moment to compute and then reveal the artifacts panel
    setTimeout(() => {
        try {
            if (window.updateArtifactsPanelLayout) window.updateArtifactsPanelLayout();
            const panel = document.getElementById('artifactsPanel');
            if (panel) panel.style.visibility = 'visible';
        } catch (e) {
            console.warn('Failed to initialize artifacts panel layout', e);
        }
    }, 30);
});

// System logs functions
function initializeSystemLogs() {
    // Clear backend logs on page load
    clearSystemLogs();

    // Show system logs immediately when page loads
    const systemLogs = document.getElementById('systemLogs');
    if (systemLogs) {
        systemLogs.style.display = 'block';
    }

    // Start polling for logs
    setInterval(pollSystemLogs, 2000);

    // Start system logs visibility watchdog
    setInterval(ensureSystemLogsVisible, 1000);

    // Add initial log
    setTimeout(() => {
        addSystemLog('System initialized and ready', 'info');
    }, 1000);
}

function ensureSystemLogsVisible() {
    const systemLogs = document.getElementById('systemLogs');
    if (systemLogs) {
        const currentDisplay = window.getComputedStyle(systemLogs).display;
        if (currentDisplay === 'none') {
            console.log('System logs were hidden, making them visible again');
            systemLogs.style.display = 'block';
        }
    }
}

function clearSystemLogs() {
    // Clear frontend display
    const logsContent = document.getElementById('logsContent');
    if (logsContent) {
        logsContent.innerHTML = `
            <div class="log-placeholder">
                <i class="fas fa-info-circle"></i>
                <p>System logs will appear here </p>
            </div>
        `;
    }

    // Clear backend logs
    fetch('/api/system_logs', {
        method: 'DELETE'
    }).catch(error => {
        console.error('Error clearing backend logs:', error);
    });
}

function toggleLogsSize() {
    const systemLogs = document.querySelector('.system-logs');
    const toggleBtn = document.querySelector('#toggleLogsBtn');

    if (systemLogs && toggleBtn) {
        if (systemLogs.classList.contains('minimized')) {
            // From minimized to normal (maximize)
            systemLogs.classList.remove('minimized');
            toggleBtn.innerHTML = '<i class="fas fa-minus"></i>';
            toggleBtn.title = 'Minimize logs';
        } else {
            // From normal to minimized
            systemLogs.classList.add('minimized');
            toggleBtn.innerHTML = '<i class="fas fa-plus"></i>';
            toggleBtn.title = 'Maximize logs';
        }
    }
}

function minimizeSystemLogs() {
    const systemLogs = document.querySelector('.system-logs');
    const toggleBtn = document.querySelector('#toggleLogsBtn');

    if (systemLogs && toggleBtn) {
        systemLogs.classList.remove('expanded');
        systemLogs.classList.add('minimized');
        toggleBtn.innerHTML = '<i class="fas fa-plus"></i>';
        toggleBtn.title = 'Maximize logs';
        console.log('📦 System logs minimized automatically');

        // Add activity indicator for 3 seconds
        systemLogs.classList.add('active');
        setTimeout(() => {
            systemLogs.classList.remove('active');
        }, 3000);
    }
}

// toggleLogsExpand removed — use toggleLogsSize() directly where needed

// Helper: remove common Markdown artifacts before showing LLM evaluation in chat

function updateArtifactsPanelLayout() {
        try {
            const panel = document.getElementById('artifactsPanel');
            const container = document.querySelector('.container');
            const chatPanel = document.querySelector('.chat-panel');
            const systemLogs = document.querySelector('.system-logs');
            if (!panel || !container || !chatPanel) return;

            // Compute available right-side area based on chatPanel width and container
            const containerRect = container.getBoundingClientRect();
            const chatRect = chatPanel.getBoundingClientRect();

            const rightMargin = 20; // match CSS right spacing
            const leftX = Math.max(chatRect.right, containerRect.left);
            const availableWidth = Math.max(200, containerRect.right - rightMargin - leftX - 10);

            panel.style.position = 'fixed';
            panel.style.left = `${leftX + 10}px`;
            panel.style.width = `${availableWidth}px`;

            // Keep top aligned below header (approx 70px) and bottom above system logs
            const topOffset = 70; // same as CSS default
            const logsHeight = systemLogs ? systemLogs.offsetHeight : 0;
            const bottomOffset = Math.max(20, logsHeight + 20);
            panel.style.top = `${topOffset}px`;
            panel.style.bottom = `${bottomOffset}px`;
            panel.style.height = 'auto'; // let top/bottom control height
        } catch (e) {
            console.warn('updateArtifactsPanelLayout error', e);
        }
    }
function cleanMarkdownForChat(text) {
    if (!text) return '';
    let s = String(text);

    // Remove fenced code blocks ```...``` including language marker
    s = s.replace(/```[\s\S]*?```/g, match => {
        // Remove the backticks but keep inner text
        return match.replace(/```/g, '');
    });

    // Remove heading markers (###, ##, #)
    s = s.replace(/^#{1,6}\s*/gm, '');

    // Remove inline code markers `code`
    s = s.replace(/`([^`]*)`/g, '$1');

    // Remove bold/italic markers **text**, *text*, __text__, _text_
    s = s.replace(/\*\*(.*?)\*\*/g, '$1');
    s = s.replace(/\*(.*?)\*/g, '$1');
    s = s.replace(/__(.*?)__/g, '$1');
    s = s.replace(/_(.*?)_/g, '$1');

    // Remove remaining triple/back ticks if any
    s = s.replace(/```/g, '');
    s = s.replace(/\s+```\s+/g, '\n');

    // Trim excessive blank lines
    s = s.replace(/\n{3,}/g, '\n\n');

    return s.trim();
}

function addSystemLog(message, type = 'info') {
    const logsContent = document.getElementById('logsContent');
    if (!logsContent) return;

    const placeholder = logsContent.querySelector('.log-placeholder');
    if (placeholder) {
        logsContent.removeChild(placeholder);
    }

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span>${message}`;
                    wrap.style.position = 'relative'; // Ensure wrap is positioned relative
                    const img = document.createElement('img'); 
    logsContent.appendChild(logEntry);
    logsContent.scrollTop = logsContent.scrollHeight;

    // Add activity indicator if logs are minimized
    const systemLogs = document.querySelector('.system-logs');
    if (systemLogs && systemLogs.classList.contains('minimized')) {
                    // Add a load/error badge (top-right)
                    const badge = document.createElement('span');
                    badge.className = 'artifact-badge artifact-badge-loading';
                    badge.title = 'Loading...';
                    wrap.appendChild(badge);
        systemLogs.classList.add('active');
        setTimeout(() => {
            systemLogs.classList.remove('active');
        }, 2000);
    }
}

let _lastLogTimestamp = '';

function pollSystemLogs() {
    // Include the last-seen timestamp so the server can return only newer logs.
    const sinceParam = _lastLogTimestamp ? `?since=${encodeURIComponent(_lastLogTimestamp)}` : '';
    fetch(`/api/system_logs${sinceParam}`)
        .then(response => {
            if (!response.ok) {
                console.warn('pollSystemLogs: non-OK response', response.status);
            }
            return response.json();
        })
        .then(data => {
            if (!data) return;

            // Support both { logs: [...] } and direct array responses for resilience
            const logs = Array.isArray(data.logs) ? data.logs : (Array.isArray(data) ? data : []);
            if (!logs || logs.length === 0) return;

            // If server returned logs already filtered by `since`, append all
            // Otherwise, fall back to client-side filtering by timestamp
            let newLogs = logs;
            if (!sinceParam) {
                newLogs = logs.filter(l => !_lastLogTimestamp || l.timestamp > _lastLogTimestamp);
            }

            if (newLogs.length > 0) {
                newLogs.forEach(log => {
                    try {
                        addSystemLog(log.message || JSON.stringify(log), log.type || 'info');
                    } catch (e) {
                        console.error('Error adding system log entry:', e, log);
                    }
                });
                // Update last-seen timestamp using the last returned log entry
                const last = newLogs[newLogs.length - 1];
                if (last && last.timestamp) _lastLogTimestamp = last.timestamp;
            }
        })
        .catch(error => {
            console.error('Error fetching system logs:', error);
        });
}