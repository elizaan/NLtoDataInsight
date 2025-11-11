// Chat Interface JavaScript - Exactly replicates run_conversation CLI flow
// VERSION: 20251105_v10 - FIXED: Animation grid padding dynamically adjusts with system logs resize

console.log('🔥🔥🔥 CHAT.JS VERSION 20251105_v10 - DYNAMIC GRID PADDING 🔥🔥🔥');

class ChatInterface {
    constructor() {
        console.log('ChatInterface constructor called');
        this.initializeElements();
        this.setupEventListeners();
        
        // Multi-panel animation management
        this.animationPanels = []; // Array of panel objects {id, path, frames, player}
        this.maxPanels = 4;
        this.nextPanelId = 1;
        
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
        this.pendingEvaluation = null; // Store evaluation to show after frames load
        this.pendingEvaluationPrompt = null; // Store a deferred continue_prompt for new-generation animations
        console.log('ChatInterface initialized successfully');
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        
        // Multi-panel elements
        this.animationGrid = document.getElementById('animationGrid');
        this.gridPlaceholder = document.getElementById('gridPlaceholder');
        
        // Legacy single-panel elements (for backward compatibility)
        this.animationViewer = document.getElementById('animationViewer');
        this.animationPlaceholder = document.querySelector('.animation-placeholder');
        this.animationPreview = document.getElementById('animationPreview');
        this.animationControls = document.getElementById('animationControls');
        this.animationStatus = document.getElementById('animationStatus');
        this.animationTitle = document.getElementById('animationTitle');

        console.log('Elements initialized:');
        console.log('- animationGrid:', this.animationGrid);
        console.log('- gridPlaceholder:', this.gridPlaceholder);
        console.log('- animationViewer:', this.animationViewer);
        console.log('- animationPlaceholder:', this.animationPlaceholder);
        console.log('- animationPreview:', this.animationPreview);
        console.log('- animationControls:', this.animationControls);
        console.log('- animationStatus:', this.animationStatus);
        console.log('- animationTitle:', this.animationTitle);

        // Additional debugging for animation controls
        if (this.animationControls) {
            console.log('✅ Animation controls element found');
            console.log('- Controls display style:', this.animationControls.style.display);
            console.log('- Controls computed style:', window.getComputedStyle(this.animationControls).display);
            console.log('- Controls parent:', this.animationControls.parentElement);
        } else {
            console.error('❌ Animation controls element NOT found');
            // Try to find it manually
            const manualSearch = document.querySelector('#animationControls');
            console.log('Manual search result:', manualSearch);
            const allAnimationElements = document.querySelectorAll('[id*="animation"]');
            console.log('All elements with "animation" in ID:', allAnimationElements);
        }
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

        // Setup progress slider interaction
        const progressSlider = document.getElementById('progressSlider');
        if (progressSlider) {
            progressSlider.addEventListener('input', (e) => {
                if (this.framePlayer && this.framePlayer.frames.length > 0) {
                    const progress = parseFloat(e.target.value);
                    const frameIndex = Math.round((progress / 100) * (this.framePlayer.frames.length - 1));
                    this.pauseAnimation(); // Pause when user manually seeks
                    this.displayFrame(frameIndex);
                }
            });
        }

        // Setup speed slider
        const speedSlider = document.getElementById('speedSlider');
        const speedLabel = document.getElementById('speedLabel');
        if (speedSlider && speedLabel) {
            speedSlider.addEventListener('input', (e) => {
                const speed = parseFloat(e.target.value);
                speedLabel.textContent = speed.toFixed(1) + 'x';
                if (this.framePlayer) {
                    this.framePlayer.playSpeed = 500 / speed; // Convert speed multiplier to delay
                }
            });
        }
        
        // Setup keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // ESC key closes fullscreen
            if (e.key === 'Escape' && this.fullscreenPanel) {
                this.closeFullscreen();
            }
        });
        
        // Setup resize handle for panel resizing
        this.setupPanelResize();
        
        // Setup resize handle for system logs
        this.setupLogsResize();
        
        // Initialize animation grid padding based on initial system logs height
        this.updateAnimationGridPadding();
    }
    
    updateAnimationGridPadding() {
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
                
                // IMPORTANT: Update animation grid bottom padding to match logs height
                const animationGrid = document.getElementById('animationGrid');
                if (animationGrid) {
                    // Add 20px buffer for visual spacing
                    animationGrid.style.paddingBottom = `${newHeight + 20}px`;
                }
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
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                resizeHandle.classList.remove('dragging');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
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
                await this.sendApiMessage(message, 'select_phenomenon');
            }
        } else if (this.conversationState === 'custom_description' || this.conversationState === 'phenomenon_selection') {
            // Directly post the custom phenomenon as the message
            await this.sendApiMessage(message, 'select_phenomenon');
        } else {
            // Fallback: if conversation state is unexpected, still attempt to send
            // as a custom description so UI input doesn't get ignored.
            await this.sendApiMessage(message, 'select_phenomenon');
        }
    }

    async sendApiMessage(message, action, extraData = {}) {
        try {
            this.updateStatus('Processing...', 'processing');

            const requestData = {
                message: message,
                action: action,
                ...extraData
            };
            // Respect the frontend LangChain toggle if present so tests can opt-in
            try {
                const lcToggle = document.getElementById('useLangChainToggle');
                if (lcToggle) {
                    requestData.use_langchain = !!lcToggle.checked;
                }
            } catch (e) {
                console.warn('Could not read useLangChainToggle state', e);
            }

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            console.log('Received response data:', data);

            if (data.status === 'error') {
                this.addMessage(`Error: ${data.message}`, 'bot');
                this.updateStatus('Error', 'error');
                return;
            }

            // Handle different response types exactly like run_conversation
            this.handleApiResponse(data);

        } catch (error) {
            console.error('Error:', error);
            this.addMessage('Sorry, there was an error processing your request.', 'bot');
            this.updateStatus('Error', 'error');
        }
    }

    handleApiResponse(data) {
        console.log('Handling API response:', data);
        console.log('Response type:', data.type);

        try {
            switch (data.type) {
                case 'dataset_selection':
                    // Show the exact same options as run_conversation
                    this.addTypingMessage(data.message, 'bot');
                    this.conversationState = 'phenomenon_selection';
                    break;

                case 'animation_generated':
                    // Animation generated, show evaluation and continue prompt (exactly like run_conversation)
                    console.log('Processing animation_generated case');
                    console.log('Data received:', JSON.stringify(data, null, 2));

                    try {
                        console.log('Step 1: Adding initial message');
                        this.addMessage(data.message, 'bot');
                        console.log('Step 1 completed successfully');

                        // Check intent_type for multi-panel logic
                        const intentType = data.intent_type || 'GENERATE_NEW';
                        console.log('🎯 Intent Type:', intentType);
                        
                        if (intentType === 'GENERATE_NEW') {
                            console.log('🆕 GENERATE_NEW detected - clearing all panels');
                            this.clearAllPanels();
                        }

                        // Detect if this is an existing animation or new generation
                        const isExistingAnimation = data.message && data.message.includes('Found existing animation. Reusing:');
                        console.log('Animation type detected:', isExistingAnimation ? 'EXISTING (pre-generated)' : 'NEW (being generated)');

                        console.log('Step 2: Attempting to show animation in panel');
                        // Try to show animation but don't let it break the flow
                        try {
                            // Use multi-panel system
                            this.addAnimationPanel(data.animation_path, isExistingAnimation, intentType);
                            console.log('Step 2 completed successfully');

                            // If server returned an evaluation directly, store it. Otherwise
                            // prompt the user whether they'd like an evaluation to avoid
                            // unnecessary OpenAI usage.
                            if (data.evaluation) {
                                this.pendingEvaluation = {
                                    evaluation: data.evaluation,
                                    continue_prompt: data.continue_prompt
                                };
                            } else if (data.evaluation_available) {
                                // If this is a pre-generated animation, show prompt
                                // immediately. If this is a new generation, defer the
                                // prompt until frames have been discovered so the
                                // user sees it after they can preview the animation.
                                const isNewGeneration = (isExistingAnimation === false);
                                if (isNewGeneration) {
                                    // Store the prompt and let showPendingEvaluation
                                    // display it once frames are available.
                                    this.pendingEvaluationPrompt = data.continue_prompt || "Would you like an automated evaluation and suggestions for this animation? (y/n)";
                                    addSystemLog('Deferred evaluation prompt until frames are available', 'info');
                                } else {
                                    // Ask the user if they'd like an evaluation now
                                    const promptHtml = `
                                        <div class="evaluation-prompt">
                                            <p>Would you like an automated evaluation and suggestions for this animation? This will use the LLM. </p>
                                            <button class="option-btn" id="evalYesBtn">Yes - Evaluate</button>
                                            <button class="option-btn" id="evalNoBtn">No - Skip</button>
                                        </div>
                                    `;
                                    this.addCustomContent(promptHtml);

                                    // Attach handlers
                                    setTimeout(() => {
                                        const yes = document.getElementById('evalYesBtn');
                                        const no = document.getElementById('evalNoBtn');
                                        if (yes) {
                                            yes.addEventListener('click', () => {
                                                addSystemLog('User requested on-demand evaluation', 'info');
                                                // Disable buttons to avoid double clicks
                                                yes.disabled = true;
                                                no.disabled = true;
                                                // Trigger evaluation API call
                                                evaluateLastAnimation();
                                            });
                                        }
                                        if (no) {
                                            no.addEventListener('click', () => {
                                                addSystemLog('User skipped evaluation', 'info');
                                                // Remove prompt or show follow-up
                                                no.disabled = true;
                                                yes.disabled = true;
                                                chatInterface.addMessage('Okay — evaluation skipped. You can request it later.', 'bot');
                                            });
                                        }
                                    }, 50);
                                }
                            }
                            console.log('Stored pending evaluation to show after frames load');
                            console.log('Pending evaluation data:', this.pendingEvaluation);
                            console.log('Continue prompt value:', data.continue_prompt);

                        } catch (animError) {
                            console.error('Animation display error (non-critical):', animError);
                            console.log('Step 2 failed but continuing');
                            this.addTypingMessage('Animation could not be displayed, but processing continues.', 'bot');

                            // Show evaluation immediately if animation failed
                            this.addTypingMessage(`Evaluation:\n${data.evaluation}`, 'bot', 30, () => {
                                // Show continue prompt after evaluation typing completes
                                this.addTypingMessage(data.continue_prompt, 'bot');
                            });
                        }

                        console.log('Step 5: Updating state and status');
                        this.conversationState = 'conversation_loop';
                        this.updateStatus('Animation Ready', 'success');
                        console.log('Step 5 completed successfully');

                        console.log('animation_generated case completed successfully');
                    } catch (caseError) {
                        console.error('Error in animation_generated case:', caseError);
                        console.error('Error stack:', caseError.stack);
                        throw caseError;
                    }
                    break;

                case 'animation_refined':
                    // Refined animation (exactly like run_conversation y/g responses)
                    this.addMessage(data.message, 'bot');

                    // This is always a new generation since user requested refinement
                    const isExistingRefinedAnimation = false;
                    this.showAnimation(data.animation_path, isExistingRefinedAnimation);
                    this.addTypingMessage(`Evaluation:\n${data.evaluation}`, 'bot', 30, () => {
                        // Show continue prompt after evaluation typing completes
                        this.addTypingMessage(data.continue_prompt, 'bot');
                    });
                    this.conversationState = 'conversation_loop';
                    this.updateStatus('Animation Updated', 'success');
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
            this.sendApiMessage(desc, 'select_phenomenon');
        }
    }

    showAnimation(animationPath, isExistingAnimation = null) {
        console.log('showAnimation called with path:', animationPath);
        console.log('Animation type:', isExistingAnimation === true ? 'EXISTING' : isExistingAnimation === false ? 'NEW' : 'UNKNOWN');

        if (animationPath) {
            // ALWAYS assume it's frame-based animation in Rendered_frames/
            // Extract the base path for frames regardless of GIF or not
            let frameBasePath;

            if (animationPath.includes('/Animation/')) {
                // Convert from: /api/animations/animation_xxx/Animation/animation_xxx.gif
                // To: /api/animations/animation_xxx/Rendered_frames/
                const pathParts = animationPath.split('/Animation/');
                frameBasePath = pathParts[0] + '/Rendered_frames/';
            } else {
                // Direct path - assume it points to the animation folder
                frameBasePath = animationPath.endsWith('/') ? animationPath + 'Rendered_frames/' : animationPath + '/Rendered_frames/';
            }

            console.log('Frame base path:', frameBasePath);

            // Initialize the frame player
            this.initializeFramePlayer(frameBasePath, isExistingAnimation);
        } else {
            console.log('No animation path provided');
        }
    }

    initializeFramePlayer(frameBasePath, isExistingAnimation = null) {
        console.log('Initializing frame player with base path:', frameBasePath);
        console.log('Animation strategy:', isExistingAnimation === true ? 'LOAD_ALL_EXISTING' : isExistingAnimation === false ? 'REAL_TIME_GENERATION' : 'AUTO_DETECT');

        // Check if required DOM elements exist
        if (!this.animationPlaceholder || !this.animationPreview || !this.animationControls || !this.animationTitle) {
            console.error('Required animation DOM elements not found');
            console.log('animationPlaceholder:', this.animationPlaceholder);
            console.log('animationPreview:', this.animationPreview);
            console.log('animationControls:', this.animationControls);
            console.log('animationTitle:', this.animationTitle);
            throw new Error('Animation DOM elements not available');
        }

        // Animation player state
        this.framePlayer = {
            basePath: frameBasePath,
            frames: [],
            currentFrame: 0,
            isPlaying: false,
            playInterval: null,
            playSpeed: 500, // milliseconds between frames
            loadedFrames: new Set(),
            isDiscovering: false, // Prevent concurrent discovery
            isExistingAnimation: isExistingAnimation // Store animation type
        };

        // Show animation area with null checks - controls now embedded in viewer
        if (this.animationPlaceholder) this.animationPlaceholder.style.display = 'none';
        if (this.animationPreview) this.animationPreview.style.display = 'block';
        if (this.animationControls) {
            console.log('🎮 Showing animation controls (now inside viewer)');
            this.animationControls.style.display = 'block';
            console.log('✅ Animation controls shown');
        } else {
            console.error('❌ Animation controls element not found!');
        }

        // Ensure system logs remain visible during animation loading
        const systemLogs = document.getElementById('systemLogs');
        if (systemLogs) {
            systemLogs.style.display = 'block';
            console.log('✅ System logs kept visible during animation initialization');
        }

        // Set appropriate loading message based on animation type
        if (this.animationTitle) {
            if (isExistingAnimation === true) {
                this.animationTitle.textContent = 'Loading Pre-generated Animation...';
            } else if (isExistingAnimation === false) {
                this.animationTitle.textContent = 'Monitoring Frame Generation...';
            } else {
                this.animationTitle.textContent = 'Loading Animation Frames...';
            }
        }

        // Choose discovery strategy based on animation type
        if (isExistingAnimation === true) {
            // For existing animations, do aggressive bulk loading
            console.log('🎬 Starting bulk loading for existing animation');
            this.discoverAllExistingFrames();
        } else if (isExistingAnimation === false) {
            // For new animations, wait for server to finish saving data before polling frames
            console.log('📹 New animation generation detected; waiting for data to be ready before polling frames');
            this.showSystemLogs();
            // Poll system logs for a readiness marker, then start discovery
            this.waitForDataReady().then(() => {
                console.log('Data ready detected; starting real-time monitoring for new animation');
                this.startFrameDiscovery();
            }).catch((err) => {
                console.warn('Data readiness wait timed out or failed; starting monitoring anyway', err);
                this.startFrameDiscovery();
            });
        } else {
            // Auto-detect mode - try existing first, then start monitoring
            console.log('🔍 Auto-detect mode - checking for existing frames first');
            this.showSystemLogs();
            this.discoverFrames();
        }
    }

    // Poll system logs until a readiness marker is found or timeout
    async waitForDataReady(timeoutSeconds = 15) {
        const start = Date.now();
        const timeout = timeoutSeconds * 1000;
        const readinessMarkers = [
            'Animation generation started',
            'Animation generation completed'
        ];

        // Helper to fetch recent system logs
        const fetchLogs = async () => {
            try {
                const resp = await fetch('/api/system_logs');
                if (!resp.ok) return [];
                const data = await resp.json();
                return data.logs || [];
            } catch (e) {
                console.warn('Failed to fetch system logs:', e);
                return [];
            }
        };

        while ((Date.now() - start) < timeout) {
            const logs = await fetchLogs();
            for (const l of logs) {
                const msg = (l.message || '').toString();
                for (const m of readinessMarkers) {
                    if (msg.includes(m)) {
                        return true;
                    }
                }
            }
            // Wait a short interval before polling again
            await new Promise(r => setTimeout(r, 200));
        }
        throw new Error('Timeout waiting for data readiness');
    }

    async discoverFrames() {
        if (this.framePlayer.isDiscovering) {
            console.log('Discovery already in progress, skipping...');
            return;
        }

        this.framePlayer.isDiscovering = true;

        // Auto-minimize system logs when starting frame discovery
        minimizeSystemLogs();

        console.log('Discovering existing frames...');
        console.log('Frame base path:', this.framePlayer.basePath);

        // Aggressive discovery for existing animations - try to find ALL frames
        let frameIndex = 0;
        let consecutiveFails = 0;
        const maxConsecutiveFails = 5; // More generous for existing animations
        let foundAny = false;

        while (consecutiveFails < maxConsecutiveFails) {
            // Skip if frame is already loaded
            if (this.framePlayer.loadedFrames.has(frameIndex)) {
                frameIndex++;
                continue;
            }

            // Always try all possible patterns - no detection needed
            const patterns = [
                `img_kf${frameIndex}f${frameIndex}.png`,
                `img_kf${frameIndex}f${frameIndex + 1}.png`,
                `img_${frameIndex}_f${frameIndex}.png`,
                `img_${frameIndex}_f${frameIndex + 1}.png`
            ];

            let found = false;
            for (const pattern of patterns) {
                const framePath = `${this.framePlayer.basePath}${pattern}`;

                try {
                    const exists = await this.checkFrameExists(framePath);
                    if (exists) {
                        this.framePlayer.frames.push({
                            index: frameIndex,
                            path: framePath,
                            pattern: pattern
                        });
                        this.framePlayer.loadedFrames.add(frameIndex);
                        console.log(`✅ Found existing frame ${frameIndex}: ${pattern}`);

                        // Show first frame immediately
                        if (frameIndex === 0) {
                            this.displayFrame(0);
                        }

                        // Update title
                        this.animationTitle.textContent = `Animation (${this.framePlayer.frames.length} frames)`;

                        found = true;
                        foundAny = true;
                        consecutiveFails = 0;
                        break;
                    }
                } catch (error) {
                    console.error(`Error checking pattern ${pattern}:`, error);
                }
            }

            if (!found) {
                consecutiveFails++;
                console.log(`❌ No existing frame found for index ${frameIndex} (${consecutiveFails}/${maxConsecutiveFails})`);
            }

            frameIndex++;

            // Small delay to avoid overwhelming the server
            await new Promise(resolve => setTimeout(resolve, 30));
        }

        this.framePlayer.isDiscovering = false;

        if (foundAny) {
            console.log(`✅ Existing frame discovery complete! Found ${this.framePlayer.frames.length} pre-generated frames`);

            // Show pending evaluation after frames are loaded
            this.showPendingEvaluation();

            if (this.framePlayer.frames.length > 10) {
                console.log('🎬 Large animation detected - likely pre-generated. Starting periodic check for new frames...');
                // Start periodic discovery for potential new frames
                this.startFrameDiscovery();
            } else {
                console.log('🆕 Small animation detected - likely being generated. Starting active frame monitoring...');
                // Start more frequent monitoring for new generation
                this.startFrameDiscovery();
            }
        } else {
            console.log('📹 No existing frames found - animation is being generated. Starting real-time monitoring...');
            // Start immediate monitoring for new generation
            this.startFrameDiscovery();
        }
    }

    async discoverAllExistingFrames() {
        if (this.framePlayer.isDiscovering) {
            console.log('Discovery already in progress, skipping...');
            return;
        }

        this.framePlayer.isDiscovering = true;

        // Auto-minimize system logs when starting frame discovery/animation
        minimizeSystemLogs();

        console.log('🎬 BULK LOADING: Discovering all existing frames...');
        console.log('Frame base path:', this.framePlayer.basePath);

        // VERY aggressive discovery for existing animations - load ALL frames quickly
        let frameIndex = 0;
        let consecutiveFails = 0;
        const maxConsecutiveFails = 3; // Quick stop for existing animations - they should be continuous
        let foundAny = false;

        console.log('Starting aggressive bulk frame discovery...');

        // Add system log for frame loading start
        if (typeof addSystemLog === 'function') {
            addSystemLog('Loading existing animation frames...', 'info');
        }

        // Ensure system logs are visible during bulk loading
        const systemLogs = document.getElementById('systemLogs');
        if (systemLogs) {
            systemLogs.style.display = 'block';
            console.log('System logs ensured visible at start of bulk loading');
        }
        while (consecutiveFails < maxConsecutiveFails) {
            // Skip if frame is already loaded
            if (this.framePlayer.loadedFrames.has(frameIndex)) {
                frameIndex++;
                continue;
            }

            // Always try all possible patterns - no detection needed
            const patterns = [
                `img_kf${frameIndex}f${frameIndex}.png`,
                `img_kf${frameIndex}f${frameIndex + 1}.png`,
                `img_${frameIndex}_f${frameIndex}.png`,
                `img_${frameIndex}_f${frameIndex + 1}.png`
            ];

            let foundInThisIndex = false;
            let framesFoundInThisIndex = 0;

            // Check both patterns for this frame index
            for (const pattern of patterns) {
                const framePath = `${this.framePlayer.basePath}${pattern}`;

                try {
                    const exists = await this.checkFrameExists(framePath);
                    if (exists) {
                        // Calculate the actual frame number for display
                        const actualFrameNumber = this.framePlayer.frames.length;

                        this.framePlayer.frames.push({
                            index: actualFrameNumber,
                            baseIndex: frameIndex, // Store the base index (0-59)
                            path: framePath,
                            pattern: pattern
                        });
                        this.framePlayer.loadedFrames.add(frameIndex);

                        // Less verbose logging for bulk loading
                        if (actualFrameNumber % 20 === 0 || actualFrameNumber < 5) {
                            console.log(`✅ Bulk loaded frame ${actualFrameNumber}: ${pattern}`);
                        }

                        // Show each frame as it's discovered for visual feedback (5 FPS preview)
                        this.displayFrame(actualFrameNumber);

                        // Update title for every frame to show progress
                        this.animationTitle.textContent = `Loading Animation (${this.framePlayer.frames.length} frames loaded)`;

                        // Add system log every 10 frames to show progress
                        if (typeof addSystemLog === 'function' && actualFrameNumber % 10 === 0 && actualFrameNumber > 0) {
                            addSystemLog(`Loaded ${this.framePlayer.frames.length} frames...`, 'info');
                        }

                        foundInThisIndex = true;
                        foundAny = true;
                        framesFoundInThisIndex++;

                        // Add delay to show each frame at ~5 FPS for visual feedback
                        await new Promise(resolve => setTimeout(resolve, 200));
                    }
                } catch (error) {
                    console.error(`Error checking pattern ${pattern}:`, error);
                }
            }

            if (!foundInThisIndex) {
                consecutiveFails++;
                if (frameIndex % 10 === 0) {
                    console.log(`❌ No frames found for index ${frameIndex} (${consecutiveFails}/${maxConsecutiveFails})`);
                }
            } else {
                consecutiveFails = 0 // Reset on any success
            }

            frameIndex++;

            // Minimal delay for existing animations - they're already there
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        this.framePlayer.isDiscovering = false;

        if (foundAny) {
            console.log(`🎬 BULK LOADING COMPLETE! Found ${this.framePlayer.frames.length} pre-generated frames`);
            this.animationTitle.textContent = `Animation Ready (${this.framePlayer.frames.length} frames)`;

            // Add system log for completion
            if (typeof addSystemLog === 'function') {
                addSystemLog(`Animation loaded: ${this.framePlayer.frames.length} frames ready for playback`, 'info');
            }

            // Show pending evaluation after frames are loaded
            this.showPendingEvaluation();

            // Ensure system logs are still visible after loading
            const systemLogs = document.getElementById('systemLogs');
            if (systemLogs) {
                systemLogs.style.display = 'block';
                console.log('✅ System logs ensured visible after bulk loading completion');
            }

            // No need for periodic discovery - all frames are already loaded
        } else {
            console.log('❌ No existing frames found in bulk loading mode');
            this.animationTitle.textContent = 'No animation frames found';
        }
    }

    async discoverNewFrames() {
        console.log('🚀 UPDATED discoverNewFrames() v2.0 - Using baseIndex logic');
        
        if (this.framePlayer.isDiscovering) {
            console.log('Discovery already in progress, skipping new frame check...');
            return;
        }

        this.framePlayer.isDiscovering = true;
        
        // Find the highest baseIndex (frame number) we've loaded so far
        let maxBaseIndex = -1;
        for (const frame of this.framePlayer.frames) {
            if (frame.baseIndex !== undefined && frame.baseIndex > maxBaseIndex) {
                maxBaseIndex = frame.baseIndex;
            }
        }
        
        const nextFrameIndex = maxBaseIndex + 1;
        console.log(`🔍 Checking for new frames starting from baseIndex ${nextFrameIndex} (total frames in array: ${this.framePlayer.frames.length})`);

        // OPTIMIZED: Parallel HEAD requests for speed
        let foundAny = false;
        const maxCheck = 10; // Check frames ahead in parallel

        // Prepare all frame checks to run in parallel
        const frameChecks = [];
        for (let i = nextFrameIndex; i < nextFrameIndex + maxCheck; i++) {
            if (!this.framePlayer.loadedFrames.has(i)) {
                // Always try all possible patterns - no detection needed
                const patterns = [
                    `img_kf${i}f${i}.png`,
                    `img_kf${i}f${i + 1}.png`,
                    `img_${i}_f${i}.png`,
                    `img_${i}_f${i + 1}.png`
                ];
                frameChecks.push({ i, patterns });
            }
        }

        // Execute all HEAD requests in parallel
        const results = await Promise.all(frameChecks.map(async ({ i, patterns }) => {
            for (const pattern of patterns) {
                const framePath = `${this.framePlayer.basePath}${pattern}`;
                const exists = await this.checkFrameExists(framePath);
                if (exists) {
                    return { found: true, i, framePath, pattern };
                }
            }
            return { found: false, i };
        }));

        // Process results in order
        for (const result of results) {
            if (result.found) {
                const displayIndex = this.framePlayer.frames.length;
                this.framePlayer.frames.push({
                    index: displayIndex,
                    baseIndex: result.i,
                    path: result.framePath,
                    pattern: result.pattern
                });
                this.framePlayer.loadedFrames.add(result.i);
                
                console.log(`🆕 Discovered new frame baseIndex ${result.i} -> displayIndex ${displayIndex}: ${result.pattern}`);
                
                // Display first frame or latest frame immediately
                if (displayIndex === 0 || !this.framePlayer.isPlaying) {
                    this.displayFrame(displayIndex);
                }
                
                // Update title
                if (this.animationTitle) {
                    this.animationTitle.textContent = `Animation (${this.framePlayer.frames.length} frames)`;
                }
                
                foundAny = true;
                
                // Track last discovery time for adaptive polling
                this.framePlayer.lastDiscoveryTime = Date.now();
            }
        }

        // Stop auto-discovery after 3 consecutive failures (same as multi-panel)
        if (foundAny) {
            // Found frames - reset failure counter
            this.framePlayer.consecutiveFailures = 0;
        } else if (this.framePlayer.frames.length > 0) {
            // No frames found - increment failure counter
            this.framePlayer.consecutiveFailures = (this.framePlayer.consecutiveFailures || 0) + 1;
            
            // Stop after 3 consecutive failures
            if (this.framePlayer.consecutiveFailures >= 3) {
                console.log('✅ No new frames detected for 3 consecutive checks - animation complete');
                console.log(`📊 Final frame count: ${this.framePlayer.frames.length} frames`);
                
                this.framePlayer.isDiscovering = false;
                if (this.frameDiscoveryInterval) {
                    clearTimeout(this.frameDiscoveryInterval);
                    this.frameDiscoveryInterval = null;
                    console.log('🛑 Frame discovery stopped');
                }
                
                // Update title to show completion
                if (this.animationTitle) {
                    this.animationTitle.textContent = `Animation Complete (${this.framePlayer.frames.length} frames)`;
                }
                
                // Show pending evaluation now that animation is complete
                this.showPendingEvaluation();
                
                return;
            } else {
                console.log(`⏳ No frames found (failure ${this.framePlayer.consecutiveFailures}/3), will check again...`);
            }
        }

        this.framePlayer.isDiscovering = false;
    }

    startFrameDiscovery() {
        console.log('🔍 Starting periodic frame discovery...');
        // Clear any existing interval
        if (this.frameDiscoveryInterval) {
            clearTimeout(this.frameDiscoveryInterval);
            this.frameDiscoveryInterval = null;
        }

        // OPTIMIZED: Adaptive polling interval
        // Start aggressive (500ms) during active generation, slow down when idle
        const adaptiveDiscovery = async () => {
            // Check if discovery was stopped
            if (!this.framePlayer || this.frameDiscoveryInterval === null) {
                console.log('🛑 Discovery was stopped, not scheduling next check');
                return;
            }
            
            await this.discoverNewFrames();
            
            // Check again after discoverNewFrames (it might have stopped discovery)
            if (!this.framePlayer || this.frameDiscoveryInterval === null) {
                console.log('🛑 Discovery stopped after frame check');
                return;
            }
            
            // Adaptive interval based on recent discoveries
            let interval;
            if (this.framePlayer.lastDiscoveryTime && (Date.now() - this.framePlayer.lastDiscoveryTime) < 5000) {
                // Found frames recently - keep aggressive polling
                interval = 500; // 500ms for active generation
            } else if (this.framePlayer.frames.length === 0) {
                // No frames yet - wait for first frame
                interval = 1000;
            } else {
                // No recent activity - slow down
                interval = 3000;
            }
            
            this.frameDiscoveryInterval = setTimeout(adaptiveDiscovery, interval);
        };

        // Start immediately
        adaptiveDiscovery();
    }

    stopFrameDiscovery() {
        console.log('🛑 Stopping frame discovery');
        if (this.frameDiscoveryInterval) {
            clearTimeout(this.frameDiscoveryInterval);
            this.frameDiscoveryInterval = null;
        }
    }

    async checkFrameExists(framePath) {
        try {
            const response = await fetch(framePath, { method: 'HEAD' });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    displayFrame(frameIndex) {
        if (this.framePlayer && this.framePlayer.frames[frameIndex]) {
            const frame = this.framePlayer.frames[frameIndex];
            this.animationPreview.src = frame.path;
            this.framePlayer.currentFrame = frameIndex;

            // Update frame counter - use correct ID from HTML
            const frameInfo = document.getElementById('frameInfo');
            if (frameInfo) {
                frameInfo.textContent = `Frame ${frameIndex + 1} / ${this.framePlayer.frames.length}`;
            }

            // Update progress slider
            const progressSlider = document.getElementById('progressSlider');
            if (progressSlider && this.framePlayer.frames.length > 0) {
                const progress = (frameIndex / (this.framePlayer.frames.length - 1)) * 100;
                progressSlider.value = progress;
                progressSlider.max = 100;
            }

            console.log(`Displaying frame ${frameIndex}: ${frame.path}`);
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

    // ========================================
    // MULTI-PANEL ANIMATION MANAGEMENT
    // ========================================
    
    clearAllPanels() {
        console.log('🗑️ Clearing all animation panels');
        this.animationPanels = [];
        if (this.animationGrid) {
            // Remove all panel items
            const panels = this.animationGrid.querySelectorAll('.animation-panel-item');
            panels.forEach(panel => panel.remove());
            
            // Show placeholder
            if (this.gridPlaceholder) {
                this.gridPlaceholder.style.display = 'flex';
            }
            
            // Reset grid class
            this.animationGrid.className = 'animation-grid';
        }
    }
    
    addAnimationPanel(animationPath, isExistingAnimation, intentType) {
        console.log('📊 Adding animation panel:', {animationPath, isExistingAnimation, intentType});
        
        // Check if we've reached max panels (FIFO: remove oldest)
        if (this.animationPanels.length >= this.maxPanels) {
            console.log('⚠️ Max panels reached, removing oldest panel');
            const oldestPanel = this.animationPanels.shift();
            const panelElement = document.getElementById(`panel-${oldestPanel.id}`);
            if (panelElement) {
                panelElement.remove();
            }
        }
        
        // Extract frame base path
        let frameBasePath;
        if (animationPath.includes('/Animation/')) {
            const pathParts = animationPath.split('/Animation/');
            frameBasePath = pathParts[0] + '/Rendered_frames/';
        } else {
            frameBasePath = animationPath.endsWith('/') ? animationPath + 'Rendered_frames/' : animationPath + '/Rendered_frames/';
        }
        
        // Create panel object
        const panelId = this.nextPanelId++;
        const panel = {
            id: panelId,
            path: animationPath,
            frameBasePath: frameBasePath,
            frames: [],
            currentFrame: 0,
            isPlaying: false,
            playInterval: null,
            playSpeed: 500,
            isExistingAnimation: isExistingAnimation,
            intentType: intentType
        };
        
        this.animationPanels.push(panel);
        
        // Hide placeholder
        if (this.gridPlaceholder) {
            this.gridPlaceholder.style.display = 'none';
        }
        
        // Create and insert panel DOM element
        this.createPanelElement(panel);
        
        // Update grid layout class
        this.updateGridLayout();
        
        // Start loading frames for this panel
        this.initializePanelFrames(panel);
    }
    
    createPanelElement(panel) {
        const panelElement = document.createElement('div');
        panelElement.className = 'animation-panel-item';
        panelElement.id = `panel-${panel.id}`;
        
        const title = panel.intentType === 'MODIFY_EXISTING' ? 
            `Modified Animation #${panel.id}` : 
            `Animation #${panel.id}`;
        
        panelElement.innerHTML = `
            <div class="panel-header">
                <h4 class="panel-title">${title}</h4>
                <div class="panel-actions">
                    <button class="panel-action-btn" onclick="chatInterface.togglePanelFullscreen(${panel.id})" title="Fullscreen">
                        <i class="fas fa-expand"></i>
                    </button>
                    <button class="panel-action-btn" onclick="chatInterface.removePanel(${panel.id})" title="Remove">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
            <div class="panel-viewer" id="panel-viewer-${panel.id}">
                <canvas class="panel-canvas" id="panel-canvas-${panel.id}"></canvas>
            </div>
            <div class="panel-controls">
                <div class="panel-control-buttons">
                    <button class="panel-control-btn" id="panel-play-${panel.id}" onclick="chatInterface.togglePanelPlay(${panel.id})">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="panel-control-btn" onclick="chatInterface.stopPanel(${panel.id})">
                        <i class="fas fa-stop"></i>
                    </button>
                    <button class="panel-control-btn" onclick="chatInterface.prevPanelFrame(${panel.id})">
                        <i class="fas fa-step-backward"></i>
                    </button>
                    <button class="panel-control-btn" onclick="chatInterface.nextPanelFrame(${panel.id})">
                        <i class="fas fa-step-forward"></i>
                    </button>
                </div>
                <div class="panel-progress">
                    <input type="range" id="panel-slider-${panel.id}" min="0" max="100" value="0" 
                           oninput="chatInterface.seekPanelFrame(${panel.id}, this.value)" />
                    <span class="panel-frame-info" id="panel-info-${panel.id}">Frame 0 / 0</span>
                </div>
            </div>
        `;
        
        this.animationGrid.appendChild(panelElement);
    }
    
    updateGridLayout() {
        const count = this.animationPanels.length;
        this.animationGrid.className = `animation-grid panel-count-${count}`;
        console.log(`🎨 Updated grid layout for ${count} panel(s)`);
    }
    
    initializePanelFrames(panel) {
        console.log(`🎬 Initializing frames for panel ${panel.id}`);
        
        if (panel.isExistingAnimation) {
            this.discoverAllPanelFrames(panel);
        } else {
            this.startPanelFrameDiscovery(panel);
        }
    }
    
    async discoverAllPanelFrames(panel) {
        console.log(`📥 Bulk loading frames for panel ${panel.id}`);
        console.log(`Frame base path for panel ${panel.id}:`, panel.frameBasePath);
        
        const canvas = document.getElementById(`panel-canvas-${panel.id}`);
        if (!canvas) return;
        
        // Use same robust discovery as main frame player
        let frameIndex = 0;
        let consecutiveFails = 0;
        const maxConsecutiveFails = 3;
        let foundAny = false;
        
        console.log(`Starting aggressive bulk frame discovery for panel ${panel.id}...`);
        
        while (consecutiveFails < maxConsecutiveFails) {
            // Try all possible frame naming patterns (same as main player)
            const patterns = [
                `img_kf${frameIndex}f${frameIndex}.png`,
                `img_kf${frameIndex}f${frameIndex + 1}.png`,
                `img_${frameIndex}_f${frameIndex}.png`,
                `img_${frameIndex}_f${frameIndex + 1}.png`
            ];
            
            let foundInThisIndex = false;
            
            // Check all patterns for this frame index
            for (const pattern of patterns) {
                const framePath = `${panel.frameBasePath}${pattern}`;
                
                try {
                    const response = await fetch(framePath, {method: 'HEAD'});
                    if (response.ok) {
                        const actualFrameNumber = panel.frames.length;
                        
                        panel.frames.push({
                            index: actualFrameNumber,
                            baseIndex: frameIndex,
                            url: framePath,
                            pattern: pattern
                        });
                        
                        // Log progress every 20 frames or first 5
                        if (actualFrameNumber % 20 === 0 || actualFrameNumber < 5) {
                            console.log(`✅ Panel ${panel.id} bulk loaded frame ${actualFrameNumber}: ${pattern}`);
                        }
                        
                        // Display frame as it's discovered for visual feedback
                        this.displayPanelFrame(panel, actualFrameNumber);
                        
                        foundInThisIndex = true;
                        foundAny = true;
                        
                        // Small delay for visual feedback
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }
                } catch (error) {
                    // Frame doesn't exist, continue checking other patterns
                }
            }
            
            if (!foundInThisIndex) {
                consecutiveFails++;
                if (frameIndex % 10 === 0) {
                    console.log(`❌ Panel ${panel.id}: No frames found for index ${frameIndex} (${consecutiveFails}/${maxConsecutiveFails})`);
                }
            } else {
                consecutiveFails = 0; // Reset on success
            }
            
            frameIndex++;
        }
        
        console.log(`✅ Panel ${panel.id} loaded ${panel.frames.length} frames total`);
        
        if (panel.frames.length > 0) {
            // Display first frame
            this.displayPanelFrame(panel, 0);
            this.updatePanelInfo(panel);
        } else {
            console.warn(`⚠️ Panel ${panel.id}: No frames found!`);
        }
    }
    
    async tryLoadPanelFrame(panel, baseIndex, timeIndex) {
        const frameUrl = `${panel.frameBasePath}img_${baseIndex}_f${timeIndex}.png`;
        try {
            const response = await fetch(frameUrl, {method: 'HEAD'});
            if (response.ok) {
                return {url: frameUrl, baseIndex, timeIndex};
            }
        } catch (e) {
            // Frame doesn't exist
        }
        return null;
    }
    
    startPanelFrameDiscovery(panel) {
        console.log(`📹 Starting real-time frame discovery for panel ${panel.id}`);
        // Use adaptive discovery similar to main player
        this.discoverPanelFramesRecursive(panel);
    }
    
    async discoverPanelFramesRecursive(panel) {
        // Find the highest baseIndex we've loaded so far
        let maxBaseIndex = -1;
        for (const frame of panel.frames) {
            if (frame.baseIndex > maxBaseIndex) {
                maxBaseIndex = frame.baseIndex;
            }
        }
        
        const nextFrameIndex = maxBaseIndex + 1;
        console.log(`🔍 Panel ${panel.id}: Checking for frames starting from baseIndex ${nextFrameIndex}`);
        
        // Check multiple frames ahead in parallel
        let foundAny = false;
        const maxCheck = 10;
        
        const frameChecks = [];
        for (let i = nextFrameIndex; i < nextFrameIndex + maxCheck; i++) {
            // Try all possible patterns
            const patterns = [
                `img_kf${i}f${i}.png`,
                `img_kf${i}f${i + 1}.png`,
                `img_${i}_f${i}.png`,
                `img_${i}_f${i + 1}.png`
            ];
            frameChecks.push({ i, patterns });
        }
        
        // Execute all HEAD requests in parallel
        const results = await Promise.all(frameChecks.map(async ({ i, patterns }) => {
            for (const pattern of patterns) {
                const framePath = `${panel.frameBasePath}${pattern}`;
                try {
                    const response = await fetch(framePath, {method: 'HEAD'});
                    if (response.ok) {
                        return { found: true, i, framePath, pattern };
                    }
                } catch (e) {
                    // Frame doesn't exist
                }
            }
            return { found: false, i };
        }));
        
        // Process results in order
        for (const result of results) {
            if (result.found) {
                const displayIndex = panel.frames.length;
                panel.frames.push({
                    index: displayIndex,
                    baseIndex: result.i,
                    url: result.framePath,
                    pattern: result.pattern
                });
                
                console.log(`🆕 Panel ${panel.id}: Discovered frame baseIndex ${result.i} -> displayIndex ${displayIndex}`);
                
                // Display first frame immediately
                if (displayIndex === 0 || !panel.isPlaying) {
                    this.displayPanelFrame(panel, displayIndex);
                }
                
                this.updatePanelInfo(panel);
                foundAny = true;
            }
        }
        
        // Continue checking with adaptive polling
        if (foundAny) {
            // Found frames - check again soon
            panel.consecutiveFailures = 0;
            setTimeout(() => this.discoverPanelFramesRecursive(panel), 500);
        } else {
            // No new frames found - increment failure counter
            panel.consecutiveFailures = (panel.consecutiveFailures || 0) + 1;
            
            // Stop after 3 consecutive failures (no frames found in 3 checks)
            if (panel.consecutiveFailures >= 3) {
                console.log(`✅ Panel ${panel.id}: Frame discovery completed. Total frames: ${panel.frames.length}`);
                panel.frameDiscoveryComplete = true;
                return;
            }
            
            // Not done yet - check again after longer delay
            setTimeout(() => this.discoverPanelFramesRecursive(panel), 2000);
        }
    }
    
    displayPanelFrame(panel, frameIndex) {
        const canvas = document.getElementById(`panel-canvas-${panel.id}`);
        if (!canvas || !panel.frames[frameIndex]) return;
        
        const ctx = canvas.getContext('2d');
        const frame = panel.frames[frameIndex];
        const img = new Image();
        
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        
        img.src = frame.url;
        panel.currentFrame = frameIndex;
        this.updatePanelInfo(panel);
    }
    
    updatePanelInfo(panel) {
        const infoSpan = document.getElementById(`panel-info-${panel.id}`);
        const slider = document.getElementById(`panel-slider-${panel.id}`);
        
        if (infoSpan) {
            infoSpan.textContent = `Frame ${panel.currentFrame + 1} / ${panel.frames.length}`;
        }
        if (slider) {
            slider.max = panel.frames.length - 1;
            slider.value = panel.currentFrame;
        }
    }
    
    // Panel control methods
    togglePanelPlay(panelId) {
        const panel = this.animationPanels.find(p => p.id === panelId);
        if (!panel) return;
        
        panel.isPlaying = !panel.isPlaying;
        const playBtn = document.getElementById(`panel-play-${panelId}`);
        
        if (panel.isPlaying) {
            playBtn.innerHTML = '<i class="fas fa-pause"></i>';
            this.playPanelAnimation(panel);
        } else {
            playBtn.innerHTML = '<i class="fas fa-play"></i>';
            if (panel.playInterval) {
                clearInterval(panel.playInterval);
                panel.playInterval = null;
            }
        }
    }
    
    playPanelAnimation(panel) {
        panel.playInterval = setInterval(() => {
            panel.currentFrame = (panel.currentFrame + 1) % panel.frames.length;
            this.displayPanelFrame(panel, panel.currentFrame);
        }, panel.playSpeed);
    }
    
    stopPanel(panelId) {
        const panel = this.animationPanels.find(p => p.id === panelId);
        if (!panel) return;
        
        panel.isPlaying = false;
        if (panel.playInterval) {
            clearInterval(panel.playInterval);
            panel.playInterval = null;
        }
        panel.currentFrame = 0;
        this.displayPanelFrame(panel, 0);
        
        const playBtn = document.getElementById(`panel-play-${panelId}`);
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
    
    prevPanelFrame(panelId) {
        const panel = this.animationPanels.find(p => p.id === panelId);
        if (!panel) return;
        
        panel.currentFrame = (panel.currentFrame - 1 + panel.frames.length) % panel.frames.length;
        this.displayPanelFrame(panel, panel.currentFrame);
    }
    
    nextPanelFrame(panelId) {
        const panel = this.animationPanels.find(p => p.id === panelId);
        if (!panel) return;
        
        panel.currentFrame = (panel.currentFrame + 1) % panel.frames.length;
        this.displayPanelFrame(panel, panel.currentFrame);
    }
    
    seekPanelFrame(panelId, value) {
        const panel = this.animationPanels.find(p => p.id === panelId);
        if (!panel) return;
        
        const frameIndex = parseInt(value);
        this.displayPanelFrame(panel, frameIndex);
    }
    
    removePanel(panelId) {
        const panelIndex = this.animationPanels.findIndex(p => p.id === panelId);
        if (panelIndex === -1) return;
        
        const panel = this.animationPanels[panelIndex];
        
        // Clean up
        if (panel.playInterval) {
            clearInterval(panel.playInterval);
        }
        
        // Remove from array
        this.animationPanels.splice(panelIndex, 1);
        
        // Remove DOM element
        const panelElement = document.getElementById(`panel-${panelId}`);
        if (panelElement) {
            panelElement.remove();
        }
        
        // Update layout
        this.updateGridLayout();
        
        // Show placeholder if no panels left
        if (this.animationPanels.length === 0 && this.gridPlaceholder) {
            this.gridPlaceholder.style.display = 'flex';
        }
    }
    
    togglePanelFullscreen(panelId) {
        const panel = this.animationPanels.find(p => p.id === panelId);
        if (!panel) return;
        
        // Create fullscreen modal if it doesn't exist
        let modal = document.getElementById('panel-fullscreen-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'panel-fullscreen-modal';
            modal.className = 'panel-fullscreen-modal';
            modal.innerHTML = `
                <div class="fullscreen-header">
                    <h3 class="fullscreen-title" id="fullscreen-title">Animation Fullscreen</h3>
                    <button class="fullscreen-close-btn" onclick="chatInterface.closeFullscreen()">
                        <i class="fas fa-times"></i> Close
                    </button>
                </div>
                <div class="fullscreen-viewer">
                    <canvas id="fullscreen-canvas" class="fullscreen-canvas"></canvas>
                </div>
                <div class="fullscreen-controls">
                    <div class="control-buttons">
                        <button id="fullscreen-play-btn" onclick="chatInterface.toggleFullscreenPlay()">
                            <i class="fas fa-play"></i>
                        </button>
                        <button onclick="chatInterface.stopFullscreen()">
                            <i class="fas fa-stop"></i>
                        </button>
                        <button onclick="chatInterface.prevFullscreenFrame()">
                            <i class="fas fa-step-backward"></i>
                        </button>
                        <button onclick="chatInterface.nextFullscreenFrame()">
                            <i class="fas fa-step-forward"></i>
                        </button>
                    </div>
                    <div class="progress-container">
                        <input type="range" id="fullscreen-slider" min="0" max="100" value="0" 
                               oninput="chatInterface.seekFullscreen(this.value)" />
                        <span id="fullscreen-info">Frame 0 / 0</span>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }
        
        // Store reference to current panel in fullscreen
        this.fullscreenPanel = panel;
        
        // Update title
        const title = document.getElementById('fullscreen-title');
        if (title) {
            const panelTitle = panel.intentType === 'MODIFY_EXISTING' ? 
                `Modified Animation #${panel.id}` : 
                `Animation #${panel.id}`;
            title.textContent = panelTitle;
        }
        
        // Show modal
        modal.classList.add('active');
        
        // Display current frame in fullscreen
        this.displayFullscreenFrame(panel.currentFrame);
        this.updateFullscreenInfo();
        
        // Sync play state
        const playBtn = document.getElementById('fullscreen-play-btn');
        if (playBtn) {
            playBtn.innerHTML = panel.isPlaying ? 
                '<i class="fas fa-pause"></i>' : 
                '<i class="fas fa-play"></i>';
        }
        
        console.log(`🖥️ Fullscreen activated for panel ${panelId}`);
    }
    
    closeFullscreen() {
        const modal = document.getElementById('panel-fullscreen-modal');
        if (modal) {
            modal.classList.remove('active');
        }
        this.fullscreenPanel = null;
    }
    
    displayFullscreenFrame(frameIndex) {
        if (!this.fullscreenPanel || !this.fullscreenPanel.frames[frameIndex]) return;
        
        const canvas = document.getElementById('fullscreen-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const frame = this.fullscreenPanel.frames[frameIndex];
        const img = new Image();
        
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        
        img.src = frame.url;
        this.fullscreenPanel.currentFrame = frameIndex;
        this.updateFullscreenInfo();
    }
    
    updateFullscreenInfo() {
        if (!this.fullscreenPanel) return;
        
        const infoSpan = document.getElementById('fullscreen-info');
        const slider = document.getElementById('fullscreen-slider');
        
        if (infoSpan) {
            infoSpan.textContent = `Frame ${this.fullscreenPanel.currentFrame + 1} / ${this.fullscreenPanel.frames.length}`;
        }
        if (slider) {
            slider.max = this.fullscreenPanel.frames.length - 1;
            slider.value = this.fullscreenPanel.currentFrame;
        }
    }
    
    toggleFullscreenPlay() {
        if (!this.fullscreenPanel) return;
        
        this.fullscreenPanel.isPlaying = !this.fullscreenPanel.isPlaying;
        const playBtn = document.getElementById('fullscreen-play-btn');
        
        if (this.fullscreenPanel.isPlaying) {
            if (playBtn) playBtn.innerHTML = '<i class="fas fa-pause"></i>';
            this.playFullscreenAnimation();
        } else {
            if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
            if (this.fullscreenPanel.playInterval) {
                clearInterval(this.fullscreenPanel.playInterval);
                this.fullscreenPanel.playInterval = null;
            }
        }
        
        // Also update the panel's play button
        const panelPlayBtn = document.getElementById(`panel-play-${this.fullscreenPanel.id}`);
        if (panelPlayBtn) {
            panelPlayBtn.innerHTML = this.fullscreenPanel.isPlaying ? 
                '<i class="fas fa-pause"></i>' : 
                '<i class="fas fa-play"></i>';
        }
    }
    
    playFullscreenAnimation() {
        if (!this.fullscreenPanel) return;
        
        if (this.fullscreenPanel.playInterval) {
            clearInterval(this.fullscreenPanel.playInterval);
        }
        
        this.fullscreenPanel.playInterval = setInterval(() => {
            if (!this.fullscreenPanel) {
                clearInterval(this.fullscreenPanel.playInterval);
                return;
            }
            
            this.fullscreenPanel.currentFrame = (this.fullscreenPanel.currentFrame + 1) % this.fullscreenPanel.frames.length;
            this.displayFullscreenFrame(this.fullscreenPanel.currentFrame);
            
            // Also update the panel display
            this.displayPanelFrame(this.fullscreenPanel, this.fullscreenPanel.currentFrame);
        }, this.fullscreenPanel.playSpeed);
    }
    
    stopFullscreen() {
        if (!this.fullscreenPanel) return;
        
        this.fullscreenPanel.isPlaying = false;
        if (this.fullscreenPanel.playInterval) {
            clearInterval(this.fullscreenPanel.playInterval);
            this.fullscreenPanel.playInterval = null;
        }
        
        this.fullscreenPanel.currentFrame = 0;
        this.displayFullscreenFrame(0);
        this.displayPanelFrame(this.fullscreenPanel, 0);
        
        const playBtn = document.getElementById('fullscreen-play-btn');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
        
        const panelPlayBtn = document.getElementById(`panel-play-${this.fullscreenPanel.id}`);
        if (panelPlayBtn) panelPlayBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
    
    prevFullscreenFrame() {
        if (!this.fullscreenPanel) return;
        
        this.fullscreenPanel.currentFrame = (this.fullscreenPanel.currentFrame - 1 + this.fullscreenPanel.frames.length) % this.fullscreenPanel.frames.length;
        this.displayFullscreenFrame(this.fullscreenPanel.currentFrame);
        this.displayPanelFrame(this.fullscreenPanel, this.fullscreenPanel.currentFrame);
    }
    
    nextFullscreenFrame() {
        if (!this.fullscreenPanel) return;
        
        this.fullscreenPanel.currentFrame = (this.fullscreenPanel.currentFrame + 1) % this.fullscreenPanel.frames.length;
        this.displayFullscreenFrame(this.fullscreenPanel.currentFrame);
        this.displayPanelFrame(this.fullscreenPanel, this.fullscreenPanel.currentFrame);
    }
    
    seekFullscreen(value) {
        if (!this.fullscreenPanel) return;
        
        const frameIndex = parseInt(value);
        this.displayFullscreenFrame(frameIndex);
        this.displayPanelFrame(this.fullscreenPanel, frameIndex);
    }

    updateStatus(message, type) {
        if (this.animationStatus) {
            const statusElement = this.animationStatus.querySelector('span:last-child');
            const indicator = this.animationStatus.querySelector('.status-indicator');

            if (statusElement) statusElement.textContent = message;
            if (indicator) indicator.className = `status-indicator ${type}`;
        }
    }

    showSystemLogs() {
        const systemLogs = document.getElementById('systemLogs');
        if (systemLogs) {
            systemLogs.style.display = 'block';
        }
    }

    // Animation player control methods
    playAnimation() {
        if (!this.framePlayer || this.framePlayer.frames.length === 0) return;

        this.framePlayer.isPlaying = true;
        console.log('Starting animation playback');

        const playNextFrame = () => {
            if (!this.framePlayer.isPlaying) return;

            this.framePlayer.currentFrame = (this.framePlayer.currentFrame + 1) % this.framePlayer.frames.length;
            this.displayFrame(this.framePlayer.currentFrame);

            this.framePlayer.playInterval = setTimeout(playNextFrame, this.framePlayer.playSpeed);
        };

        this.framePlayer.playInterval = setTimeout(playNextFrame, this.framePlayer.playSpeed);
        this.updatePlayButton(true);
    }

    pauseAnimation() {
        if (!this.framePlayer) return;

        this.framePlayer.isPlaying = false;
        if (this.framePlayer.playInterval) {
            clearTimeout(this.framePlayer.playInterval);
            this.framePlayer.playInterval = null;
        }
        console.log('Animation paused');
        this.updatePlayButton(false);
    }

    stopAnimation() {
        if (!this.framePlayer) return;

        this.pauseAnimation();
        this.framePlayer.currentFrame = 0;
        this.displayFrame(0);
        console.log('Animation stopped');
    }

    nextFrame() {
        if (!this.framePlayer || this.framePlayer.frames.length === 0) return;

        this.pauseAnimation();
        this.framePlayer.currentFrame = Math.min(this.framePlayer.currentFrame + 1, this.framePlayer.frames.length - 1);
        this.displayFrame(this.framePlayer.currentFrame);
    }

    previousFrame() {
        if (!this.framePlayer || this.framePlayer.frames.length === 0) return;

        this.pauseAnimation();
        this.framePlayer.currentFrame = Math.max(this.framePlayer.currentFrame - 1, 0);
        this.displayFrame(this.framePlayer.currentFrame);
    }

    updatePlayButton(isPlaying) {
        // Update play/pause button if it exists
        const playButton = document.getElementById('playPauseBtn');
        if (playButton) {
            const icon = playButton.querySelector('i');
            if (icon) {
                icon.className = isPlaying ? 'fas fa-pause' : 'fas fa-play';
            }
        }
    }

    showPendingEvaluation() {
        if (this.pendingEvaluation) {
            console.log('🎯 Showing pending evaluation after frames loaded');
            console.log('Pending evaluation object:', this.pendingEvaluation);
            console.log('Continue prompt:', this.pendingEvaluation.continue_prompt);

            // Clean evaluation text from markdown artifacts before displaying
            const cleanedEval = cleanMarkdownForChat(this.pendingEvaluation.evaluation || '');

            // Show evaluation with typing effect and callback for continue prompt
            this.addTypingMessage(`Evaluation:\n${cleanedEval}`, 'bot', 30, () => {
                // This callback runs when evaluation typing is complete
                console.log('✅ Evaluation typing completed, now showing continue prompt');

                if (this.pendingEvaluation && this.pendingEvaluation.continue_prompt) {
                    this.addTypingMessage(this.pendingEvaluation.continue_prompt, 'bot');
                    console.log('✅ Continue prompt started typing');
                } else {
                    console.log('⚠️ No continue prompt found, showing fallback');
                    // Fallback: show a default continue prompt
                    this.addTypingMessage('Would you like me to generate modified animation (y), or would you like to provide specific guidance (g), or would you like to finish (n)? You can also type \'quit\' to exit.', 'bot');
                    console.log('✅ Fallback continue prompt started typing');
                }
            });

            // Clear pending evaluation
            this.pendingEvaluation = null;
        }

        // If there is a deferred evaluation prompt (for new-generation animations),
        // show it now that frames are available for preview.
        if (this.pendingEvaluationPrompt) {
            console.log('Showing deferred evaluation prompt now that frames are available');
            const promptHtml = `
                <div class="evaluation-prompt">
                    <p>${this.pendingEvaluationPrompt}</p>
                    <button class="option-btn" id="evalYesBtn">Yes - Evaluate</button>
                    <button class="option-btn" id="evalNoBtn">No - Skip</button>
                </div>
            `;
            this.addCustomContent(promptHtml);

            // Attach handlers
            setTimeout(() => {
                const yes = document.getElementById('evalYesBtn');
                const no = document.getElementById('evalNoBtn');
                if (yes) {
                    yes.addEventListener('click', () => {
                        addSystemLog('User requested on-demand evaluation', 'info');
                        yes.disabled = true;
                        no.disabled = true;
                        evaluateLastAnimation();
                    });
                }
                if (no) {
                    no.addEventListener('click', () => {
                        addSystemLog('User skipped evaluation', 'info');
                        no.disabled = true;
                        yes.disabled = true;
                        chatInterface.addMessage('Okay — evaluation skipped. You can request it later.', 'bot');
                    });
                }
            }, 50);

            // Clear the deferred prompt so we don't show it again
            this.pendingEvaluationPrompt = null;
        } else {
            console.log('⚠️ No pending evaluation to show');
        }
    }
}

// Animation control functions
function togglePlayPause() {
    console.log('Toggle play/pause');
    if (chatInterface && chatInterface.framePlayer) {
        if (chatInterface.framePlayer.isPlaying) {
            chatInterface.pauseAnimation();
        } else {
            chatInterface.playAnimation();
        }
    }
}

function stopAnimation() {
    console.log('Stop animation');
    if (chatInterface && chatInterface.framePlayer) {
        chatInterface.stopAnimation();
    }
}

function previousFrame() {
    console.log('Previous frame');
    if (chatInterface && chatInterface.framePlayer) {
        chatInterface.previousFrame();
    }
}

function nextFrame() {
    console.log('Next frame');
    if (chatInterface && chatInterface.framePlayer) {
        chatInterface.nextFrame();
    }
}

function exportAnimation() {
    console.log('Export animation');
    // TODO: Implement export functionality - could generate GIF from frames
}

// Trigger on-demand evaluation for the last generated animation
function evaluateLastAnimation() {
    addSystemLog('Requesting on-demand evaluation from server...', 'info');
    fetch('/api/evaluate_last_animation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(resp => resp.json())
        .then(data => {
            if (!data) {
                addSystemLog('No response from evaluation endpoint', 'warning');
                return;
            }
            if (data.status === 'success' && data.evaluation) {
                // Clean markdown-like artifacts before displaying to avoid literal ### and ```
                const cleaned = cleanMarkdownForChat(data.evaluation || '');
                chatInterface.addTypingMessage(`Evaluation:\n${cleaned}`, 'bot', 30);
                addSystemLog('Evaluation received and displayed', 'info');
            } else {
                addSystemLog(`Evaluation failed or unavailable: ${data.message || JSON.stringify(data)}`, 'warning');
                chatInterface.addMessage('Evaluation is temporarily unavailable.', 'bot');
            }
        })
        .catch(err => {
            console.error('Error requesting evaluation:', err);
            addSystemLog(`Error requesting evaluation: ${err}`, 'error');
            chatInterface.addMessage('There was an error requesting evaluation.', 'bot');
        });
}

// Global functions called from HTML
function selectDataset(type) {
    console.log('selectDataset called with type:', type);
    if (type === 'available') {
        // Start the conversation when user selects available dataset
        console.log('Starting conversation...');
        chatInterface.startConversation();
    } else if (type === 'upload') {
        chatInterface.addMessage('Dataset upload feature is not yet implemented. Please use the available dataset.', 'bot');
    }
}

function selectPhenomenon(id) {
    if (chatInterface) {
        if (id === 'custom') {
            chatInterface.selectPhenomenon('0', 'Custom Description');
        } else {
            // Map the phenomenon IDs to names
            const phenomenaNames = {
                '1': 'Agulhas Ring Current - Ocean Temperature',
                '2': 'Agulhas Ring Current - Ocean Salinity with Streamlines',
                '3': 'Mediterranean Sea Current - Ocean Temperature',
                '4': 'Mediterranean Sea Current - Ocean Salinity with Streamlines'
            };
            const name = phenomenaNames[id] || 'Unknown Phenomenon';
            chatInterface.selectPhenomenon(id, name);
        }
    }
}

function submitCustomDescription() {
    const description = document.getElementById('customDescription').value.trim();
    if (description && chatInterface) {
    chatInterface.addMessage(description, 'user');
    // Send the custom description as the phenomenon message. Do not send
    // the legacy numeric 'choice' field — backend expects free-text.
    chatInterface.sendApiMessage(description, 'select_phenomenon');
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
                <p>System logs will appear here during animation generation</p>
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

function toggleLogsExpand() {
    // Legacy function - redirect to new function
    toggleLogsSize();
}

// Helper: remove common Markdown artifacts before showing LLM evaluation in chat
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

    logsContent.appendChild(logEntry);
    logsContent.scrollTop = logsContent.scrollHeight;

    // Add activity indicator if logs are minimized
    const systemLogs = document.querySelector('.system-logs');
    if (systemLogs && systemLogs.classList.contains('minimized')) {
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