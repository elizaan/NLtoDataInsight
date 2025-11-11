// This file contains JavaScript code for client-side interactivity, handling user inputs, and making API calls to the backend.

document.addEventListener('DOMContentLoaded', function() {
    // Import feature modules
    import ('/static/js/features/datasetList.js').then(({ fetchDatasets, renderDatasetList }) => {
        import ('/static/js/features/datasetSummary.js').then(({ fetchDatasetSummary, renderDatasetSummary }) => {
            import ('/static/js/features/phenomenonSelection.js').then(({ renderPhenomenonOptions }) => {
                // UI element IDs
                const datasetSelectionId = 'datasetSelection';
                const datasetListContainerId = 'available-datasets-list';
                const datasetSummaryContainerId = 'datasetSummaryContainer';
                const phenomenonSelectionId = 'phenomenonSelection';
                // Use shared mdRenderer when possible to ensure consistent
                // Markdown -> sanitized HTML behavior across the app.
                async function getMdToHtml() {
                    if (window.sharedMdToHtml) return window.sharedMdToHtml;
                    try {
                        const mod = await
                        import ('/static/js/lib/mdRenderer.js');
                        if (mod && mod.mdToHtml) {
                            window.sharedMdToHtml = mod.mdToHtml;
                            return window.sharedMdToHtml;
                        }
                    } catch (e) {
                        console.warn('Could not load shared mdRenderer, falling back.', e);
                    }
                    // Fallback: small inline safe renderer (keeps compatibility)
                    return function fallbackMdToHtml(text) {
                        if (!text) return '';
                        let s = String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        s = s.replace(/(^|\n)\s{0,3}(?:#|&#35;){1,6}\s*/g, '$1');
                        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                        s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
                        const lines = s.split(/\r?\n/);
                        let inList = false;
                        const out = [];
                        lines.forEach(line => {
                            const m = line.match(/^\s*-\s+(.*)$/);
                            if (m) {
                                if (!inList) { out.push('<ul>');
                                    inList = true; }
                                out.push(`<li>${m[1]}</li>`);
                            } else {
                                if (inList) { out.push('</ul>');
                                    inList = false; }
                                if (/^\s*$/.test(line)) return;
                                out.push(`<p>${line}</p>`);
                            }
                        });
                        if (inList) out.push('</ul>');
                        return out.join('\n');
                    };
                }
                // Step 1: Show dataset list when 'Use Available Dataset' is clicked
                window.selectDataset = function(type) {
                    if (type === 'available') {
                        document.getElementById(datasetSelectionId).style.display = 'block';
                        fetchDatasets().then(result => {
                            renderDatasetList(result.datasets, datasetListContainerId, function(dataset) {
                                // Hide dataset list after selection
                                document.getElementById(datasetSelectionId).style.display = 'none';
                                onDatasetSelect(dataset);
                            });
                        });
                    }
                };
                // Step 2: Show dataset summary after selection
                function onDatasetSelect(dataset) {
                    // Show agent message: summarizing dataset details with loading dots...
                    const chatMessages = document.getElementById('chatMessages');
                    const agentMsg = document.createElement('div');
                    agentMsg.className = 'message bot-message';
                    agentMsg.innerHTML = `
                                                        <div class='message-avatar'><i class='fas fa-robot'></i></div>
                                                        <div class='message-content'>
                                                            <p id='loading-summary-text'>Summarizing dataset details<span id='loading-dots'></span></p>
                                                        </div>`;
                    chatMessages.appendChild(agentMsg);

                    // Add animated dots to simulate loading
                    let dotCount = 0;
                    const loadingDots = document.getElementById('loading-dots');
                    const loadingInterval = setInterval(() => {
                        dotCount = (dotCount + 1) % 4;
                        loadingDots.textContent = '.'.repeat(dotCount);
                    }, 500);

                    fetchDatasetSummary(dataset.id).then(async result => {
                        const md = await getMdToHtml();
                        // Use LLM summary if available, fallback to heuristic
                        let summaryText = '';
                        if (result.summary && result.summary.llm) {
                            const llm = result.summary.llm;
                            summaryText = `<h4>${md(llm.title || 'Dataset Summary')}</h4>`;
                            if (llm.summary) summaryText += md(llm.summary);
                            if (llm.visualizations && Array.isArray(llm.visualizations) && llm.visualizations.length) {
                                summaryText += '<h5>Suggested Visualizations</h5><ul>' +
                                    llm.visualizations.map(v => `<li>${md(v)}</li>`).join('') +
                                    '</ul>';
                            }
                            if (llm.recommended_camera) {
                                summaryText += `<p><strong>Recommended Camera:</strong> ${md(llm.recommended_camera)}</p>`;
                            }
                            if (llm.tf_notes) {
                                summaryText += `<p><strong>Transfer Function Notes:</strong> ${md(llm.tf_notes)}</p>`;
                            }
                        } else if (result.summary && result.summary.heuristic) {
                            summaryText = `<h4>Dataset Summary</h4><pre>${JSON.stringify(result.summary.heuristic, null, 2)}</pre>`;
                        } else if (result.summary && result.summary.llm_text) {
                            // Show raw LLM text if JSON parsing failed but text exists. Render minimal markdown.
                            summaryText = `<h4>Dataset Summary</h4>${md(result.summary.llm_text)}`;
                        } else {
                            summaryText = '<h4>Dataset Summary</h4><p>No summary available.</p>';
                        }
                        // Remove loading message and spinner
                        clearInterval(loadingInterval);
                        if (agentMsg && agentMsg.parentNode) {
                            agentMsg.parentNode.removeChild(agentMsg);
                        }

                        const summaryMsg = document.createElement('div');
                        summaryMsg.className = 'message bot-message';
                        summaryMsg.innerHTML = `<div class='message-avatar'><i class='fas fa-robot'></i></div><div class='message-content'>${summaryText}</div>`;
                        chatMessages.appendChild(summaryMsg);
                        // (Removed automatic question prompt per user request.)

                        // Tell the chat controller that we are expecting a custom description
                        // so the next user-typed message is treated like choice '0'.
                        if (window.chatInterface && typeof window.chatInterface === 'object') {
                            try { window.chatInterface.conversationState = 'custom_description'; } catch (e) { console.warn('Unable to set chatInterface state', e); }
                        } else {
                            // If ChatInterface is not initialized yet, set a pending flag it can read on init
                            window.pendingConversationState = 'custom_description';
                        }

                        // Step 3: Show chat input area for user to describe what they want
                        const chatInputContainer = document.getElementById('chatInputContainer');
                        if (chatInputContainer) chatInputContainer.style.display = 'block';
                        // If there's an input element, set a helpful placeholder to guide the user
                        const chatInput = document.getElementById('chatInput');
                        if (chatInput) chatInput.placeholder = `Describe the ${dtype} phenomenon you'd like to animate (e.g. temperature front, eddy, salinity intrusion)...`;
                    });
                }
                // Step 4: Handle phenomenon selection (call backend, show animation, etc.)
            });
        });
    });
});