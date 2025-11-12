// Feature: Fetch and display available datasets
export async function fetchDatasets() {
    const response = await fetch('/api/datasets');
    if (!response.ok) throw new Error('Failed to fetch datasets');
    return await response.json();
}

export function renderDatasetList(datasets, containerId, onSelect) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (datasets.length > 0) {
        // Insert as a bot message in chatMessages
        const chatMessages = document.getElementById('chatMessages');
        const botMsg = document.createElement('div');
        botMsg.className = 'message bot-message';
        
        // Build HTML for all datasets
        let datasetCardsHTML = datasets.map(ds => `
            <div class='dataset-card' data-id='${ds.id || ds.name}' style='margin-top:10px;cursor:pointer;padding:12px;border:1px solid #444;border-radius:8px;background:#2a2a2a;transition:background 0.2s;'>
                <i class="fas fa-water" style="font-size:24px;color:#4fc3f7;margin-right:10px;"></i>
                <div>
                    <h4 style="margin:0;color:#fff;">${ds.name || 'Unknown Dataset'}</h4>
                    <p style="margin:4px 0 0 0;color:#aaa;font-size:0.9em;">${ds.description || ds.type || 'High Resolution Ocean Simulation'}</p>
                </div>
            </div>
        `).join('');
        
        botMsg.innerHTML = `
            <div class='message-avatar'><i class='fas fa-robot'></i></div>
            <div class='message-content'>
                <p>Please choose a dataset from the list below:</p>
                ${datasetCardsHTML}
            </div>
        `;
        
        // Add click handlers to all cards
        chatMessages.appendChild(botMsg);
        datasets.forEach((ds, index) => {
            const card = botMsg.querySelectorAll('.dataset-card')[index];
            if (card) {
                card.onclick = () => {
                    // Visual feedback
                    card.style.background = '#1e5a7d';
                    onSelect(ds);
                };
                // Hover effect
                card.onmouseenter = () => { card.style.background = '#3a3a3a'; };
                card.onmouseleave = () => { card.style.background = '#2a2a2a'; };
            }
        });
        
        // Hide the bottom panel container
        document.getElementById(containerId).style.display = 'none';
    }
}