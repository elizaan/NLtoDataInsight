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
            <div class='dataset-card' data-id='${ds.id || ds.name}'>
                <i class="fas fa-water dataset-icon"></i>
                <div class='dataset-meta'>
                    <h4 class='dataset-name'>${ds.name || 'Unknown Dataset'}</h4>
                    <p class='dataset-desc'>${ds.description || ds.type || 'High Resolution Ocean Simulation'}</p>
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
                    // Visual feedback: use CSS .selected class so theme variables apply
                    botMsg.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
                    card.classList.add('selected');
                    onSelect(ds);
                };
            }
        });
        
        // Hide the bottom panel container
        document.getElementById(containerId).style.display = 'none';
    }
}