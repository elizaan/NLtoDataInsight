// Feature: Fetch and display available datasets
export async function fetchDatasets() {
    const response = await fetch('/api/datasets');
    if (!response.ok) throw new Error('Failed to fetch datasets');
    return await response.json();
}

export function renderDatasetList(datasets, containerId, onSelect) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    // Conversational UI: show only the dataset name as a single card
    if (datasets.length > 0) {
        const ds = datasets[0];
        // Insert as a bot message in chatMessages
        const chatMessages = document.getElementById('chatMessages');
        const botMsg = document.createElement('div');
        botMsg.className = 'message bot-message';
        botMsg.innerHTML = `<div class='message-avatar'><i class='fas fa-robot'></i></div><div class='message-content'><p>Please choose one dataset from below list (for now just one dataset):</p><div class='dataset-card' style='margin-top:10px;cursor:pointer;'><i class=\"fas fa-water\"></i><div><h4>${ds.name}</h4><p>High Resolution Ocean Simulation</p></div></div></div>`;
        // Add click handler to card
        botMsg.querySelector('.dataset-card').onclick = () => onSelect(ds);
        chatMessages.appendChild(botMsg);
        // Hide the bottom panel container
        document.getElementById(containerId).style.display = 'none';
    }
}