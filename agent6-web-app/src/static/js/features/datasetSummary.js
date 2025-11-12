// Feature: Fetch and display dataset summary
export async function fetchDatasetSummary(datasetId) {
    // The backend /api/chat expects 'dataset_id' for the select_dataset action

    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'select_dataset', dataset_id: datasetId, use_langchain: use_langchain })
    });
    if (!response.ok) throw new Error('Failed to fetch dataset summary');
    return await response.json();
}

import { mdToHtml } from '/static/js/lib/mdRenderer.js';

export function renderDatasetSummary(summary, containerId) {
    const container = document.getElementById(containerId);

    const name = summary.name ? `<h4>${mdToHtml(summary.name)}</h4>` : '';
    const desc = summary.description ? `<div class='dataset-description'>${mdToHtml(summary.description)}</div>` : '';
    const fields = Array.isArray(summary.fields) ? `<ul>${summary.fields.map(f => `<li>${mdToHtml(f)}</li>`).join('')}</ul>` : '';
    const dims = summary.dimensions ? `<p>Dimensions: ${mdToHtml(String(summary.dimensions))}</p>` : '';
    const tr = summary.time_range ? `<p>Time Range: ${mdToHtml(String(summary.time_range))}</p>` : '';
    const vis = Array.isArray(summary.visualization_options) ? `<p>Visualization Options: ${summary.visualization_options.map(v => mdToHtml(String(v))).join(', ')}</p>` : '';

    container.innerHTML = `<div class='dataset-summary'>${name}${desc}${fields}${dims}${tr}${vis}</div>`;
}