// Feature: Phenomenon selection UI and callback
export function renderPhenomenonOptions(options, containerId, onSelect) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    options.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'phenomenon-btn';
        btn.innerHTML = `<i class='${opt.icon}'></i> <span>${opt.label}</span>`;
        btn.onclick = () => onSelect(opt);
        container.appendChild(btn);
    });
}