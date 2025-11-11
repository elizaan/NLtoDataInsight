// Shared Markdown -> sanitized HTML renderer used across the UI.
// Exports: mdToHtml(text) -> sanitized HTML string.
export function mdToHtml(text) {
    function sanitizeText(s) {
        if (!s || typeof s !== 'string') return '';
        // Remove horizontal rules (---, ***, ___)
        s = s.replace(/^\s*(?:[-*_]{3,})\s*$/gm, '');
        // Strip leading Markdown heading markers (#) or HTML-escaped '#' (&#35;)
        // at the start of lines so the UI doesn't show raw '###'. Keep the
        // heading text but remove the markers.
        s = s.replace(/(^|\n)\s{0,3}(?:#|&#35;){1,6}\s*/g, '$1');
        // Also remove any leftover run of 2+ hashes that may appear mid-line
        // (e.g. ' ... ### ...') leaving surrounding whitespace intact.
        s = s.replace(/(^|\s)(?:#|&#35;){2,}\s*/g, '$1');
        // Remove any remaining hashes that appear as heading markers
        // (hashes followed by whitespace) anywhere in the text. Avoid removing
        // single-hash uses like 'C#' or '#1' because those do not have a
        // whitespace after the hash.
        s = s.replace(/(^|\s)(?:#|&#35;)+\s+/g, '$1');
        // Collapse excessive blank lines
        s = s.replace(/\n{3,}/g, '\n\n');
        return s.trim();
    }

    if (!text) return '';
    const pre = sanitizeText(String(text));

    // If marked or DOMPurify are not available, provide a safe fallback.
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
        let s = String(pre).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        // Convert heading lines to bold paragraphs to keep structure compact
        s = s.replace(/^\s{0,3}(#{1,6})\s*(.+)$/gm, function(_, hashes, content) {
            return `<p><strong>${content.trim()}</strong></p>`;
        });
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
                if (/^\s*<p>\s*<strong>/.test(line)) { out.push(line); } else { out.push(`<p>${line}</p>`); }
            }
        });
        if (inList) out.push('</ul>');
        return out.join('\n');
    }

    try {
        const raw = marked.parse(String(pre || ''));
        let sanitized = DOMPurify.sanitize(raw, { ADD_ATTR: ['target'] });
        // Convert <h1>-<h6> to bold paragraphs to avoid oversized headings
        sanitized = sanitized.replace(/<h[1-6]>([\s\S]*?)<\/h[1-6]>/gi, function(_, inner) {
            return `<p><strong>${inner.trim()}</strong></p>`;
        });
        // Defensive cleanup of any remaining heading-like hashes
        sanitized = sanitized.replace(/(^|\s)(?:&#35;|#)+\s+/g, '$1');
        sanitized = sanitized.replace(/(^|\s)(?:&#35;|#){2,}\s*/g, '$1');
        return sanitized;
    } catch (e) {
        console.error('mdRenderer.mdToHtml error', e);
        return String(pre).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
}

export default { mdToHtml };