const API = window.location.origin;

const state = {
    mode: "chat",
    conversationHistory: [],
    fundFilter: null,
    documents: [],
    currentSources: [],
    suggestions: {},
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ───── Initialization ─────

document.addEventListener("DOMContentLoaded", () => {
    initSidebar();
    initModeSelector();
    initChatForm();
    loadDocuments();
    loadSuggestions();
    checkHealth();
});

// ───── Sidebar ─────

function initSidebar() {
    $("#sidebarToggle").addEventListener("click", () => {
        $("#sidebar").classList.toggle("collapsed");
    });
    $("#sourcePanelClose").addEventListener("click", () => {
        $("#sourcePanel").classList.remove("open");
    });
    $("#docSearch").addEventListener("input", (e) => {
        filterDocuments(e.target.value);
    });
    $("#filterClear").addEventListener("click", () => {
        state.fundFilter = null;
        $("#fundFilterBar").style.display = "none";
        $$(".doc-item").forEach((el) => el.classList.remove("active"));
    });
}

// ───── Mode Selector ─────

function initModeSelector() {
    $$(".mode-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".mode-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            state.mode = btn.dataset.mode;
            updateModeHint();
            renderSuggestions(state.mode);
            $("#welcomeScreen").style.display = "";
            $("#messages").classList.remove("active");
        });
    });
}

function updateModeHint() {
    const hints = {
        chat: "Chat mode — ask questions about your documents",
        compare: "Compare mode — compare funds or documents side by side",
        calculate: "Calculate mode — extract figures and perform calculations",
    };
    $("#modeHint").textContent = hints[state.mode] || "Chat mode";
}

// ───── Chat Form ─────

function initChatForm() {
    const textarea = $("#queryInput");
    const form = $("#chatForm");

    textarea.addEventListener("input", () => {
        textarea.style.height = "auto";
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
        $("#sendBtn").disabled = !textarea.value.trim();
    });

    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (textarea.value.trim()) form.dispatchEvent(new Event("submit"));
        }
    });

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = textarea.value.trim();
        if (!query) return;
        sendMessage(query);
        textarea.value = "";
        textarea.style.height = "auto";
        $("#sendBtn").disabled = true;
    });
}

// ───── Messaging ─────

async function sendMessage(query) {
    $("#welcomeScreen").style.display = "none";
    const messagesEl = $("#messages");
    messagesEl.classList.add("active");

    appendMessage("user", query);

    state.conversationHistory.push({ role: "user", content: query });

    const loadingEl = appendLoading();

    try {
        const res = await fetch(`${API}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query,
                conversation_history: state.conversationHistory.slice(-6),
                fund_filter: state.fundFilter,
                top_k: state.mode === "compare" ? 12 : 8,
                mode: state.mode,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Request failed");
        }

        const data = await res.json();
        loadingEl.remove();
        appendMessage("assistant", data.answer, data.sources);

        state.conversationHistory.push({ role: "assistant", content: data.answer });
    } catch (err) {
        loadingEl.remove();
        appendMessage("assistant", `**Error:** ${err.message}. Please check that the backend is running and your API keys are configured.`);
    }

    scrollToBottom();
}

function appendMessage(role, content, sources = null) {
    const messagesEl = $("#messages");
    const div = document.createElement("div");
    div.className = "message";

    const avatarIcon = role === "user"
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>';

    let sourcesBtn = "";
    if (sources && sources.length > 0) {
        sourcesBtn = `
            <button class="message-sources-btn" onclick='showSources(${JSON.stringify(sources).replace(/'/g, "&#39;")})'>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                ${sources.length} source${sources.length !== 1 ? "s" : ""}
            </button>`;
    }

    div.innerHTML = `
        <div class="message-avatar ${role}">${avatarIcon}</div>
        <div class="message-body">
            <div class="message-sender">${role === "user" ? "You" : "VDR Copilot"}</div>
            <div class="message-content">${renderMarkdown(content)}</div>
            ${sourcesBtn}
        </div>`;

    messagesEl.appendChild(div);
    scrollToBottom();
}

function appendLoading() {
    const messagesEl = $("#messages");
    const div = document.createElement("div");
    div.className = "message loading-message";
    div.innerHTML = `
        <div class="message-avatar assistant">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        </div>
        <div class="message-body">
            <div class="message-sender">VDR Copilot</div>
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>`;
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    const container = $("#chatContainer");
    container.scrollTop = container.scrollHeight;
}

// ───── Sources Panel ─────

function showSources(sources) {
    state.currentSources = sources;
    const panel = $("#sourcePanel");
    const body = $("#sourcePanelBody");

    body.innerHTML = sources
        .map(
            (s, i) => `
        <div class="source-card">
            <div class="source-card-header">
                <span class="source-card-name">${escapeHtml(s.fund_name)}</span>
                <span class="source-card-score">${(s.score * 100).toFixed(1)}% match</span>
            </div>
            <div class="source-card-meta">
                ${getBadgeHtml(s.category)} · Page ${s.page_number} · ${escapeHtml(s.source_file)}
            </div>
            <div class="source-card-excerpt">${escapeHtml(s.text)}</div>
        </div>`
        )
        .join("");

    panel.classList.add("open");
}

// ───── Documents ─────

async function loadDocuments() {
    try {
        const res = await fetch(`${API}/api/documents`);
        const data = await res.json();
        state.documents = data.documents || [];
        renderDocuments(state.documents);
    } catch {
        $("#docList").innerHTML = '<div class="loading-docs">Could not load documents</div>';
    }
}

function renderDocuments(docs) {
    const list = $("#docList");
    if (!docs.length) {
        list.innerHTML = '<div class="loading-docs">No documents found</div>';
        return;
    }

    list.innerHTML = docs
        .map(
            (doc) => `
        <div class="doc-item" data-fund="${escapeHtml(doc.fund_name)}" onclick="selectDocument(this, '${escapeHtml(doc.fund_name).replace(/'/g, "\\'")}')">
            <div class="doc-item-name">${escapeHtml(doc.fund_name)}</div>
            <div class="doc-item-meta">${getBadgeHtml(doc.category)} · ${escapeHtml(doc.folder)}</div>
        </div>`
        )
        .join("");
}

function filterDocuments(query) {
    const q = query.toLowerCase();
    const filtered = state.documents.filter(
        (d) => d.fund_name.toLowerCase().includes(q) || d.category.toLowerCase().includes(q) || d.folder.toLowerCase().includes(q)
    );
    renderDocuments(filtered);
}

function selectDocument(el, fundName) {
    $$(".doc-item").forEach((d) => d.classList.remove("active"));
    if (state.fundFilter === fundName) {
        state.fundFilter = null;
        $("#fundFilterBar").style.display = "none";
    } else {
        el.classList.add("active");
        state.fundFilter = fundName;
        $("#filterValue").textContent = fundName;
        $("#fundFilterBar").style.display = "flex";
    }
}

// ───── Suggestions ─────

async function loadSuggestions() {
    try {
        const res = await fetch(`${API}/api/suggest`);
        state.suggestions = await res.json();
        renderSuggestions(state.mode);
    } catch {
        /* suggestions are non-critical */
    }
}

function renderSuggestions(mode) {
    const grid = $("#suggestionGrid");
    if (!grid) return;

    const items = state.suggestions[mode] || [];

    const icons = {
        shield: "🛡️",
        scale: "⚖️",
        calculator: "🔢",
        target: "🎯",
        leaf: "🌿",
        "pie-chart": "📊",
        search: "🔍",
        clock: "🕐",
        globe: "🌐",
        coins: "💰",
        "trending-up": "📈",
    };

    const bgColors = {
        chat: "var(--accent-soft)",
        compare: "var(--purple-soft)",
        calculate: "var(--green-soft)",
    };

    grid.innerHTML = items
        .map(
            (s) => `
        <div class="suggestion-card" onclick="useSuggestion('${escapeHtml(s.text).replace(/'/g, "\\'")}', '${mode}')">
            <div class="suggestion-card-icon" style="background:${bgColors[mode] || "var(--accent-soft)"}">${icons[s.icon] || "💬"}</div>
            <div class="suggestion-card-text">${escapeHtml(s.text)}</div>
        </div>`
        )
        .join("");
}

function useSuggestion(text, mode) {
    state.mode = mode;
    $$(".mode-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.mode === mode);
    });
    updateModeHint();
    $("#queryInput").value = text;
    $("#queryInput").dispatchEvent(new Event("input"));
    $("#chatForm").dispatchEvent(new Event("submit"));
}

// ───── Health Check ─────

async function checkHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();
        const el = $("#indexHealth");
        if (data.status === "healthy") {
            const count = data.total_vectors || 0;
            el.innerHTML = `<div class="health-dot healthy"></div><span>${count.toLocaleString()} vectors indexed</span>`;
        } else {
            el.innerHTML = `<div class="health-dot degraded"></div><span>Index unavailable</span>`;
        }
    } catch {
        const el = $("#indexHealth");
        el.innerHTML = `<div class="health-dot degraded"></div><span>Backend offline</span>`;
    }
}

// ───── Markdown Rendering ─────

function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    // Tables
    html = html.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)*)/gm, (match, header, sep, body) => {
        const headers = header.split("|").filter(Boolean).map((h) => `<th>${h.trim()}</th>`).join("");
        const rows = body.trim().split("\n").map((row) => {
            const cells = row.split("|").filter(Boolean).map((c) => `<td>${c.trim()}</td>`).join("");
            return `<tr>${cells}</tr>`;
        }).join("");
        return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    });

    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/`(.+?)`/g, "<code>$1</code>");
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^# (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

    html = html.split("\n\n").map((p) => {
        p = p.trim();
        if (!p) return "";
        if (p.startsWith("<h") || p.startsWith("<table") || p.startsWith("<ul") || p.startsWith("<ol") || p.startsWith("<li")) return p;
        return `<p>${p}</p>`;
    }).join("");

    html = html.replace(/\n/g, "<br>");

    return html;
}

// ───── Utilities ─────

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function getBadgeHtml(category) {
    const cls = category.toLowerCase().includes("prospectus")
        ? "prospectus"
        : category.toLowerCase().includes("kiid") || category.toLowerCase().includes("kid")
            ? "kiid"
            : category.toLowerCase().includes("factsheet")
                ? "factsheet"
                : category.toLowerCase().includes("report")
                    ? "report"
                    : "default";
    return `<span class="doc-badge ${cls}">${escapeHtml(category)}</span>`;
}
