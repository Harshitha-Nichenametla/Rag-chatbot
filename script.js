// ==========================================
// script.js — RAG Chatbot Frontend Logic
// Works together with index.html
// ==========================================

const API_URL        = "http://127.0.0.1:8000/chat";
const UPLOAD_URL     = "http://127.0.0.1:8000/upload-document";
const LIST_URL       = "http://127.0.0.1:8000/list-documents";
const FILES_URL      = "http://127.0.0.1:8000/files";
const HISTORY_URL    = "http://127.0.0.1:8000/history";
const CLEAR_HIST_URL = "http://127.0.0.1:8000/clear-history";

// ==========================================
// State
// ==========================================
let chatHistory   = [];
let isLoading     = false;
let currentChatId = null;
let pendingFile   = null;
let filePanelOpen = false;
let currentTab    = "active";
let filesData     = { active: [], backups: [] };

// ==========================================
// Init
// ==========================================
window.onload = () => {
  loadChatHistorySidebar();

  document.getElementById("doc-upload").addEventListener("change", e => {
    const f = e.target.files[0];
    if (f) uploadDocument(f, false);
    e.target.value = "";
  });

  document.getElementById("panel-upload").addEventListener("change", e => {
    const f = e.target.files[0];
    if (f) uploadDocument(f, false);
    e.target.value = "";
  });
};

// ==========================================
// Helpers
// ==========================================
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function showToast(msg, type = "info", duration = 3500) {
  const c = document.getElementById("toast-container");
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = "0";
    t.style.transition = "opacity 0.3s";
    setTimeout(() => t.remove(), 350);
  }, duration);
}

function getFileIconClass(filename) {
  const ext = filename.split(".").pop().toLowerCase();
  if (ext === "txt")  return "txt";
  if (ext === "docx") return "docx";
  if (ext === "pdf")  return "pdf";
  return "other";
}

function getFileIconLabel(filename) {
  return filename.split(".").pop().toUpperCase().slice(0, 4);
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
  });
}

// ==========================================
// File Panel Toggle
// ==========================================
function toggleFilePanel() {
  filePanelOpen = !filePanelOpen;
  const panel = document.getElementById("file-panel");
  const grid  = document.getElementById("app-grid");
  const tbBtn = document.getElementById("topbar-files-btn");
  const sbBtn = document.getElementById("files-toggle-btn");

  if (filePanelOpen) {
    grid.style.gridTemplateColumns = `var(--sidebar-w) 1fr var(--panel-w)`;
    panel.style.display = "flex";
    tbBtn.classList.add("active");
    sbBtn.classList.add("active-tab");
    loadFiles();
  } else {
    grid.style.gridTemplateColumns = `var(--sidebar-w) 1fr 0`;
    panel.style.display = "none";
    tbBtn.classList.remove("active");
    sbBtn.classList.remove("active-tab");
  }
}

// ==========================================
// Tab Switching
// ==========================================
function switchTab(tab) {
  currentTab = tab;
  document.getElementById("tab-active").classList.toggle("active", tab === "active");
  document.getElementById("tab-backup").classList.toggle("active", tab === "backup");
  document.getElementById("upload-zone").style.display = tab === "active" ? "block" : "none";
  renderFileList();
}

// ==========================================
// Load Files from Backend
// ==========================================
async function loadFiles() {
  const listEl = document.getElementById("file-list");
  listEl.innerHTML = `<div class="file-empty"><div class="file-empty-icon">⏳</div>Loading…</div>`;
  try {
    const res  = await fetch(LIST_URL);
    const data = await res.json();
    filesData  = data;
    document.getElementById("badge-active").textContent = data.counts.active;
    document.getElementById("badge-backup").textContent = data.counts.backup;
    renderFileList();
  } catch (err) {
    listEl.innerHTML = `<div class="file-empty"><div class="file-empty-icon">⚠️</div>Could not load files.<br/>Is the backend running?</div>`;
  }
}

// ==========================================
// Render File List
// ==========================================
function renderFileList() {
  const listEl = document.getElementById("file-list");
  const files  = currentTab === "active" ? filesData.active : filesData.backups;
  listEl.innerHTML = "";

  if (!files || files.length === 0) {
    const label = currentTab === "active" ? "No active documents." : "No backup files.";
    const icon  = currentTab === "active" ? "📄" : "🗄️";
    const sub   = currentTab === "active" ? "Upload a file to get started." : "Deleted files appear here.";
    listEl.innerHTML = `<div class="file-empty"><div class="file-empty-icon">${icon}</div>${label}<br/>${sub}</div>`;
    return;
  }

  files.forEach((file, idx) => {
    const card = document.createElement("div");
    card.className = "file-card";
    card.id = `file-card-${currentTab}-${idx}`;

    const iconClass = getFileIconClass(file.filename);
    const iconLabel = getFileIconLabel(file.filename);
    const chunkInfo = (file.chunks !== undefined) ? `<span class="chunk-tag">${file.chunks} chunks</span>` : "";
    const dateStr   = formatDate(file.modified);
    const sizeStr   = formatBytes(file.size_bytes);

    if (currentTab === "active") {
      card.innerHTML = `
        <div class="file-card-top">
          <div class="file-icon ${iconClass}">${iconLabel}</div>
          <div class="file-info">
            <div class="file-name" title="${file.filename}">${file.filename}</div>
            <div class="file-meta">
              <span>${sizeStr}</span>
              <span>${dateStr}</span>
              ${chunkInfo}
            </div>
          </div>
        </div>
        <div class="file-actions">
          <button class="fa-btn delete" onclick="initiateDelete('${file.filename}', ${idx})">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/></svg>
            Move to Backup
          </button>
        </div>
        <div class="confirm-row" id="confirm-${idx}">
          <span>Move "${file.filename}" to backup?</span>
          <button class="confirm-no"  onclick="cancelConfirm(${idx})">No</button>
          <button class="confirm-yes" onclick="executeDelete('${file.filename}', ${idx})">Yes, Move</button>
        </div>`;
    } else {
      card.innerHTML = `
        <div class="file-card-top">
          <div class="file-icon ${iconClass}">${iconLabel}</div>
          <div class="file-info">
            <div class="file-name" title="${file.filename}">${file.filename}</div>
            <div class="file-meta">
              <span>${sizeStr}</span>
              <span>${dateStr}</span>
            </div>
          </div>
        </div>
        <div class="file-actions">
          <button class="fa-btn restore" onclick="executeRestore('${file.filename}', ${idx})">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            Restore
          </button>
          <button class="fa-btn perm-delete" onclick="initiatePermanentDelete('${file.filename}', ${idx})">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
            Delete Forever
          </button>
        </div>
        <div class="confirm-row" id="confirm-${idx}">
          <span>Permanently delete "${file.filename}"? Cannot be undone.</span>
          <button class="confirm-no"  onclick="cancelConfirm(${idx})">No</button>
          <button class="confirm-yes" onclick="executePermanentDelete('${file.filename}', ${idx})">Delete</button>
        </div>`;
    }

    listEl.appendChild(card);
  });
}

// ==========================================
// DELETE (active -> backup)
// ==========================================
function initiateDelete(filename, idx) {
  const row = document.getElementById(`confirm-${idx}`);
  if (row) row.classList.add("visible");
}

function cancelConfirm(idx) {
  const row = document.getElementById(`confirm-${idx}`);
  if (row) row.classList.remove("visible");
}

async function executeDelete(filename, idx) {
  const btn = document.querySelector(`#file-card-active-${idx} .fa-btn.delete`);
  if (btn) { btn.disabled = true; btn.classList.add("loading"); }

  try {
    const res  = await fetch(`${FILES_URL}/${encodeURIComponent(filename)}`, { method: "DELETE" });
    const data = await res.json();

    if (!res.ok) {
      showToast(`⚠️ ${data.detail || "Delete failed"}`, "error");
      if (btn) { btn.disabled = false; btn.classList.remove("loading"); }
      cancelConfirm(idx);
      return;
    }

    showToast(`🗑️ "${filename}" moved to backup. ${data.chunks_removed} chunks removed.`, "success");
    addSystemMessage(`🗑️ "${filename}" moved to backup. ${data.chunks_removed} embedding chunks removed.`);
    await loadFiles();

  } catch (err) {
    showToast("⚠️ Could not delete file. Is the backend running?", "error");
    console.error(err);
    if (btn) { btn.disabled = false; btn.classList.remove("loading"); }
  }
}

// ==========================================
// RESTORE (backup -> active)
// ==========================================
async function executeRestore(filename, idx) {
  const btn = document.querySelector(`#file-card-backup-${idx} .fa-btn.restore`);
  if (btn) { btn.disabled = true; btn.classList.add("loading"); }

  try {
    const res  = await fetch(`${FILES_URL}/${encodeURIComponent(filename)}/restore`, { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      showToast(`⚠️ ${data.detail || "Restore failed"}`, "error");
      addSystemMessage(`⚠️ Restore failed for "${filename}": ${data.detail}`);
      if (btn) { btn.disabled = false; btn.classList.remove("loading"); }
      return;
    }

    showToast(`♻️ "${filename}" restored! ${data.chunks_embedded} chunks re-embedded.`, "success");
    addSystemMessage(`♻️ "${filename}" restored. ${data.chunks_embedded} chunks added.`);
    await loadFiles();

  } catch (err) {
    showToast("⚠️ Could not restore file. Is the backend running?", "error");
    console.error(err);
    if (btn) { btn.disabled = false; btn.classList.remove("loading"); }
  }
}

// ==========================================
// PERMANENT DELETE
// ==========================================
function initiatePermanentDelete(filename, idx) {
  const row = document.getElementById(`confirm-${idx}`);
  if (row) row.classList.add("visible");
}

async function executePermanentDelete(filename, idx) {
  const btn = document.querySelector(`#file-card-backup-${idx} .fa-btn.perm-delete`);
  if (btn) { btn.disabled = true; btn.classList.add("loading"); }

  try {
    const res  = await fetch(`${FILES_URL}/${encodeURIComponent(filename)}/permanent`, { method: "DELETE" });
    const data = await res.json();

    if (!res.ok) {
      showToast(`⚠️ ${data.detail || "Delete failed"}`, "error");
      if (btn) { btn.disabled = false; btn.classList.remove("loading"); }
      cancelConfirm(idx);
      return;
    }

    showToast(`🗑️ "${filename}" permanently deleted.`, "warning");
    addSystemMessage(`🗑️ "${filename}" permanently deleted. This cannot be undone.`);
    await loadFiles();

  } catch (err) {
    showToast("⚠️ Could not delete. Is the backend running?", "error");
    console.error(err);
    if (btn) { btn.disabled = false; btn.classList.remove("loading"); }
  }
}

// ==========================================
// Drag & Drop Upload
// ==========================================
function onDragOver(e) {
  e.preventDefault();
  document.getElementById("upload-zone").classList.add("drag-over");
}
function onDragLeave(e) {
  document.getElementById("upload-zone").classList.remove("drag-over");
}
function onDrop(e) {
  e.preventDefault();
  document.getElementById("upload-zone").classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadDocument(file, false);
}

function triggerUpload()      { document.getElementById("doc-upload").click(); }
function triggerPanelUpload() { document.getElementById("panel-upload").click(); }

// ==========================================
// Sidebar / Chat History
// ==========================================
function loadChatHistorySidebar() {
  const div   = document.getElementById("chat-history");
  div.innerHTML = "";
  const chats = JSON.parse(localStorage.getItem("ragChats") || "{}");

  Object.keys(chats).reverse().forEach(chatId => {
    const btn = document.createElement("div");
    btn.classList.add("history-item");
    if (chatId === currentChatId) btn.classList.add("active");
    btn.textContent = chats[chatId].title || "Untitled";
    btn.onclick = () => loadChat(chatId);
    btn.oncontextmenu = (e) => {
      e.preventDefault();
      if (!confirm("Delete this chat?")) return;
      delete chats[chatId];
      localStorage.setItem("ragChats", JSON.stringify(chats));
      if (currentChatId === chatId) {
        document.getElementById("chat-box").innerHTML = buildEmptyState();
        chatHistory = [];
        currentChatId = null;
        document.getElementById("session-label").textContent = "—";
      }
      loadChatHistorySidebar();
    };
    div.appendChild(btn);
  });
}

function buildEmptyState() {
  return `<div class="empty-state" id="empty-state">
    <div class="empty-state-icon">◈</div>
    <h2>Ask anything</h2>
    <p>Upload documents via File Manager,<br/>then ask questions about them.</p>
  </div>`;
}

function saveCurrentChat() {
  if (!currentChatId || !chatHistory.length) return;
  const chats = JSON.parse(localStorage.getItem("ragChats") || "{}");
  chats[currentChatId] = {
    title: chatHistory[0]?.content?.substring(0, 36) || "New Chat",
    messages: chatHistory
  };
  localStorage.setItem("ragChats", JSON.stringify(chats));
  loadChatHistorySidebar();
}

function loadChat(chatId) {
  const chats = JSON.parse(localStorage.getItem("ragChats") || "{}");
  const chat  = chats[chatId];
  if (!chat) return;
  currentChatId = chatId;
  chatHistory   = chat.messages;
  document.getElementById("session-label").textContent = chat.title || chatId;
  document.getElementById("chat-box").innerHTML = "";
  chat.messages.forEach(m => addMessage(m.content, m.role, m.sources || []));
}

function newChat() {
  currentChatId = "chat_" + Date.now();
  chatHistory   = [];
  document.getElementById("chat-box").innerHTML = buildEmptyState();
  document.getElementById("session-label").textContent = "New Chat";
  loadChatHistorySidebar();
}

// ==========================================
// Messages
// ==========================================
function removeEmptyState() {
  const es = document.getElementById("empty-state");
  if (es) es.remove();
}

function addMessage(text, sender, sources = []) {
  removeEmptyState();
  const box  = document.getElementById("chat-box");
  const wrap = document.createElement("div");
  wrap.classList.add("message", sender);

  const avatar = document.createElement("div");
  avatar.classList.add("msg-avatar");
  avatar.textContent = sender === "user" ? "U" : "AI";

  const body = document.createElement("div");
  body.classList.add("msg-body");

  const content = document.createElement("div");
  content.classList.add("msg-content");
  content.textContent = text;
  body.appendChild(content);

  if (sources && sources.length) {
    const src = document.createElement("div");
    src.classList.add("msg-sources");
    src.innerHTML = `Sources: ${sources.map(s => `<span>${s}</span>`).join(", ")}`;
    body.appendChild(src);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(body);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

function addSystemMessage(text) {
  if (!currentChatId) currentChatId = "chat_" + Date.now();
  addMessage(text, "bot");
  chatHistory.push({ role: "bot", content: text });
  saveCurrentChat();
}

function showLoader() {
  removeEmptyState();
  const box = document.getElementById("chat-box");
  const row = document.createElement("div");
  row.classList.add("loader-row");
  row.id = "loader";
  row.innerHTML = `<div class="loader-avatar">AI</div><div class="loader-dots"><span></span><span></span><span></span></div>`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function removeLoader() {
  const l = document.getElementById("loader");
  if (l) l.remove();
}

// ==========================================
// Send Message
// ==========================================
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey && !isLoading) {
    e.preventDefault();
    sendMessage();
  }
}

async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById("user-input");
  const text  = input.value.trim();
  if (!text) return;

  if (!currentChatId) currentChatId = "chat_" + Date.now();

  isLoading = true;
  document.getElementById("send-btn").disabled = true;

  addMessage(text, "user");
  chatHistory.push({ role: "user", content: text });
  document.getElementById("session-label").textContent =
    chatHistory[0]?.content?.substring(0, 30) || currentChatId;

  input.value = "";
  input.style.height = "auto";
  showLoader();

  try {
    const res  = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text })
    });
    const data = await res.json();
    removeLoader();

    const answer  = data.answer || "⚠️ No solution returned.";

    const sources = data.sources || [];
    addMessage(answer, "bot", sources);
    chatHistory.push({ role: "bot", content: answer, sources });
    saveCurrentChat();

  } catch (err) {
    removeLoader();
    addMessage("⚠️ Could not connect to the server. Is the backend running?", "bot");
    console.error(err);
  }

  isLoading = false;
  document.getElementById("send-btn").disabled = false;
}

// ==========================================
// Upload Document
// ==========================================
async function uploadDocument(file, replace = false) {
  if (!currentChatId) currentChatId = "chat_" + Date.now();

  showToast(`📤 Uploading "${file.name}"…`, "info", 2000);
  showLoader();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch(`${UPLOAD_URL}?replace=${replace}`, { method: "POST", body: formData });
    const data = await res.json();
    removeLoader();

    // Duplicate detected — open modal for user decision
    if (data.action_required) {
      pendingFile = file;
      openModal(data.type, file.name);
      return;
    }

    const msg = data.message || data.error || "Upload complete.";
    addMessage(`📎 ${msg}`, "bot");
    chatHistory.push({ role: "bot", content: `📎 ${msg}` });
    saveCurrentChat();
    showToast(msg, data.error ? "error" : "success");

    if (filePanelOpen) await loadFiles();

  } catch (err) {
    removeLoader();
    addMessage("⚠️ Upload failed. Is the backend running?", "bot");
    showToast("⚠️ Upload failed.", "error");
    console.error(err);
  }
}

async function retryUpload() {
  closeModal();
  if (!pendingFile) return;
  await uploadDocument(pendingFile, true);
  pendingFile = null;
}

// ==========================================
// Duplicate Modal
// ==========================================
function openModal(type, filename) {
  const modal = document.getElementById("dup-modal");
  document.getElementById("modal-icon").textContent  = type === "same_name" ? "📝" : "🔄";
  document.getElementById("modal-title").textContent = type === "same_name" ? "Name Already Exists" : "Identical Content";
  document.getElementById("modal-msg").textContent   = type === "same_name"
    ? `A file named "${filename}" is already in the system. Replace it?`
    : `A file with the same content as "${filename}" already exists. Replace?`;
  modal.classList.add("open");
}

function closeModal() {
  document.getElementById("dup-modal").classList.remove("open");
  pendingFile = null;
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("dup-modal").addEventListener("click", function(e) {
    if (e.target === this) closeModal();
  });
});

// ==========================================
// Backend History
// ==========================================
async function viewBackendHistory() {
  try {
    const res  = await fetch(HISTORY_URL);
    const data = await res.json();
    newChat();
    document.getElementById("session-label").textContent = "Backend History";
    if (!data.length) { addMessage("📭 No backend history found.", "bot"); return; }
    data.slice(0, 40).forEach(item => {
      addMessage(item.question, "user");
      addMessage(item.answer, "bot", item.sources ? item.sources.split(", ") : []);
    });
  } catch (err) {
    showToast("⚠️ Could not load backend history.", "error");
    console.error(err);
  }
}

async function clearBackendHistory() {
  if (!confirm("Clear all backend chat history?")) return;
  try {
    await fetch(CLEAR_HIST_URL, { method: "DELETE" });
    showToast("🗑️ Backend history cleared.", "success");
  } catch (err) {
    showToast("⚠️ Failed to clear history.", "error");
    console.error(err);
  }
}
