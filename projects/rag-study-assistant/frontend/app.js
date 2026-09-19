const API_BASE = "http://localhost:8000";

const log = document.getElementById("log");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");

const ingestForm = document.getElementById("ingest-form");
const ingestFile = document.getElementById("ingest-file");
const ingestStatus = document.getElementById("ingest-status");

function getSessionId() {
  let id = localStorage.getItem("session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("session_id", id);
  }
  return id;
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function appendSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `${sources.length} source chunk(s)`;
  details.appendChild(summary);
  for (const source of sources) {
    const chunk = document.createElement("div");
    chunk.className = "source-chunk";
    chunk.textContent = `[${source.source}] ${source.text}`;
    details.appendChild(chunk);
  }
  container.appendChild(details);
}

// Parses one accumulated SSE buffer into complete "\n\n"-delimited events,
// returning the leftover partial tail to keep accumulating.
function consumeSseEvents(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const tail = parts.pop();
  for (const part of parts) {
    const lines = part.split("\n");
    let eventType = "message";
    let data = "";
    for (const line of lines) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) data = line.slice(5).trim();
    }
    if (data) onEvent(eventType, JSON.parse(data));
  }
  return tail;
}

ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = ingestFile.files[0];
  if (!file) return;

  ingestStatus.textContent = `Ingesting ${file.name}...`;
  ingestFile.disabled = true;

  const body = new FormData();
  body.append("file", file);
  body.append("source", file.name);

  try {
    const resp = await fetch(`${API_BASE}/documents/ingest`, { method: "POST", body });
    if (!resp.ok) {
      throw new Error(`Request failed: ${resp.status}`);
    }
    const data = await resp.json();
    ingestStatus.textContent = `Ingested "${data.source}" — ${data.chunks} chunk(s)`;
    ingestForm.reset();
  } catch (err) {
    ingestStatus.textContent = `Error: ${err.message}`;
  } finally {
    ingestFile.disabled = false;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  input.value = "";
  input.disabled = true;

  const assistantEl = appendMessage("assistant", "");
  let answerText = "";

  try {
    const resp = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: getSessionId(), message }),
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`Request failed: ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = consumeSseEvents(buffer, (eventType, payload) => {
        if (eventType === "message") {
          answerText += payload.token;
          assistantEl.textContent = answerText;
          log.scrollTop = log.scrollHeight;
        } else if (eventType === "done") {
          appendSources(assistantEl, payload.sources);
        } else if (eventType === "error") {
          assistantEl.textContent = `Error: ${payload.error}`;
        }
      });
    }
  } catch (err) {
    assistantEl.textContent = `Error: ${err.message}`;
  } finally {
    input.disabled = false;
    input.focus();
  }
});
