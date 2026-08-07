const DEFAULT_PORT = 8765;

const statusEl = document.getElementById("status");
const portEl = document.getElementById("port");
const pasteEl = document.getElementById("paste");
const resultEl = document.getElementById("result");

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function loadPort() {
  const { port } = await chrome.storage.sync.get({ port: DEFAULT_PORT });
  portEl.value = port || DEFAULT_PORT;
}

portEl.addEventListener("change", async () => {
  const port = parseInt(portEl.value, 10) || DEFAULT_PORT;
  await chrome.storage.sync.set({ port });
  checkHealth();
});

async function checkHealth() {
  statusEl.className = "status";
  statusEl.textContent = "checking backend…";
  const resp = await chrome.runtime.sendMessage({ type: "trace:health" });
  if (resp && resp.ok) {
    const d = resp.data;
    statusEl.className = "status ok";
    statusEl.textContent = `connected — ${d.chunks} chunks across ${d.papers.length} paper(s)`;
  } else {
    statusEl.className = "status bad";
    statusEl.innerHTML =
      "backend not reachable. Start it with:<br><code>trace serve --index corpus.json</code>";
  }
}

async function explain(generic) {
  const text = pasteEl.value.trim();
  if (text.length < 8) {
    resultEl.textContent = "Paste a passage first.";
    return;
  }
  resultEl.textContent = "Retrieving…";
  const resp = await chrome.runtime.sendMessage({ type: "trace:explain", text, generic });
  if (!resp || !resp.ok) {
    resultEl.innerHTML = `<span style="color:#b42318">${escapeHtml(
      (resp && resp.error) || "failed"
    )}</span>`;
    return;
  }
  const { explanation, sources } = resp.data;
  let html = `<div>${escapeHtml(explanation).replace(/\n/g, "<br>")}</div>`;
  if (sources && sources.length) {
    html += '<div class="src" style="margin-top:8px">Sources:<br>';
    html += sources.map((s) => escapeHtml(s)).join("<br>");
    html += "</div>";
  }
  resultEl.innerHTML = html;
}

document.getElementById("explainBtn").addEventListener("click", () => explain(false));
document.getElementById("genericBtn").addEventListener("click", () => explain(true));

loadPort().then(checkHealth);
