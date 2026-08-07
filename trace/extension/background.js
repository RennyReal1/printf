// Trace service worker: the only component that talks to the local backend.
// Doing the fetch here (not in the content script) sidesteps page CSP/CORS —
// the worker has host_permissions for localhost and the backend returns
// permissive CORS headers anyway.

const DEFAULT_PORT = 8765;

async function backendBase() {
  const { port } = await chrome.storage.sync.get({ port: DEFAULT_PORT });
  return `http://127.0.0.1:${port || DEFAULT_PORT}`;
}

async function call(path, options) {
  const base = await backendBase();
  const resp = await fetch(base + path, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "trace-explain",
    title: "Explain selection with Trace",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "trace-explain" && tab && tab.id != null) {
    chrome.tabs.sendMessage(tab.id, {
      type: "trace:explainSelection",
      text: info.selectionText,
    });
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "trace:explain") {
    call("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        highlight: msg.text,
        paper: msg.paper || "",
        generic: !!msg.generic,
      }),
    })
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true; // keep the channel open for the async reply
  }
  if (msg.type === "trace:health") {
    call("/health", {})
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true;
  }
});
