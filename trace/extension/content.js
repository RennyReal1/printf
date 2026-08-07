// Trace content script: capture the highlighted passage, show an "Explain"
// chip near it, and render the grounded explanation in a side panel.
//
// The network call is delegated to the background service worker (see
// background.js) — this script only deals with selection and DOM.

(() => {
  let chip = null;
  let panel = null;

  function removeChip() {
    if (chip) {
      chip.remove();
      chip = null;
    }
  }

  function selectionText() {
    const s = window.getSelection();
    return s ? s.toString().trim() : "";
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function truncate(s, n) {
    return s.length > n ? escapeHtml(s.slice(0, n)) + "…" : escapeHtml(s);
  }

  document.addEventListener("mouseup", () => {
    // Defer so the selection is finalized before we read it.
    setTimeout(() => {
      const text = selectionText();
      removeChip();
      if (text.length < 8) return; // ignore stray clicks / tiny selections
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      chip = document.createElement("button");
      chip.className = "trace-chip";
      chip.textContent = "Explain with Trace";
      chip.style.top = `${window.scrollY + rect.bottom + 6}px`;
      chip.style.left = `${window.scrollX + rect.left}px`;
      // Keep the selection alive when the chip is pressed.
      chip.addEventListener("mousedown", (e) => e.preventDefault());
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        explain(text);
      });
      document.body.appendChild(chip);
    }, 0);
  });

  document.addEventListener("mousedown", (e) => {
    if (chip && !chip.contains(e.target)) removeChip();
  });

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "trace:explainSelection" && msg.text) {
      explain(msg.text.trim());
    }
  });

  function ensurePanel() {
    if (panel) return panel;
    panel = document.createElement("div");
    panel.className = "trace-panel";
    panel.innerHTML =
      '<div class="trace-panel-head">' +
      '<span class="trace-panel-title">Trace</span>' +
      '<button class="trace-panel-close" title="Close">×</button>' +
      "</div>" +
      '<div class="trace-panel-body"></div>';
    panel.querySelector(".trace-panel-close").addEventListener("click", () => {
      panel.remove();
      panel = null;
    });
    document.body.appendChild(panel);
    return panel;
  }

  function setBody(html) {
    ensurePanel().querySelector(".trace-panel-body").innerHTML = html;
  }

  async function explain(text) {
    removeChip();
    setBody(
      `<div class="trace-quote">${truncate(text, 240)}</div>` +
        '<div class="trace-loading">Retrieving from your corpus…</div>'
    );

    let resp;
    try {
      resp = await chrome.runtime.sendMessage({ type: "trace:explain", text });
    } catch (e) {
      setBody(`<div class="trace-error">Extension error: ${escapeHtml(String(e))}</div>`);
      return;
    }

    if (!resp || !resp.ok) {
      setBody(
        '<div class="trace-error">Couldn’t reach the Trace backend.<br><br>' +
          "Start it in a terminal:<br>" +
          "<code>trace serve --index corpus.json</code>" +
          `<br><br><small>${escapeHtml(resp && resp.error ? resp.error : "connection failed")}</small></div>`
      );
      return;
    }

    const { explanation, sources } = resp.data;
    let html = `<div class="trace-quote">${truncate(text, 240)}</div>`;
    html += `<div class="trace-explanation">${escapeHtml(explanation).replace(/\n/g, "<br>")}</div>`;
    if (sources && sources.length) {
      html += '<div class="trace-sources-label">Sources</div><ul class="trace-sources">';
      for (const s of sources) html += `<li>${escapeHtml(s)}</li>`;
      html += "</ul>";
    }
    setBody(html);
  }
})();
