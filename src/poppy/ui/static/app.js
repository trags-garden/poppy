// Poppy memory UI — vanilla, no build step.

const TYPES = ["fact", "decision", "preference", "lesson", "context"];

const state = {
  scope: "today",
  type: null,
  project: null,
  source: null,
  q: "",
  items: [],
  selected: null,
  facets: { types: {}, projects: {}, sources: {}, totals: {} },
  density: "comfortable",
  today: null,
};

// ---------- API ----------
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

async function loadList() {
  if (state.scope === "today") {
    setView("today");
    if (!state.today) await loadToday();
    else renderToday();
    return;
  }
  const params = new URLSearchParams({ scope: state.scope });
  if (state.q) params.set("q", state.q);
  if (state.type) params.set("type", state.type);
  if (state.project) params.set("project", state.project);
  if (state.source) params.set("source", state.source);
  const data = await api(`/api/memories?${params.toString()}`);
  state.items = data.items;
  renderList();
  // Re-render the detail pane if the selected item changed identity (e.g. after edit/delete).
  if (state.selected) {
    const fresh = state.items.find((i) => i.id === state.selected.id);
    if (fresh) {
      state.selected = fresh;
      renderDetail();
    } else if (state.scope === "active") {
      // Selected item moved to tombstoned — try to fetch via direct GET.
      try {
        const m = await api(`/api/memories/${state.selected.id}`);
        state.selected = m;
        renderDetail();
      } catch {
        state.selected = null;
        renderDetail();
      }
    }
  }
}

async function loadFacets() {
  const f = await api("/api/facets");
  state.facets = f;
  renderFacets();
}

async function loadStats() {
  const s = await api("/api/stats");
  document.getElementById("engine-name").textContent = s.engine.name;
  document.getElementById("engine-meta").textContent =
    `${s.engine.memory_count.toLocaleString()} memories · ${(s.engine.storage_bytes / 1024).toFixed(1)} KB`;
  document.getElementById("status").textContent =
    `${s.engine.memory_count} memories · ${s.tombstoned} tombstoned`;
}

async function loadToday() {
  const t = await api("/api/today");
  state.today = t;
  document.getElementById("count-today").textContent = t.totals.today;
  if (state.scope === "today") renderToday();
}

// ---------- Rendering ----------
function applyFilterAndJump(stateMutator) {
  // Filtering implies you want to see results. Today doesn't honor facet filters,
  // so any filter click bumps the view to "All memories".
  stateMutator();
  if (state.scope === "today") {
    state.scope = "active";
    document.querySelectorAll("#scope button").forEach((b) => {
      b.classList.toggle("on", b.dataset.scope === "active");
    });
  }
  loadList();
  renderFacets();
}

function renderFacets() {
  const totals = state.facets.totals || {};
  document.getElementById("count-active").textContent = totals.active ?? "—";
  document.getElementById("count-tomb").textContent = totals.tombstoned ?? "—";

  // Types
  const typeNav = document.getElementById("facet-type");
  typeNav.innerHTML = "";
  const allTypes = ["all", ...TYPES];
  for (const t of allTypes) {
    const btn = document.createElement("button");
    const count = t === "all"
      ? Object.values(state.facets.types || {}).reduce((a, b) => a + b, 0)
      : (state.facets.types?.[t] || 0);
    btn.innerHTML = `<span>${t === "all" ? "All types" : t}</span><span class="count">${count}</span>`;
    const matches = (t === "all" && state.type === null) || state.type === t;
    if (matches) btn.classList.add("on");
    btn.addEventListener("click", () => {
      applyFilterAndJump(() => {
        state.type = t === "all" ? null : t;
      });
    });
    typeNav.appendChild(btn);
  }

  // Projects
  const projNav = document.getElementById("facet-project");
  projNav.innerHTML = "";
  const projects = Object.entries(state.facets.projects || {}).sort((a, b) => b[1] - a[1]);
  const allProj = document.createElement("button");
  allProj.innerHTML = `<span>All projects</span><span class="count">${
    projects.reduce((a, [, n]) => a + n, 0)
  }</span>`;
  if (state.project === null) allProj.classList.add("on");
  allProj.addEventListener("click", () => {
    applyFilterAndJump(() => {
      state.project = null;
    });
  });
  projNav.appendChild(allProj);
  for (const [p, n] of projects.slice(0, 12)) {
    const btn = document.createElement("button");
    btn.innerHTML = `<span>${escapeHtml(p)}</span><span class="count">${n}</span>`;
    if (state.project === p) btn.classList.add("on");
    btn.addEventListener("click", () => {
      applyFilterAndJump(() => {
        state.project = state.project === p ? null : p;
      });
    });
    projNav.appendChild(btn);
  }

  // Sources
  const srcNav = document.getElementById("facet-source");
  srcNav.innerHTML = "";
  const sources = Object.entries(state.facets.sources || {}).sort((a, b) => b[1] - a[1]);
  const allSrc = document.createElement("button");
  allSrc.innerHTML = `<span>All sources</span><span class="count">${
    sources.reduce((a, [, n]) => a + n, 0)
  }</span>`;
  if (state.source === null) allSrc.classList.add("on");
  allSrc.addEventListener("click", () => {
    applyFilterAndJump(() => {
      state.source = null;
    });
  });
  srcNav.appendChild(allSrc);
  for (const [s, n] of sources) {
    const btn = document.createElement("button");
    btn.innerHTML = `<span>${escapeHtml(s)}</span><span class="count">${n}</span>`;
    if (state.source === s) btn.classList.add("on");
    btn.addEventListener("click", () => {
      applyFilterAndJump(() => {
        state.source = state.source === s ? null : s;
      });
    });
    srcNav.appendChild(btn);
  }
}

function setView(scope) {
  const isToday = scope === "today";
  document.getElementById("today").hidden = !isToday;
  document.getElementById("list").hidden = isToday;
  document.getElementById("list-head").hidden = isToday;
  if (isToday) document.getElementById("empty").hidden = true;
}

function renderList() {
  setView(state.scope);
  if (state.scope === "today") return;
  const list = document.getElementById("list");
  const empty = document.getElementById("empty");
  list.innerHTML = "";
  const eyebrow = document.getElementById("list-eyebrow");
  const title = document.getElementById("list-title");

  if (state.scope === "tombstoned") {
    eyebrow.textContent = "Tombstoned · restorable for 7 days";
    title.innerHTML = `Recently <em>forgotten</em>`;
  } else if (state.q) {
    eyebrow.textContent = `Search · "${state.q}"`;
    title.innerHTML = `${state.items.length} match${state.items.length === 1 ? "" : "es"}`;
  } else {
    eyebrow.textContent = "All memories · recent first";
    title.textContent = labelFor(state);
  }

  if (state.items.length === 0) {
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  for (const m of state.items) {
    const li = document.createElement("li");
    li.className = "row";
    if (m.tombstoned) li.classList.add("tombstoned");
    if (state.selected && state.selected.id === m.id) li.classList.add("active");
    li.dataset.id = m.id;

    const pinClass = TYPES.includes(m.memory_type) ? m.memory_type : "fact";
    const proj = m.project ? `<span class="proj">${escapeHtml(m.project)}</span>` : "";
    const src = m.source_type ? `<span class="src">${escapeHtml(m.source_type)}</span>` : "";
    const score = m.score != null ? `<span class="score">${m.score.toFixed(3)}</span>` : "";
    const tombTag = m.tombstoned ? `<span class="tomb-tag">tombstoned</span>` : "";
    // Lifecycle relations: forward (this Replaces N) and backward (Replaced by).
    let relBadge = "";
    if (m.tombstoned && m.superseded_by) {
      relBadge = `<span class="rel-badge" title="Replaced by ${escapeHtml(m.superseded_by)}">↻</span>`;
    } else if (!m.tombstoned && m.related_to && m.related_to.length) {
      const n = m.related_to.length;
      relBadge = `<span class="rel-badge" title="Replaces ${n} memor${n === 1 ? "y" : "ies"}">↺${n > 1 ? n : ""}</span>`;
    }
    const age = ageString(m.created_at);

    li.innerHTML = `
      <span class="pin ${pinClass}"></span>
      <div class="body">
        <div class="content">${escapeHtml(m.content)}</div>
        <div class="meta">
          <span class="type">${m.memory_type}</span>
          ${proj}
          ${src}
          <span class="age">${age}</span>
          ${tombTag}
          ${relBadge}
          ${score}
        </div>
      </div>
    `;
    li.addEventListener("click", () => select(m));
    list.appendChild(li);
  }
}

function renderToday() {
  const root = document.getElementById("today");
  const t = state.today;
  if (!t) {
    root.innerHTML = `<p class="margin-note">Loading today…</p>`;
    return;
  }

  const headerLabel = `${t.weekday} · ${t.month} · WEEK ${t.iso_week}`;

  const breakdown = t.today_breakdown || { types: {}, projects: {}, sources: {} };
  const distinctProjects = Object.keys(breakdown.projects).length;
  const distinctSources = Object.keys(breakdown.sources).length;
  const topType = Object.entries(breakdown.types).sort((a, b) => b[1] - a[1])[0];

  const stats = [
    [t.totals.today, "captured today"],
    [distinctProjects, distinctProjects === 1 ? "project" : "projects"],
    [distinctSources, distinctSources === 1 ? "source" : "sources"],
    [t.totals.this_week, "this week"],
  ];

  const feedHtml = renderFeed(t.feed);

  const topProjectsHtml = (t.top_projects_week || [])
    .map(
      ([p, n]) =>
        `<div class="row-line"><span><span class="bullet">●</span>${escapeHtml(p)}</span><span class="num">${n}</span></div>`,
    )
    .join("") || `<div class="quiet-empty">No project memories this week.</div>`;

  const sparkHtml = renderSpark(t.activity_7d || []);

  const quietHtml = (t.quiet || []).length
    ? t.quiet
        .map(
          (q) =>
            `<div class="quiet-line">You haven't touched <em>${escapeHtml(q.project)}</em> in ${q.days_quiet} days.</div>`,
        )
        .join("")
    : `<div class="quiet-empty">All your projects look alive. Carry on.</div>`;

  root.innerHTML = `
    <section class="today-hero">
      <div class="today-date">
        <span class="day-num">${t.day}</span>
        <span>${headerLabel}</span>
      </div>
      <h1 class="today-quote">${t.summary}</h1>
      <div class="today-meta">
        ${stats
          .map(([n, label]) => `<div class="stat"><strong>${n}</strong><span>${label}</span></div>`)
          .join("")}
      </div>
    </section>

    <div class="today-section-head">
      <div>
        <div class="eyebrow">The feed</div>
        <h2>What you <em>captured</em> today</h2>
      </div>
    </div>

    ${feedHtml}

    <div class="today-cards">
      <div class="today-card">
        <div class="h-mono-label">Top projects · 7 days</div>
        ${topProjectsHtml}
      </div>
      <div class="today-card">
        <div class="h-mono-label">Captures · 7 days</div>
        ${sparkHtml}
      </div>
      <div class="today-card">
        <div class="h-mono-label">Quiet observations</div>
        ${quietHtml}
      </div>
    </div>
  `;

  // Click handlers on feed items → jump to detail view.
  root.querySelectorAll(".feed-item").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.dataset.id;
      const item = (state.today.feed || []).find((m) => m.id === id);
      if (item) select(item);
    });
  });

  wireSpark(root);
}

function renderFeed(feed) {
  if (!feed || !feed.length) {
    return `<div class="empty" style="display:flex; padding: 40px 0;">
      <div class="empty-mark">∅</div>
      <h3>Nothing yet today.</h3>
      <p>Memories captured today will appear here, on a timeline.</p>
    </div>`;
  }
  const bands = ["Morning", "Midday", "Afternoon", "Evening"];
  // Group feed items by band (preserve oldest→newest order within each).
  const byBand = new Map(bands.map((b) => [b, []]));
  for (const m of feed) {
    if (byBand.has(m.band)) byBand.get(m.band).push(m);
  }

  const groupHtml = bands
    .filter((b) => byBand.get(b).length > 0)
    .map((b) => {
      const items = byBand.get(b);
      const itemsHtml = items
        .map((m) => {
          const time = new Date(m.created_at).toLocaleTimeString(undefined, {
            hour: "2-digit",
            minute: "2-digit",
          });
          const pinClass = ["fact", "decision", "preference", "lesson", "context"].includes(m.memory_type)
            ? m.memory_type
            : "fact";
          const proj = m.project ? escapeHtml(m.project) : "—";
          return `
            <div class="feed-item" data-id="${escapeHtml(m.id)}">
              <span class="when">${time}</span>
              <span class="pin ${pinClass}"></span>
              <div class="body">
                <div class="text">${escapeHtml(m.content)}</div>
                <div class="meta">
                  <span>${m.memory_type}</span>
                  <span>${proj}</span>
                  <span>${escapeHtml(m.source_type)}</span>
                </div>
              </div>
            </div>
          `;
        })
        .join("");
      return `
        <section class="feed-band">
          <aside class="band-label">
            <span class="tick on">${b}</span>
            <span class="band-count">${items.length}</span>
          </aside>
          <div class="band-items">${itemsHtml}</div>
        </section>
      `;
    })
    .join("");

  return `<div class="feed-grouped">${groupHtml}</div>`;
}

function renderSpark(activity) {
  if (!activity.length) return `<div class="quiet-empty">No activity yet.</div>`;
  const w = 200, h = 60;
  const max = Math.max(1, ...activity.map((d) => d.count));
  const step = activity.length > 1 ? w / (activity.length - 1) : 0;
  const pts = activity.map((d, i) => {
    const x = i * step;
    const y = h - 6 - ((h - 12) * d.count) / max;
    return [x, y, d];
  });
  const path = pts.map(([x, y], i) => (i === 0 ? `M${x},${y}` : `L${x},${y}`)).join(" ");
  const fill = `${path} L${w},${h} L0,${h} Z`;
  const labels = activity.map((d) => {
    const dt = new Date(d.date);
    return dt.toLocaleDateString(undefined, { weekday: "short" }).slice(0, 3);
  });

  const dots = pts
    .map(([x, y, d], i) => {
      const dt = new Date(d.date);
      const day = dt.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
      const tip = `${d.count} · ${day}`;
      return `<g class="spark-pt" data-x="${x}" data-y="${y}" data-tip="${escapeHtml(tip)}">
        <circle class="hot" cx="${x}" cy="${h / 2}" r="${Math.max(8, step / 2)}" />
        <circle class="dot" cx="${x}" cy="${y}" r="2.5" />
        <line class="rule" x1="${x}" x2="${x}" y1="0" y2="${h}" />
      </g>`;
    })
    .join("");

  return `
    <div class="spark-wrap">
      <svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="sparkGrad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stop-color="currentColor" stop-opacity="0.28"/>
            <stop offset="1" stop-color="currentColor" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <path d="${fill}" fill="url(#sparkGrad)"/>
        <path d="${path}" fill="none" stroke="currentColor" stroke-width="1.6"/>
        ${dots}
      </svg>
      <div class="spark-tip" hidden></div>
    </div>
    <div class="spark-axis">${labels.map((l) => `<span>${l}</span>`).join("")}</div>
  `;
}

function wireSpark(root) {
  const wrap = root.querySelector(".spark-wrap");
  if (!wrap) return;
  const svg = wrap.querySelector("svg.spark");
  const tip = wrap.querySelector(".spark-tip");
  const pts = wrap.querySelectorAll(".spark-pt");
  pts.forEach((pt) => {
    pt.addEventListener("mouseenter", () => {
      pts.forEach((p) => p.classList.remove("on"));
      pt.classList.add("on");
      const rect = svg.getBoundingClientRect();
      const wrapRect = wrap.getBoundingClientRect();
      const xUser = parseFloat(pt.dataset.x);
      const xPx = (xUser / 200) * rect.width;
      tip.textContent = pt.dataset.tip;
      tip.hidden = false;
      tip.style.left = `${xPx + (rect.left - wrapRect.left)}px`;
    });
  });
  wrap.addEventListener("mouseleave", () => {
    pts.forEach((p) => p.classList.remove("on"));
    tip.hidden = true;
  });
}

function renderDetail() {
  const pane = document.getElementById("detail");
  const m = state.selected;
  if (!m) {
    pane.innerHTML = `
      <div class="detail-empty">
        <div class="rail-num">§ 02<br/>detail</div>
        <p class="margin-note">
          Select a memory to <em>read it</em>, edit, or tombstone. Deletes are reversible for 7 days.
        </p>
      </div>
    `;
    return;
  }

  const tombBanner = m.tombstoned
    ? `<div class="tomb-banner">
        <span><strong>Tombstoned.</strong></span>
        ${
          m.superseded_by
            ? `<span>Replaced by <code class="rel-id" data-id="${escapeHtml(m.superseded_by)}">${escapeHtml(m.superseded_by)}</code></span>`
            : ""
        }
        <span class="countdown">Auto-purges ${ageString(m.tombstone_expires_at, true)}.</span>
      </div>`
    : "";

  const expiryRow = m.tombstoned
    ? ""
    : `<dt>Expires</dt>
       <dd>
         <input id="d-expires-at" type="text" value="${m.expires_at ? escapeHtml(m.expires_at) : ""}"
                placeholder="never (ISO date or 30d)" />
       </dd>`;

  const projects = Object.keys(state.facets.projects || {}).sort();
  const projectOptions = [`<option value="">— none —</option>`]
    .concat(
      projects.map((p) =>
        `<option value="${escapeHtml(p)}" ${p === m.project ? "selected" : ""}>${escapeHtml(p)}</option>`,
      ),
    )
    .concat([
      m.project && !projects.includes(m.project)
        ? `<option value="${escapeHtml(m.project)}" selected>${escapeHtml(m.project)}</option>`
        : "",
    ])
    .join("");

  const typeOptions = TYPES.map(
    (t) => `<option value="${t}" ${t === m.memory_type ? "selected" : ""}>${t}</option>`,
  ).join("");

  pane.innerHTML = `
    <div class="row-spread">
      <span class="type-pill">
        <span class="pin ${TYPES.includes(m.memory_type) ? m.memory_type : "fact"}"></span>
        ${m.memory_type}
      </span>
      <span class="id">${escapeHtml(m.id)}</span>
    </div>

    <h1>${escapeHtml(truncate(m.content, 110))}</h1>

    ${tombBanner}

    <div class="content-box">
      <textarea id="d-content" ${m.tombstoned ? "readonly" : ""}>${escapeHtml(m.content)}</textarea>
    </div>

    <dl class="meta-grid">
      <dt>Type</dt>
      <dd>
        <select id="d-type" ${m.tombstoned ? "disabled" : ""}>${typeOptions}</select>
      </dd>
      <dt>Project</dt>
      <dd>
        <input id="d-project" type="text" list="proj-list" value="${escapeHtml(m.project || "")}" ${m.tombstoned ? "disabled" : ""} placeholder="—" />
        <datalist id="proj-list">${projects
          .map((p) => `<option value="${escapeHtml(p)}"></option>`)
          .join("")}</datalist>
      </dd>
      <dt>Source</dt>
      <dd>${escapeHtml(m.source_type)}${m.source_session_id ? ` · ${escapeHtml(m.source_session_id.slice(0, 8))}` : ""}</dd>
      <dt>Created</dt>
      <dd>${formatDate(m.created_at)}</dd>
      <dt>Updated</dt>
      <dd>${formatDate(m.updated_at)}</dd>
      <dt>Confidence</dt>
      <dd>${m.confidence.toFixed(2)}</dd>
      ${expiryRow}
      ${
        m.related_to && m.related_to.length
          ? `<dt>Replaces</dt>
             <dd>${m.related_to.map((rid) => `<code class="rel-id" data-id="${escapeHtml(rid)}">${escapeHtml(rid)}</code>`).join(" ")}</dd>`
          : ""
      }
    </dl>

    <div class="actions">
      ${
        m.tombstoned
          ? `<button class="btn primary" id="d-restore">Restore</button>`
          : `<button class="btn primary" id="d-save">Save changes</button>
             <button class="btn" id="d-supersede">Supersede</button>
             <button class="btn danger" id="d-delete">Tombstone</button>`
      }
      <button class="btn ghost" id="d-copy">Copy ID</button>
    </div>
  `;

  if (!m.tombstoned) {
    document.getElementById("d-save").addEventListener("click", saveSelected);
    document.getElementById("d-supersede").addEventListener("click", openSupersedeDialog);
    document.getElementById("d-delete").addEventListener("click", deleteSelected);
  } else {
    document.getElementById("d-restore").addEventListener("click", restoreSelected);
  }
  pane.querySelectorAll(".rel-id").forEach((el) => {
    el.addEventListener("click", async () => {
      const rid = el.dataset.id;
      try {
        const m2 = await api(`/api/memories/${rid}`);
        select(m2);
      } catch (e) {
        toast(`Could not load ${rid}: ${e.message}`, { error: true });
      }
    });
  });
  document.getElementById("d-copy").addEventListener("click", () => {
    navigator.clipboard.writeText(m.id);
    toast(`Copied ${m.id}`);
  });
}

// ---------- Actions ----------
function select(m) {
  state.selected = m;
  document.querySelectorAll(".row").forEach((r) => r.classList.toggle("active", r.dataset.id === m.id));
  renderDetail();
}

async function saveSelected() {
  const m = state.selected;
  if (!m) return;
  const expiresInput = document.getElementById("d-expires-at");
  const expiresRaw = expiresInput ? expiresInput.value.trim() : "";
  const patch = {
    content: document.getElementById("d-content").value,
    memory_type: document.getElementById("d-type").value,
    project: document.getElementById("d-project").value || null,
  };
  if (expiresInput) {
    if (!expiresRaw) {
      patch.clear_expiry = true;
    } else if (/^\d/.test(expiresRaw) && /[wdhms]$/i.test(expiresRaw)) {
      patch.ttl = expiresRaw;
    } else {
      patch.expires_at = expiresRaw;
    }
  }
  try {
    const updated = await api(`/api/memories/${m.id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    state.selected = updated;
    toast("Saved.");
    await loadList();
    await loadFacets();
  } catch (e) {
    toast(`Save failed: ${e.message}`, { error: true });
  }
}

function openSupersedeDialog() {
  const m = state.selected;
  if (!m) return;
  const wrap = document.getElementById("modal-wrap");
  const typeOptions = TYPES.map(
    (t) => `<option value="${t}" ${t === m.memory_type ? "selected" : ""}>${t}</option>`,
  ).join("");
  wrap.innerHTML = `
    <div class="modal-overlay" id="modal-overlay">
      <div class="modal-panel" role="dialog" aria-labelledby="modal-title">
        <div class="modal-head">
          <div>
            <div class="eyebrow">Supersede · tombstones the old, writes the new</div>
            <h2 id="modal-title">Replace <em>${escapeHtml(truncate(m.content, 60))}</em></h2>
          </div>
          <button class="btn ghost" id="modal-close" aria-label="Close">Esc</button>
        </div>
        <div class="modal-body">
          <label class="modal-label" for="sup-content">New content</label>
          <textarea id="sup-content" rows="6" placeholder="What replaces it?"></textarea>
          <div class="modal-row">
            <div>
              <label class="modal-label" for="sup-type">Type</label>
              <select id="sup-type">${typeOptions}</select>
            </div>
            <div>
              <label class="modal-label" for="sup-project">Project</label>
              <input id="sup-project" type="text" value="${escapeHtml(m.project || "")}" placeholder="—" />
            </div>
            <div>
              <label class="modal-label" for="sup-ttl">TTL <span class="hint">(e.g. 30d, blank=permanent)</span></label>
              <input id="sup-ttl" type="text" placeholder="never" />
            </div>
          </div>
          <p class="margin-note">
            The old memory <code>${escapeHtml(m.id)}</code> moves to the tombstone (restorable for 7 days).
            The new memory keeps a back-pointer to it.
          </p>
        </div>
        <div class="modal-actions">
          <button class="btn primary" id="sup-submit">Replace</button>
          <button class="btn ghost" id="sup-cancel">Cancel</button>
        </div>
      </div>
    </div>
  `;
  const close = () => {
    wrap.innerHTML = "";
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };
  document.addEventListener("keydown", onKey);
  document.getElementById("modal-close").addEventListener("click", close);
  document.getElementById("sup-cancel").addEventListener("click", close);
  document.getElementById("modal-overlay").addEventListener("click", (e) => {
    if (e.target.id === "modal-overlay") close();
  });
  document.getElementById("sup-submit").addEventListener("click", async () => {
    const content = document.getElementById("sup-content").value.trim();
    if (!content) {
      toast("New content is required.", { error: true });
      return;
    }
    const ttlRaw = document.getElementById("sup-ttl").value.trim();
    const body = {
      content,
      memory_type: document.getElementById("sup-type").value,
      project: document.getElementById("sup-project").value.trim() || null,
    };
    if (ttlRaw) {
      if (/^\d/.test(ttlRaw) && /[wdhms]$/i.test(ttlRaw)) body.ttl = ttlRaw;
      else body.expires_at = ttlRaw;
    }
    try {
      const resp = await api(`/api/memories/${m.id}/supersede`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      close();
      toast(`Replaced ${m.id}. Old is tombstoned (7-day undo).`, {
        undo: () => restoreById(m.id),
      });
      await Promise.all([loadList(), loadFacets(), loadStats()]);
      // Select the new memory if it landed in the visible list.
      const newId = resp.new && resp.new.id;
      if (newId) {
        const fresh = state.items.find((i) => i.id === newId);
        if (fresh) select(fresh);
        else {
          try {
            const fetched = await api(`/api/memories/${newId}`);
            select(fetched);
          } catch { /* ignore */ }
        }
      }
    } catch (e) {
      toast(`Supersede failed: ${e.message}`, { error: true });
    }
  });
  document.getElementById("sup-content").focus();
}

async function deleteSelected() {
  const m = state.selected;
  if (!m) return;
  try {
    await api(`/api/memories/${m.id}`, { method: "DELETE" });
    toast("Tombstoned. Restorable for 7 days.", { undo: () => restoreById(m.id) });
    await loadList();
    await loadFacets();
    await loadStats();
  } catch (e) {
    toast(`Delete failed: ${e.message}`, { error: true });
  }
}

async function restoreSelected() {
  const m = state.selected;
  if (!m) return;
  await restoreById(m.id);
}

async function restoreById(id) {
  try {
    const restored = await api(`/api/memories/${id}/restore`, { method: "POST" });
    state.selected = restored;
    toast("Restored.");
    await loadList();
    await loadFacets();
    await loadStats();
  } catch (e) {
    toast(`Restore failed: ${e.message}`, { error: true });
  }
}

// ---------- Toast ----------
function toast(msg, opts = {}) {
  const wrap = document.getElementById("toast-wrap");
  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = `<span>${escapeHtml(msg)}</span>`;
  if (opts.undo) {
    const u = document.createElement("button");
    u.className = "undo";
    u.textContent = "Undo";
    u.addEventListener("click", () => {
      opts.undo();
      el.remove();
    });
    el.appendChild(u);
  }
  wrap.appendChild(el);
  setTimeout(() => el.remove(), opts.error ? 6000 : 4500);
}

// ---------- Helpers ----------
function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function formatDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function ageString(iso, future = false) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  const now = Date.now();
  const ms = future ? then - now : now - then;
  const sec = Math.max(0, Math.floor(ms / 1000));
  const min = Math.floor(sec / 60);
  const hr = Math.floor(min / 60);
  const day = Math.floor(hr / 24);
  const verb = future ? "in" : "";
  const suffix = future ? "" : "ago";
  if (day >= 1) return `${verb} ${day}d ${suffix}`.trim();
  if (hr >= 1) return `${verb} ${hr}h ${suffix}`.trim();
  if (min >= 1) return `${verb} ${min}m ${suffix}`.trim();
  return future ? "soon" : "just now";
}

function labelFor(s) {
  const parts = [];
  if (s.type) parts.push(s.type);
  if (s.project) parts.push(s.project);
  if (s.source) parts.push(`source: ${s.source}`);
  return parts.length ? parts.join(" · ") : "Recent first";
}

// ---------- Wiring ----------
function wireSearch() {
  const input = document.getElementById("q");
  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      state.q = input.value.trim();
      // Typing in search jumps out of Today into All memories so results have somewhere to render.
      if (state.q && state.scope === "today") {
        state.scope = "active";
        document.querySelectorAll("#scope button").forEach((b) => {
          b.classList.toggle("on", b.dataset.scope === "active");
        });
      }
      loadList();
    }, 120);
  });
  // Slash hotkey
  window.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    }
    if (e.key === "Escape" && document.activeElement === input) {
      input.value = "";
      state.q = "";
      input.blur();
      loadList();
    }
  });
}

function wireScope() {
  document.querySelectorAll("#scope button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#scope button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      state.scope = b.dataset.scope;
      loadList();
    });
  });
}

function wireDensity() {
  // Restore persisted choice on load.
  const stored = localStorage.getItem("poppy.density");
  if (stored === "compact" || stored === "comfortable") {
    state.density = stored;
    document.body.classList.toggle("compact", stored === "compact");
    document.querySelectorAll("#density button").forEach((b) => {
      b.classList.toggle("on", b.dataset.d === stored);
    });
  }
  document.querySelectorAll("#density button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#density button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      state.density = b.dataset.d;
      document.body.classList.toggle("compact", state.density === "compact");
      localStorage.setItem("poppy.density", state.density);
    });
  });
}

async function init() {
  wireSearch();
  wireScope();
  wireDensity();
  await Promise.all([loadFacets(), loadStats(), loadToday()]);
  await loadList();
}

init().catch((e) => {
  toast(`Failed to load: ${e.message}`, { error: true });
});
