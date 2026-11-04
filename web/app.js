"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const video = $("cam"), hud = $("hud"), freezeCv = $("freeze"), hctx = hud.getContext("2d");
  const CAP_LONG_SIDE = 800, TARGET_FPS = 8, JPEG_Q = 0.7, MAX_INFLIGHT = 2;
  const COLORS = { GOOD: "#2fd35c", DEFECT: "#ff4538", UNCERTAIN: "#ffc21a", ALIGNING: "#ffc21a" };

  const S = { ws: null, open: false, seq: 0, sent: new Map(), inflight: 0, frozen: false, snapPending: false,
              last: null, lastVerdict: null, track: null, audio: null, rtt: [], backoff: 500, timer: null,
              cap: document.createElement("canvas") };

  // ---------- camera ----------
  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    await video.play();
    S.track = stream.getVideoTracks()[0];
    const caps = (S.track.getCapabilities && S.track.getCapabilities()) || {};
    $("torch").disabled = !caps.torch;
    if (!caps.torch) $("torch").title = "Torch not supported on this device/browser";
  }

  $("torch").onclick = async () => {
    const on = $("torch").getAttribute("aria-pressed") !== "true";
    try {
      await S.track.applyConstraints({ advanced: [{ torch: on }] });
      $("torch").setAttribute("aria-pressed", String(on));
    } catch (e) { setMsg("Torch failed: " + e.message); }
  };

  // ---------- websocket ----------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/inspect`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { S.ws = ws; S.open = true; S.backoff = 500; S.inflight = 0; S.sent.clear(); setConn(true); };
    ws.onclose = () => {
      S.open = false; setConn(false); setHint("Reconnecting…");
      setTimeout(connect, S.backoff); S.backoff = Math.min(S.backoff * 2, 5000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
  }

  function onMessage(m) {
    if (m.type === "result" || m.type === "snapshot") {
      const t0 = S.sent.get(m.seq);
      if (t0 !== undefined) { S.sent.delete(m.seq); S.rtt.push(performance.now() - t0); if (S.rtt.length > 20) S.rtt.shift(); }
      S.inflight = Math.max(0, S.inflight - 1 - (m.dropped || 0));
      if (m.type === "snapshot") { S.snapPending = false; }
      if (S.frozen && m.type !== "snapshot") return;     // frozen: ignore live results
      S.last = m;
      render(m);
      feedback(m);
    } else if (m.type === "error") {
      S.inflight = Math.max(0, S.inflight - 1);
      if (S.snapPending) { S.snapPending = false; unfreeze(); }
      setMsg(m.error);
    } else if (m.type === "reset") {
      S.last = null; S.lastVerdict = null; render(null);
    }
  }

  // ---------- capture loop ----------
  function grabFrame() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return null;
    const k = CAP_LONG_SIDE / Math.max(vw, vh);
    const c = S.cap; c.width = Math.round(vw * k); c.height = Math.round(vh * k);
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    return c;
  }

  function sendFrame(isSnapshot) {
    const c = grabFrame();
    if (!c) return;
    const seq = S.seq = (S.seq + 1) >>> 0;
    if (isSnapshot) {                       // keep the exact frame the server will inspect
      freezeCv.width = c.width; freezeCv.height = c.height;
      freezeCv.getContext("2d").drawImage(c, 0, 0);
      S.frozenSize = [c.width, c.height];
    }
    c.toBlob(async (blob) => {
      if (!blob || !S.open) return;
      const body = new Uint8Array(4 + blob.size);
      new DataView(body.buffer).setUint32(0, seq);
      body.set(new Uint8Array(await blob.arrayBuffer()), 4);
      if (isSnapshot) S.ws.send(JSON.stringify({ cmd: "snapshot" }));
      S.sent.set(seq, performance.now()); S.inflight++;
      S.ws.send(body);
    }, "image/jpeg", JPEG_Q);
  }

  function loop() {
    if (S.open && !S.frozen && S.inflight < MAX_INFLIGHT && !document.hidden) sendFrame(false);
    S.timer = setTimeout(loop, 1000 / TARGET_FPS);
  }

  // ---------- rendering ----------
  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    hud.width = Math.round(innerWidth * dpr); hud.height = Math.round(innerHeight * dpr);
    hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (S.last) drawBoxes(S.last);
  }

  // object-fit: contain (letterbox) mapping from frame pixel coords to screen coords
  function mapper(fw, fh) {
    const s = Math.min(innerWidth / fw, innerHeight / fh);
    return { s, ox: (innerWidth - fw * s) / 2, oy: (innerHeight - fh * s) / 2 };
  }

  function drawBoxes(m) {
    hctx.clearRect(0, 0, innerWidth, innerHeight);
    if (!m || !m.boxes || !m.boxes.length) return;
    const [fw, fh] = m.frame_size || S.frozenSize || [video.videoWidth, video.videoHeight];
    const { s, ox, oy } = mapper(fw, fh);
    hctx.lineWidth = 3; hctx.font = "700 15px system-ui, sans-serif"; hctx.textBaseline = "top";
    for (const b of m.boxes) {
      const [x1, y1, x2, y2] = b.coords, col = COLORS[b.status] || COLORS.ALIGNING;
      const x = ox + x1 * s, y = oy + y1 * s, w = (x2 - x1) * s, h = (y2 - y1) * s;
      hctx.strokeStyle = col; hctx.fillStyle = col + "22"; hctx.fillRect(x, y, w, h); hctx.strokeRect(x, y, w, h);
      const label = b.count === undefined ? b.id : `${b.id} · ${b.count}`;
      const tw = hctx.measureText(label).width + 12;
      hctx.fillStyle = col; hctx.fillRect(x, y, tw, 22);
      hctx.fillStyle = "#06090c"; hctx.fillText(label, x + 6, y + 3);
    }
  }

  function render(m) {
    drawBoxes(m);
    const banner = $("banner");
    const v = m && m.packet_verdict;
    banner.className = v || ""; banner.textContent = v === "UNCERTAIN" ? "UNCERTAIN – RE-SCAN" : (v || "");
    if (m) setHint(m.hint || m.state);
    else setHint("Point camera at packet");
    const med = S.rtt.length ? Math.round([...S.rtt].sort((a, b) => a - b)[S.rtt.length >> 1]) : null;
    $("stats").textContent = med === null ? "– ms" : `${med} ms`;
    renderMatrix(m);
  }

  function renderMatrix(m) {
    const by = {};
    if (m && m.boxes) for (const b of m.boxes) by[b.id] = b;
    let html = "";
    for (let r = 1; r <= 5; r++) for (let c = 1; c <= 2; c++) {
      const id = `R${r}C${c}`, b = by[id];
      const st = b ? b.status : "";
      html += `<div class="cell ${st}"><div><small>${id}</small><b>${b && b.count !== undefined ? b.count : "–"}</b></div>` +
              `<small>${b && b.conf !== undefined ? Math.round(b.conf * 100) + "%" : ""}<br>${st}</small></div>`;
    }
    $("matrix").innerHTML = html;
  }

  function setHint(t) { $("hint").textContent = t; }
  function setConn(on) { $("dot").classList.toggle("on", on); }
  function setMsg(t) { $("msg").textContent = t || ""; if (t) { document.body.classList.add("open"); $("grab").setAttribute("aria-expanded", "true"); } }

  // ---------- haptics + audio ----------
  function beep(freqs, dur = 0.14) {
    if (!S.audio) return;
    const t = S.audio.currentTime;
    freqs.forEach((f, i) => {
      const o = S.audio.createOscillator(), g = S.audio.createGain();
      o.frequency.value = f; o.type = "sine"; o.connect(g); g.connect(S.audio.destination);
      g.gain.setValueAtTime(0.0001, t + i * dur); g.gain.exponentialRampToValueAtTime(0.35, t + i * dur + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + (i + 1) * dur);
      o.start(t + i * dur); o.stop(t + (i + 1) * dur + 0.02);
    });
  }
  function feedback(m) {
    const v = (m.state === "LATCHED" || m.state === "SNAPSHOT") ? m.packet_verdict : null;
    if (v === S.lastVerdict) return;
    S.lastVerdict = v;                                     // fires once per new verdict, not every frame
    if (!v) return;
    const vib = navigator.vibrate ? (p) => navigator.vibrate(p) : () => {};
    if (v === "PASS") { vib(80); beep([880, 1320]); }
    else if (v === "FAIL") { vib([100, 50, 100]); beep([440, 330, 440], 0.16); }
    else { vib([250]); beep([330], 0.3); }
  }

  // ---------- controls ----------
  function unfreeze() { S.frozen = false; S.snapPending = false; document.body.classList.remove("frozen"); $("shutter").textContent = "📷 Freeze"; }
  $("shutter").onclick = () => {
    if (S.frozen) { unfreeze(); S.lastVerdict = null; render(S.last = null); return; }
    if (!S.open) return setMsg("Not connected");
    S.frozen = true; S.snapPending = true;
    document.body.classList.add("frozen"); $("shutter").textContent = "▶ Resume";
    setHint("Inspecting snapshot…"); sendFrame(true);
  };
  $("reset").onclick = () => {
    if (S.frozen) unfreeze();
    if (S.open) S.ws.send(JSON.stringify({ cmd: "reset" }));
    S.lastVerdict = null; setMsg("");
  };
  $("grab").onclick = () => {
    const open = document.body.classList.toggle("open");
    $("grab").setAttribute("aria-expanded", String(open));
  };

  $("go").onclick = async () => {
    try {
      S.audio = new (window.AudioContext || window.webkitAudioContext)();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
        throw new Error("Camera needs a secure (https) connection.");
      await startCamera();
      $("start").style.display = "none";
      fitCanvas(); renderMatrix(null); connect(); loop();
    } catch (e) {
      $("err").textContent = e.name === "NotAllowedError" ? "Camera permission denied. Allow it in browser settings." : e.message;
    }
  };

  addEventListener("resize", fitCanvas);
  addEventListener("orientationchange", () => setTimeout(fitCanvas, 200));
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
  window.__strand = S;    // debugging / automated tests
})();
