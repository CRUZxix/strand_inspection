// Minimal app-shell cache: the UI loads offline, inspection itself needs the server.
const V = "strand-v1", SHELL = ["./", "index.html", "app.js", "manifest.webmanifest", "icon.svg"];
self.addEventListener("install", (e) => { e.waitUntil(caches.open(V).then((c) => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== V).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(fetch(e.request).then((r) => { const c = r.clone(); caches.open(V).then((ch) => ch.put(e.request, c)); return r; })
    .catch(() => caches.match(e.request)));     // network-first so updates show immediately
});
