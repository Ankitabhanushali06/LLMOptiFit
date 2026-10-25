"use strict";
const CACHE = "llmoptifit-shell-v1";
const ASSETS = ["/", "/index.html", "/style.css", "/app.js", "/privacy.html", "/manifest.webmanifest", "/icons/icon.svg", "/icons/icon-192.png", "/icons/icon-512.png", "/icons/apple-touch-icon.png"];
self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS)));
  // Wait for old tabs to close before replacing their cached app version.
});
self.addEventListener("activate", event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith("llmoptifit-shell-") && k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", event => {
  const request = event.request, url = new URL(request.url);
  // Only exact, same-origin public shell URLs enter the cache.
  if (request.method !== "GET" || url.origin !== self.location.origin || url.search || !ASSETS.includes(url.pathname) || request.headers.has("X-Nebius-API-Key")) return;
  event.respondWith(caches.open(CACHE).then(async cache => {
    const cached = await cache.match(url.pathname);
    return cached || fetch(request); // Never cache arbitrary responses or API traffic.
  }));
});
