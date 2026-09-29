/**
 * SkyGuard AI — Home Page JS
 * Health check, station bubbles interaction, smooth init.
 */

(function () {
  "use strict";

  // ── Health check ─────────────────────────────────────────────
  async function checkHealth() {
    const dot  = document.querySelector(".status-dot");
    const text = document.querySelector(".status-text");
    try {
      const res  = await fetch("/health");
      const data = await res.json();
      if (data.status === "ok") {
        dot.classList.add("online");
        text.textContent = data.artefacts_loaded ? "Model Ready" : "API Online";
      } else throw new Error("not ok");
    } catch {
      dot.classList.add("offline");
      text.textContent = "API Offline";
    }
  }

  // ── Station bubble hover ─────────────────────────────────────
  document.querySelectorAll(".station-bubble").forEach(bubble => {
    bubble.addEventListener("mouseenter", () => {
      bubble.style.zIndex = "20";
    });
    bubble.addEventListener("mouseleave", () => {
      bubble.style.zIndex = "";
    });
    bubble.addEventListener("click", () => {
      window.location.href = "/dashboard?station=" + bubble.dataset.station;
    });
  });

  // ── Intersection observer for scroll animations ───────────────
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = "1";
          e.target.style.transform = "translateY(0)";
        }
      });
    },
    { threshold: 0.15 }
  );

  document.querySelectorAll(".feature-card, .stat-card, .pipe-step").forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(30px)";
    el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
    observer.observe(el);
  });

  // ── Init ─────────────────────────────────────────────────────
  checkHealth();
  setInterval(checkHealth, 30000);
})();
