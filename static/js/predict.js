/**
 * SkyGuard AI — ML Predict Page JS
 * Handles form submission, slider sync, presets,
 * batch predictions, gauge + feature chart rendering.
 */

(function () {
  "use strict";

  Chart.defaults.color       = "#94a3b8";
  Chart.defaults.borderColor = "rgba(56,189,248,0.1)";

  const $ = id => document.getElementById(id);

  let gaugeChart   = null;
  let featuresChart = null;

  // ── Health ─────────────────────────────────────────────────────
  async function checkHealth() {
    const dot    = document.querySelector(".status-dot");
    const text   = document.querySelector(".status-text");
    const banner = $("modelBanner");
    const bIcon  = $("modelBannerIcon");
    const bText  = $("modelBannerText");

    try {
      const res  = await fetch("/health");
      const data = await res.json();
      if (data.status === "ok") {
        dot.className    = "status-dot online";
        text.textContent = data.artefacts_loaded ? "Model Ready" : "API Online";
        if (data.artefacts_loaded) {
          banner.classList.add("online");
          bIcon.textContent = "✅";
          bText.textContent = "Model artefacts loaded and ready for inference.";
        } else {
          bIcon.textContent = "⚠️";
          bText.textContent = "API online but model artefacts not loaded. Run POST /train first.";
        }
      } else throw new Error();
    } catch {
      dot.className    = "status-dot offline";
      text.textContent = "API Offline";
      banner.classList.add("offline");
      bIcon.textContent = "❌";
      bText.textContent = "Cannot reach SkyGuard AI API. Make sure Flask is running.";
    }
  }

  // ── Slider ↔ Number sync ─────────────────────────────────────
  function linkSlider(sliderId, numId) {
    const slider = $(sliderId);
    const num    = $(numId);
    if (!slider || !num) return;

    function updateSliderTrack() {
      const min = +slider.min, max = +slider.max, val = +slider.value;
      const pct = ((val - min) / (max - min)) * 100;
      slider.style.background = `linear-gradient(to right, var(--accent) ${pct}%, var(--bg-card) ${pct}%)`;
    }

    slider.addEventListener("input", () => {
      num.value = slider.value;
      updateSliderTrack();
    });
    num.addEventListener("input", () => {
      slider.value = num.value;
      updateSliderTrack();
    });
    updateSliderTrack();
  }

  linkSlider("fTempSlider",  "fTemp");
  linkSlider("fHumSlider",   "fHum");
  linkSlider("fPresSlider",  "fPres");

  // ── Preset buttons ────────────────────────────────────────────
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      $("fTemp").value = btn.dataset.temp;
      $("fHum").value  = btn.dataset.hum;
      $("fPres").value = btn.dataset.pres;
      ["fTempSlider", "fHumSlider", "fPresSlider"].forEach((sid, i) => {
        const vals = [btn.dataset.temp, btn.dataset.hum, btn.dataset.pres];
        $(sid).value = vals[i];
        $(sid).dispatchEvent(new Event("input"));
      });
    });
  });

  // ── Set default datetime ──────────────────────────────────────
  const now = new Date();
  now.setMinutes(0, 0, 0);
  $("fTime").value = now.toISOString().slice(0, 16);

  // ── Build payload from form ───────────────────────────────────
  function buildPayload() {
    return [{
      station:              $("fStation").value,
      time:                 $("fTime").value,
      temperature_2m:       +$("fTemp").value,
      relative_humidity_2m: +$("fHum").value,
      surface_pressure:     +$("fPres").value,
    }];
  }

  // ── API call ──────────────────────────────────────────────────
  async function runPredict(payload) {
    const res = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Prediction failed");
    }
    return await res.json();
  }

  // ── Gauge chart (semi-circle) ─────────────────────────────────
  function buildGauge(score) {
    const ctx   = $("gaugeChart")?.getContext("2d");
    if (!ctx) return;

    const SEV_COLOR = score < 0.35 ? "#34d399" : score < 0.55 ? "#818cf8" : score < 0.75 ? "#fbbf24" : "#f87171";

    if (gaugeChart) gaugeChart.destroy();
    gaugeChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        datasets: [{
          data: [score * 100, (1 - score) * 100],
          backgroundColor: [SEV_COLOR, "rgba(255,255,255,0.05)"],
          borderWidth: 0,
          circumference: 180,
          rotation: 270,
        }],
      },
      options: {
        responsive: false,
        cutout: "70%",
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });

    $("gaugeScore").textContent = score.toFixed(3);
    $("gaugeScore").style.color = SEV_COLOR;
  }

  // ── Feature contributions chart ───────────────────────────────
  function buildFeaturesChart(features) {
    const section = $("contributionsSection");
    const list    = $("contributionsList");
    if (!features || features.length === 0) {
      section.style.display = "none";
      return;
    }
    section.style.display = "block";

    const labels = features.map(f => f.feature);
    const vals   = features.map(f => Math.abs(f.impact));
    const maxVal = Math.max(...vals, 0.001);

    // Bar chart
    const ctx = $("featuresChart")?.getContext("2d");
    if (!ctx) return;

    if (featuresChart) featuresChart.destroy();
    featuresChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Feature Impact",
          data:  vals,
          backgroundColor: vals.map(v => `rgba(56,189,248,${0.3 + 0.7 * (v / maxVal)})`),
          borderColor: "#38bdf8",
          borderWidth: 1.5,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#64748b" }, grid: { color: "rgba(56,189,248,0.05)" } },
          y: { ticks: { color: "#94a3b8", font: { size: 11 } } },
        },
      },
    });

    // Contribution list
    list.innerHTML = features.map(f => {
      const pct = (Math.abs(f.impact) / maxVal * 100).toFixed(0);
      return `
        <div class="contrib-item">
          <span class="contrib-feature">${f.feature}</span>
          <div class="contrib-bar-wrap">
            <div class="contrib-bar" style="width:${pct}%"></div>
          </div>
          <span class="contrib-val">${f.impact > 0 ? "+" : ""}${(+f.impact).toFixed(4)}</span>
        </div>
      `;
    }).join("");
  }

  // ── Display single result ─────────────────────────────────────
  function displayResult(pred) {
    $("resultsPlaceholder").style.display = "none";
    $("batchResults").style.display       = "none";
    $("resultContent").style.display      = "flex";
    $("resultContent").style.flexDirection = "column";

    const sev = pred.severity || "NORMAL";
    const badge = $("severityBadge");
    badge.textContent = sev;
    badge.className   = "severity-badge " + sev;

    $("severityHeader").style.borderColor = {
      NORMAL: "rgba(52,211,153,0.3)", ELEVATED: "rgba(129,140,248,0.3)",
      WARNING: "rgba(251,191,36,0.3)", CRITICAL: "rgba(248,113,113,0.4)"
    }[sev] || "var(--border)";

    $("resultStation").textContent = (pred.station || "").replace("_2025", "").toUpperCase();
    $("resultTime").textContent    = pred.time ? new Date(pred.time).toLocaleString() : "—";

    buildGauge(pred.anomaly_score || 0);

    $("rTemp").textContent = pred.temperature_2m?.toFixed(1) + "°C";
    $("rHum").textContent  = pred.relative_humidity_2m?.toFixed(0) + "%";
    $("rPres").textContent = pred.surface_pressure?.toFixed(1);

    $("anomalyType").textContent = pred.anomaly_type || "—";
    $("anomalyReason").textContent = pred.reason || "—";
    $("anomalyDetails").style.borderColor = pred.is_anomaly
      ? "rgba(248,113,113,0.3)" : "var(--border)";

    buildFeaturesChart(pred.top_contributing_features || []);
  }

  // ── Single predict ────────────────────────────────────────────
  $("predictForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn  = $("predictBtn");
    const text = $("predictBtnText");
    btn.disabled   = true;
    text.textContent = "⏳ Running…";

    try {
      const payload = buildPayload();
      const data    = await runPredict(payload);
      const pred    = data.predictions?.[0];
      if (pred) displayResult(pred);
      else throw new Error("No prediction returned");
    } catch (err) {
      alert("❌ Prediction error: " + err.message);
    } finally {
      btn.disabled = false;
      text.textContent = "🚀 Run Anomaly Detection";
    }
  });

  // ── Batch predict ─────────────────────────────────────────────
  $("batchBtn")?.addEventListener("click", async () => {
    const raw = $("batchInput")?.value?.trim();
    let payload;
    try {
      payload = raw ? JSON.parse(raw) : buildDemoPayload();
    } catch {
      alert("❌ Invalid JSON in batch input.");
      return;
    }

    const btn = $("batchBtn");
    btn.disabled = true;
    btn.textContent = "⏳ Running batch…";

    try {
      const data = await runPredict(Array.isArray(payload) ? payload : [payload]);
      displayBatchResults(data.predictions || []);
    } catch (err) {
      alert("❌ Batch error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "📦 Run Batch Prediction";
    }
  });

  function buildDemoPayload() {
    return [
      { station: "delhi_2025",   time: "2026-09-01T12:00", temperature_2m: 28.5,  relative_humidity_2m: 62,  surface_pressure: 988  },
      { station: "delhi_2025",   time: "2026-09-01T13:00", temperature_2m: 49.8,  relative_humidity_2m: 12,  surface_pressure: 988  },
      { station: "jodhpur_2025", time: "2026-09-01T14:00", temperature_2m: 38.0,  relative_humidity_2m: 99,  surface_pressure: 820  },
    ];
  }

  const SEV_COLORS = {
    NORMAL:   { bg: "rgba(52,211,153,0.15)",  fg: "#34d399" },
    ELEVATED: { bg: "rgba(129,140,248,0.15)", fg: "#818cf8" },
    WARNING:  { bg: "rgba(251,191,36,0.15)",  fg: "#fbbf24" },
    CRITICAL: { bg: "rgba(248,113,113,0.15)", fg: "#f87171" },
  };

  function displayBatchResults(predictions) {
    $("resultsPlaceholder").style.display = "none";
    $("resultContent").style.display      = "none";
    $("batchResults").style.display       = "block";

    const tbody = $("batchTableBody");
    tbody.innerHTML = predictions.map(p => {
      const sev   = p.severity || "NORMAL";
      const sc    = SEV_COLORS[sev] || SEV_COLORS.NORMAL;
      return `
        <tr>
          <td style="color:var(--accent)">${(p.station || "").replace("_2025","")}</td>
          <td>${p.time ? p.time.slice(0,16) : "—"}</td>
          <td>${(+p.temperature_2m).toFixed(1)}</td>
          <td>${(+p.relative_humidity_2m).toFixed(0)}</td>
          <td>${(+p.surface_pressure).toFixed(1)}</td>
          <td style="font-family:var(--font-display)">${(+p.anomaly_score).toFixed(3)}</td>
          <td>
            <span class="tbl-badge" style="background:${sc.bg};color:${sc.fg}">${sev}</span>
          </td>
          <td style="color:var(--text-muted);font-size:0.78rem">${p.anomaly_type || "—"}</td>
        </tr>
      `;
    }).join("");
  }

  // ── Init ─────────────────────────────────────────────────────
  checkHealth();
  setInterval(checkHealth, 30_000);
})();
