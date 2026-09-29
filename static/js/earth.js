/**
 * SkyGuard AI — Interactive Three.js Earth
 * Earth revolves toward mouse cursor position.
 * Features: atmosphere glow, stars, station particles, cloud layer.
 */

(function () {
  "use strict";

  const canvas = document.getElementById("earthCanvas");
  if (!canvas) return;

  // ── Scene Setup ─────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 0, 3.2);

  // ── Textures (procedural — no external files needed) ────────
  function makeProcTexture(size, drawFn) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = size;
    drawFn(cv.getContext("2d"), size);
    return new THREE.CanvasTexture(cv);
  }

  // Earth surface — blue oceans + green/brown landmasses
  const earthTex = makeProcTexture(1024, (ctx, sz) => {
    // Ocean gradient
    const grad = ctx.createLinearGradient(0, 0, sz, sz);
    grad.addColorStop(0,   "#0d2a4a");
    grad.addColorStop(0.4, "#0e3a5f");
    grad.addColorStop(1,   "#0a1a3a");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, sz, sz);

    // Landmasses (simplified blobs)
    const lands = [
      { x: 0.35, y: 0.30, rx: 0.12, ry: 0.10, rot: -0.3 }, // Europe/Africa-ish
      { x: 0.22, y: 0.35, rx: 0.08, ry: 0.14, rot: 0.1  }, // Americas-ish
      { x: 0.60, y: 0.32, rx: 0.14, ry: 0.11, rot: 0.2  }, // Asia-ish
      { x: 0.65, y: 0.55, rx: 0.08, ry: 0.06, rot: 0.4  }, // Australia-ish
      { x: 0.50, y: 0.70, rx: 0.06, ry: 0.04, rot: 0.0  }, // Antarctica
    ];

    lands.forEach(({ x, y, rx, ry, rot }) => {
      ctx.save();
      ctx.translate(x * sz, y * sz);
      ctx.rotate(rot);
      const lg = ctx.createRadialGradient(0, 0, 0, 0, 0, rx * sz);
      lg.addColorStop(0,   "#2a5c2a");
      lg.addColorStop(0.5, "#3d7a30");
      lg.addColorStop(0.8, "#8b6914");
      lg.addColorStop(1,   "transparent");
      ctx.fillStyle = lg;
      ctx.beginPath();
      ctx.ellipse(0, 0, rx * sz, ry * sz, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    // India highlight (approx position)
    ctx.save();
    ctx.translate(0.595 * sz, 0.385 * sz);
    const igr = ctx.createRadialGradient(0, 0, 0, 0, 0, 40);
    igr.addColorStop(0,   "rgba(56,189,248,0.6)");
    igr.addColorStop(0.4, "rgba(56,189,248,0.2)");
    igr.addColorStop(1,   "transparent");
    ctx.fillStyle = igr;
    ctx.beginPath();
    ctx.arc(0, 0, 40, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Ocean shimmer lines
    ctx.strokeStyle = "rgba(56,189,248,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 30; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * (sz / 30));
      ctx.lineTo(sz, i * (sz / 30) + (Math.random() - 0.5) * 30);
      ctx.stroke();
    }
  });

  // Cloud texture
  const cloudTex = makeProcTexture(512, (ctx, sz) => {
    ctx.clearRect(0, 0, sz, sz);
    for (let i = 0; i < 60; i++) {
      const x = Math.random() * sz;
      const y = Math.random() * sz;
      const r = 20 + Math.random() * 60;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0,   "rgba(255,255,255,0.45)");
      g.addColorStop(0.5, "rgba(255,255,255,0.15)");
      g.addColorStop(1,   "transparent");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  // ── Earth Sphere ─────────────────────────────────────────────
  const earthGeo  = new THREE.SphereGeometry(1, 64, 64);
  const earthMat  = new THREE.MeshPhongMaterial({
    map:          earthTex,
    shininess:    15,
    specular:     new THREE.Color(0x112244),
  });
  const earth = new THREE.Mesh(earthGeo, earthMat);
  scene.add(earth);

  // ── Cloud Layer ──────────────────────────────────────────────
  const cloudGeo = new THREE.SphereGeometry(1.012, 48, 48);
  const cloudMat = new THREE.MeshPhongMaterial({
    map:         cloudTex,
    transparent: true,
    opacity:     0.55,
    depthWrite:  false,
  });
  const clouds = new THREE.Mesh(cloudGeo, cloudMat);
  scene.add(clouds);

  // ── Atmosphere Glow ──────────────────────────────────────────
  const atmGeo = new THREE.SphereGeometry(1.15, 32, 32);
  const atmMat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      void main() {
        float intensity = pow(0.72 - dot(vNormal, vec3(0,0,1)), 3.0);
        gl_FragColor = vec4(0.18, 0.60, 0.95, 1.0) * intensity * 1.6;
      }
    `,
    blending:   THREE.AdditiveBlending,
    side:       THREE.BackSide,
    transparent: true,
  });
  const atmosphere = new THREE.Mesh(atmGeo, atmMat);
  scene.add(atmosphere);

  // ── Station Marker Particles ─────────────────────────────────
  const STATION_COORDS = [
    { name: "Delhi",   lat: 28.61, lon: 77.20 },
    { name: "Mumbai",  lat: 19.07, lon: 72.88 },
    { name: "Jodhpur", lat: 26.29, lon: 73.02 },
    { name: "Lucknow", lat: 26.85, lon: 80.94 },
    { name: "Kochi",   lat: 9.93,  lon: 76.27 },
    { name: "Manali",  lat: 32.27, lon: 77.17 },
  ];

  function latLonToVec3(lat, lon, r) {
    const phi   = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -r * Math.sin(phi) * Math.cos(theta),
       r * Math.cos(phi),
       r * Math.sin(phi) * Math.sin(theta)
    );
  }

  const stationGroup = new THREE.Group();
  STATION_COORDS.forEach(st => {
    const pos = latLonToVec3(st.lat, st.lon, 1.02);

    // Dot
    const dotGeo = new THREE.SphereGeometry(0.018, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const dot = new THREE.Mesh(dotGeo, dotMat);
    dot.position.copy(pos);

    // Pulse ring
    const ringGeo = new THREE.RingGeometry(0.025, 0.038, 16);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.6
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.copy(pos);
    ring.lookAt(pos.clone().multiplyScalar(2));

    stationGroup.add(dot);
    stationGroup.add(ring);
  });
  earth.add(stationGroup);

  // ── Stars Background ─────────────────────────────────────────
  const starCount = 2000;
  const starPositions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = 80 + Math.random() * 100;
    starPositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPositions[i * 3 + 2] = r * Math.cos(phi);
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.25, sizeAttenuation: true });
  scene.add(new THREE.Points(starGeo, starMat));

  // ── Lighting ─────────────────────────────────────────────────
  const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
  sunLight.position.set(5, 3, 5);
  scene.add(sunLight);

  const ambientLight = new THREE.AmbientLight(0x112244, 0.8);
  scene.add(ambientLight);

  const rimLight = new THREE.DirectionalLight(0x1a6090, 0.4);
  rimLight.position.set(-5, 0, -3);
  scene.add(rimLight);

  // ── Mouse Tracking ───────────────────────────────────────────
  let mouseX = 0;
  let mouseY = 0;
  let targetRotX = 0;
  let targetRotY = 0;

  document.addEventListener("mousemove", (e) => {
    mouseX = (e.clientX / window.innerWidth  - 0.5) * 2;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  document.addEventListener("touchmove", (e) => {
    if (e.touches.length > 0) {
      mouseX = (e.touches[0].clientX / window.innerWidth  - 0.5) * 2;
      mouseY = (e.touches[0].clientY / window.innerHeight - 0.5) * 2;
    }
  }, { passive: true });

  // ── Resize Handler ───────────────────────────────────────────
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ── Animation Loop ───────────────────────────────────────────
  const clock = new THREE.Clock();
  let baseRotY = 0;

  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime();

    // Cursor-driven rotation target
    targetRotY = mouseX * Math.PI * 0.5;
    targetRotX = -mouseY * Math.PI * 0.25;

    // Smooth lerp toward cursor
    earth.rotation.y += (baseRotY + targetRotY - earth.rotation.y) * 0.04;
    earth.rotation.x += (targetRotX - earth.rotation.x) * 0.04;

    // Slow auto-spin
    baseRotY += delta * 0.06;

    // Cloud drift (slightly faster than earth)
    clouds.rotation.y += delta * 0.07;

    // Pulse station rings
    stationGroup.children.forEach((child, i) => {
      if (i % 2 === 1) { // ring meshes
        const scale = 1 + 0.4 * Math.sin(elapsed * 2 + i);
        child.material.opacity = 0.5 + 0.3 * Math.cos(elapsed * 2 + i);
        child.scale.setScalar(scale);
      }
    });

    // Atmosphere shimmer
    atmosphere.material.uniforms && Object.keys(atmosphere.material.uniforms).length;

    renderer.render(scene, camera);
  }

  animate();
})();
