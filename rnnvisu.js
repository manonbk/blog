// -------------------------------
  // 3D Neural network visualizer (updated to load MATLAB .mat)
  // - Exicitatory neurons (first n_E) are green -> brighten to white on activation
  // - Inhibitory neurons (last n_I) are red -> brighten to white on activation
  // - Accepts MATLAB .mat saved with: save('my_network.mat', 'M.W', 'M.n_t', 'M.is_AP', 'M.n_E', 'M.n_I')
  // -------------------------------

  // CONFIG (will be overwritten when loading data)
  let N = 500;          // number of neurons (example) - mutable
  let LAYOUT_ITER = 400; // layout iterations

  // internal data holders
  let M = null; // Float32Array flat n*n
  let A = null; // array length T of Float32Array(n)
  let n_E = 0, n_I = 0;

  function initNNVis(selector="#nnvis-container"){
    window.nnvisContainer = selector;
  }



  // helper: make sample data (keeps behavior if no file loaded)
  function makeSampleData(n, t){
    const M = new Float32Array(n*n);
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) M[i*n+j] = (i===j)?0:Math.random()*0.02 + ((Math.abs(i-j) < 5)?0.4:0);
    const A = new Array(t);
    for(let tt=0; tt<t; tt++){
      const arr = new Float32Array(n);
      const center = Math.floor((tt/t)*n);
      for(let i=0;i<n;i++){ const d = Math.abs(i-center); arr[i] = Math.max(0, Math.exp(-(d*d)/(2*200)) + (Math.random()*0.1 -0.05)); if(arr[i]>1) arr[i]=1; }
      A[tt]=arr;
    }
    return {M,A,n_E:Math.floor(n*0.8), n_I:Math.floor(n*0.2)};
  }

  // compute layout (force-directed). returns Float32Array(n*3)
  function computeLayout(n, Mflat, iterations){
    console.log(Mflat)
    const pos = new Float32Array(n*3);
    for(let i=0;i<n;i++){ pos[3*i] = (Math.random()-0.5)*200; pos[3*i+1] = (Math.random()-0.5)*200; pos[3*i+2] = (Math.random()-0.5)*200; }
    const degree = new Float32Array(n);
    for(let i=0;i<n;i++){ let s=0; for(let j=0;j<n;j++) s += (Mflat[i*n+j]||0) + (Mflat[j*n+i]||0); degree[i]=s; }
    let maxd = 0; for(let i=0;i<n;i++) if(degree[i]>maxd) maxd=degree[i]; if(maxd==0) maxd=1; for(let i=0;i<n;i++) degree[i]/=maxd;
    const epsilon=1e-4, dt=0.02;
    for(let it=0; it<iterations; it++){
      const fx = new Float32Array(n), fy = new Float32Array(n), fz = new Float32Array(n);
      for(let i=0;i<n;i++){
        const xi=pos[3*i], yi=pos[3*i+1], zi=pos[3*i+2];
        for(let j=i+1;j<n;j++){
          const w = (Mflat[i*n + j]||0) + (Mflat[j*n + i]||0);
          if(w <= 1e-7) continue;
          const xj=pos[3*j], yj=pos[3*j+1], zj=pos[3*j+2];
          let dx = xj-xi, dy = yj-yi, dz = zj-zi;
          let dist = Math.sqrt(dx*dx + dy*dy + dz*dz) + epsilon;
          const pref = 20 * Math.pow(1/Math.max(w,1e-6), 0.2);
          const strength = 100 * Math.min(0.0001, w*5);
          const f = (dist - pref) * strength;
          const nx = (dx/dist)*f, ny=(dy/dist)*f, nz=(dz/dist)*f;
          fx[i] += nx; fy[i] += ny; fz[i] += nz; fx[j] -= nx; fy[j] -= ny; fz[j] -= nz;
        }
      }
      for(let i=0;i<n;i++){
        pos[3*i] += (-pos[3*i])*0.001*degree[i] + fx[i]*dt;
        pos[3*i+1] += (-pos[3*i+1])*0.001*degree[i] + fy[i]*dt;
        pos[3*i+2] += (-pos[3*i+2])*0.001*degree[i] + fz[i]*dt;
      }
    }
    return pos;
  }

  // --- three.js scene setup
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 2000);
  camera.position.set(0,0,500);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  const container = document.querySelector(window.nnvisContainer || "body");
  container.appendChild(renderer.domElement);
  renderer.setClearColor(0x252525, 1);

  // geometry/material holders
  let points = null;
  let edgeLines = null;
  let baseColors = null; // Float32Array n*3 storing base color per neuron (green or red)

  function buildSceneFromData(n, Mflat, Aarr, nE, nI){
    console.log("buildSceneFromData");
    // update globals
    N = n; M = Mflat; A = Aarr; n_E = nE; n_I = nI;
    // compute layout
    console.log('Computing layout for n=',n, ' iterations=', LAYOUT_ITER);
    const positions = computeLayout(n, Mflat, LAYOUT_ITER);

    // build buffers
    const posAttr = new Float32Array(n*3);
    const colorAttr = new Float32Array(n*3);
    baseColors = new Float32Array(n*3);
    const sizeAttr = new Float32Array(n);

    for(let i=0;i<n;i++){
      posAttr[3*i] = positions[3*i]; posAttr[3*i+1] = positions[3*i+1]; posAttr[3*i+2] = positions[3*i+2];
      const isExc = i < nE;
      const base = isExc ? [44/255, 135/255, 138/255] : [191/255, 143/255, 189/255];
      baseColors[3*i] = base[0]; baseColors[3*i+1] = base[1]; baseColors[3*i+2] = base[2];
      // initial visible color = base
      colorAttr[3*i] = base[0]; colorAttr[3*i+1] = base[1]; colorAttr[3*i+2] = base[2];
      sizeAttr[i] = 5.0 + (Math.random()*2-1);
    }

    // remove old if exist
    if(points){ scene.remove(points); points.geometry.dispose(); }
    if(edgeLines){ scene.remove(edgeLines); edgeLines.geometry.dispose(); }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(posAttr, 3));
    geometry.setAttribute('customColor', new THREE.BufferAttribute(colorAttr,3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizeAttr,1));

    const material = new THREE.ShaderMaterial({
      uniforms: {pointScale: {value: window.innerHeight/2}},
      vertexShader: `attribute float size; attribute vec3 customColor; varying vec3 vColor; uniform float pointScale; void main(){ vColor = customColor; vec4 mvPosition = modelViewMatrix * vec4(position,1.0); gl_PointSize = size * (pointScale / -mvPosition.z); gl_Position = projectionMatrix * mvPosition; }`,
      fragmentShader: `varying vec3 vColor; void main(){ float d = length(gl_PointCoord - vec2(0.5)); if(d>0.5) discard; gl_FragColor = vec4(vColor,1.0); }`,
      transparent: false
    });

    points = new THREE.Points(geometry, material);
    scene.add(points);

    // build edges with current threshold
    buildEdges(parseFloat(document.getElementById('edgeThreshold').value));

    // reset activations
    for(let i=0;i<N;i++){ currentActivation[i]=0; targetActivation[i]=0; }
    document.getElementById('time').max = A.length-1; document.getElementById('time').value = 0;
  }

  function buildEdges(thresh){
    if(!M || N===0) return;
    if(edgeLines){ scene.remove(edgeLines); edgeLines.geometry.dispose(); edgeLines = null; }
    const edges = [];
    for(let i=0;i<N;i++){
      for(let j=i+1;j<N;j++){
        const w = (M[i*N + j]||0) + (M[j*N + i]||0);
        if(w >= thresh){
          edges.push(positions_for_line(i)); edges.push(positions_for_line(j));
        }
      }
    }
    if(edges.length===0) return;
    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(edges.flat()),3));
    const edgeMat = new THREE.LineBasicMaterial({color:0xAAAAAA, linewidth:1, transparent:true, opacity:0.15});
    edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
    scene.add(edgeLines);

    function positions_for_line(idx){
      const p = points.geometry.attributes.position.array;
      return [p[3*idx], p[3*idx+1], p[3*idx+2]];
    }
  }

  // Activation smoothing arrays
  let currentActivation = new Float32Array(N);
  let targetActivation = new Float32Array(N);

  // default sample load
  (function(){ const s = makeSampleData(200, 100); const flat = s.M; const Aarr = s.A; buildSceneFromData(200, flat, Aarr, s.n_E, s.n_I); })();

  // UI elements
  const timeSlider = document.getElementById('time');
  const speedControl = document.getElementById('speed');
  const playBtn = document.getElementById('play');
  const pauseBtn = document.getElementById('pause');
  const edgeSlider = document.getElementById('edgeThreshold');
  const matfile = document.getElementById('matfile');

  let playing = false; let timeIndex = 0;

  // Ajouter les listeners seulement si les boutons existent
    if (playBtn != null) {
        playBtn.addEventListener('click', () => {
            isPlaying = true;
        });
    }

    if (pauseBtn != null) {
        pauseBtn.addEventListener('click', () => {
            isPlaying = false;
        });
    }
  timeSlider.addEventListener('input', ()=>{ timeIndex = parseInt(timeSlider.value); updateTargets(); });
  edgeSlider.addEventListener('input', ()=>{ buildEdges(parseFloat(edgeSlider.value)); });

  const jsonfile = document.getElementById('jsonfile');

  jsonfile.addEventListener('change', async (ev)=>{
    const f = ev.target.files[0];
    if(!f) return;

    try{
      await loadJSON(f);
    }catch(e){
      alert('Erreur lecture JSON: ' + e.message);
      console.error(e);
    }
  });


  async function loadJSON(file){
    const text = await file.text();
    const obj = JSON.parse(text);

    // expected keys: W, is_AP, n_E, n_I
    const MW = obj.W;
    const isAP = obj.A;

    if(!MW || !isAP){
      throw new Error("JSON must contain fields: W (NxN), is_AP (NxT)");
    }

    const n = MW.length;
    const flat = new Float32Array(n*n);

    for(let i=0;i<n;i++){
      for(let j=0;j<n;j++){
        flat[i*n+j] = Number(MW[i][j] || 0);
      }
    }

    const T = isAP[0].length;
    const Aarr = [];

    for(let t=0;t<T;t++){
      const frame = new Float32Array(n);
      for(let i=0;i<n;i++){
        frame[i] = Number(isAP[i][t] || 0);
      }
      Aarr.push(frame);
    }

    const nE = obj.n_E ?? Math.floor(n*0.8);
    const nI = obj.n_I ?? (n - nE);

    buildSceneFromData(n, flat, Aarr, nE, nI);

    camera.position.set(0,0, Math.max(300, n*0.2));
  }


  // Expose setData for console usage
  window.nnvis = {
    setData: function(obj){
      const flat = (obj.M instanceof Float32Array)? obj.M : new Float32Array(obj.M);
      const Aarr = obj.A.map(a=> (a instanceof Float32Array)? a : new Float32Array(a));
      const ne = obj.n_E || obj.nE || 0; const ni = obj.n_I || obj.nI || 0;
      const n = Math.sqrt(flat.length)|0;
      buildSceneFromData(n, flat, Aarr, ne>0?ne:Math.floor(n*0.8), ni>0?ni:(n - (ne>0?ne:Math.floor(n*0.8))));
    }
  };

  // updateTargets: copy A[timeIndex] into targetActivation (and resize arrays if needed)
  function updateTargets(){ if(!A) return; const arr = A[Math.max(0, Math.min(A.length-1, timeIndex))]; if(arr.length !== targetActivation.length){ targetActivation = new Float32Array(arr.length); currentActivation = new Float32Array(arr.length); }
    for(let i=0;i<arr.length;i++) targetActivation[i] = arr[i]; }

  // Animation loop: smooth activation and color blending between base color and white
  let last = performance.now();
  function animate(now){
    requestAnimationFrame(animate);
    const dt = (now - last)/1000; last = now;
    if(playing && A){ const s = parseFloat(speedControl.value); const advance = s * dt * 10; timeIndex = Math.min(A.length-1, Math.floor(timeIndex + advance)); if(timeIndex >= A.length-1) playing=false; document.getElementById('time').value = timeIndex; updateTargets(); }

    const lerpK = 6.0 * dt;
    if(points){
      const colors = points.geometry.attributes.customColor.array;
      const sizes = points.geometry.attributes.size.array;
      const base = baseColors;
      const n = Math.min(N, currentActivation.length);
      for(let i=0;i<n;i++){
        const c = currentActivation[i] || 0; const t = targetActivation[i] || 0;
        const nc = c + (t - c) * Math.min(1, lerpK);
        currentActivation[i] = nc;
        // blend base -> white by nc
        colors[3*i]   = base[3*i]   * (1-nc) + 1.0 * nc;
        colors[3*i+1] = base[3*i+1] * (1-nc) + 1.0 * nc;
        colors[3*i+2] = base[3*i+2] * (1-nc) + 1.0 * nc;
        sizes[i] = 5.0 + 8.0*nc;
      }
      points.geometry.attributes.customColor.needsUpdate = true;
      points.geometry.attributes.size.needsUpdate = true;
    }

    // rotate
    scene.rotation.y = rotY; scene.rotation.x = rotX;
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);

  // simple drag/zoom controls
  let isDragging=false, prevX=0, prevY=0, rotX=0, rotY=0;
  renderer.domElement.addEventListener('pointerdown', e=>{isDragging=true; prevX=e.clientX; prevY=e.clientY;});
  window.addEventListener('pointerup', ()=> isDragging=false);
  window.addEventListener('pointermove', e=>{ if(!isDragging) return; const dx = (e.clientX - prevX); const dy = (e.clientY - prevY); prevX=e.clientX; prevY=e.clientY; rotY += dx*0.005; rotX += dy*0.005; });
  window.addEventListener('wheel', e=>{ camera.position.z += e.deltaY*0.3; camera.position.z = Math.max(50, Math.min(5000, camera.position.z)); });

  // responsive
  window.addEventListener('resize', ()=>{ camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); if(points) points.material.uniforms.pointScale.value = window.innerHeight/2; });

