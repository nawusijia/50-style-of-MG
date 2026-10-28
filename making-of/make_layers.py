import sys
src,dst=sys.argv[1],sys.argv[2]
h=open(src,encoding='utf8').read()
def R(o,n,c=1):
    global h
    assert h.count(o)==c,(o,h.count(o)); h=h.replace(o,n)
R("const W = 1920, H = 1080;","const W = 1920, H = 1080;\nfunction L(n) { const o = window.__LY; return !o || o[n] !== false; }")
R("  drawStars(t, -cam.psi * 700 - t * 8, cam.yh, 1);","  if (L('stars')) drawStars(t, -cam.psi * 700 - t * 8, cam.yh, 1);",2)
R("  drawSun(sunX, sunY, R, 8, t, bright, white);\n  drawTerrain(cam);\n  drawLane(cam, 1);",
  "  if (L('sun')) drawSun(sunX, sunY, R, 8, t, bright, white);\n  if (L('terrain')) { drawTerrain(cam);\n  drawLane(cam, 1); }")
R("  if (!white) drawCar(cam, t);","  if (!white && L('car')) drawCar(cam, t);")
R("  ctx.fillStyle = hz; ctx.fillRect(-300, cam.yh - 140, W + 600, 146);\n  drawTerrain(cam);",
  "  ctx.fillStyle = hz; ctx.fillRect(-300, cam.yh - 140, W + 600, 146);\n  if (L('terrain')) drawTerrain(cam);")
R("  if (ca > 0) {","  if (ca > 0 && L('terrain')) {")
R("  if (t >= T_BOOM && t < T_BOOM + 0.4) {","  if (t >= T_BOOM && t < T_BOOM + 0.4 && L('particles')) {")
R("  drawNebGlow(t);\n  drawParticles(t, false);\n  if (t >= SWEEP0) {\n    buildLayer(t);",
  "  if (L('particles')) { drawNebGlow(t);\n  drawParticles(t, false); }\n  if (t >= SWEEP0 && L('title')) {\n    buildLayer(t);")
R("    if (t >= GLOW_ON) {","    if (t >= GLOW_ON && L('glow')) {")
R("  drawParticles(t, true);\n  if (t >= SWEEP0) drawLayer(LAY, 0, 0, false);\n  drawDebris(t);\n  drawSubtitle(t);\n  if (withPen) drawPen(ctx, penState(t), 1);",
  "  if (L('particles')) drawParticles(t, true);\n  if (t >= SWEEP0 && L('title')) drawLayer(LAY, 0, 0, false);\n  if (L('pen')) drawDebris(t);\n  if (L('sub')) drawSubtitle(t);\n  if (withPen && L('pen')) drawPen(ctx, penState(t), 1);")
R("  drawSubMask(t);\n  // flash","  if (L('sub')) drawSubMask(t);\n  // flash")
R("  else if (t >= T_BOOM) { drawPen(ctx, penState(t), 1); }","  else if (t >= T_BOOM && L('pen')) { drawPen(ctx, penState(t), 1); }")
R("function glitchI(t) {\n","function glitchI(t) {\n  if (!L('glitch')) return 0;\n")
R("  if (t < T_CRT) {\n    const fr = Math.floor(t * 60 + 1e-6);","  if (t < T_CRT && L('fx')) {\n    const fr = Math.floor(t * 60 + 1e-6);")
R("  if (t >= T_HOLD && t < T_CRT) {","  if (t >= T_HOLD && t < T_CRT && L('fx')) {")
R('<audio id="music" src="music.wav" preload="auto"></audio>','<audio id="music" preload="none"></audio>')
open(dst,'w',encoding='utf8').write(h); print('ok')
