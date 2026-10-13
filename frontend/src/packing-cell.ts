import * as THREE from "three"

function solid(color: string, roughness = 0.82) {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.04 })
}

function box(size: [number, number, number], position: [number, number, number], color: string, roughness?: number) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), solid(color, roughness))
  mesh.position.set(...position)
  return mesh
}

export function partTexture(label: string, body: string, panel: string) {
  const canvas = document.createElement("canvas")
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext("2d")
  if (!ctx) return null
  ctx.fillStyle = body
  ctx.fillRect(0, 0, 128, 128)
  ctx.strokeStyle = "rgba(0,0,0,0.35)"
  ctx.lineWidth = 8
  ctx.strokeRect(4, 4, 120, 120)
  ctx.fillStyle = panel
  ctx.fillRect(28, 22, 72, 52)
  ctx.fillStyle = "rgba(0,0,0,0.45)"
  ctx.beginPath()
  ctx.arc(22, 22, 5, 0, Math.PI * 2)
  ctx.arc(106, 22, 5, 0, Math.PI * 2)
  ctx.arc(22, 106, 5, 0, Math.PI * 2)
  ctx.arc(106, 106, 5, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = "#f4efe4"
  ctx.font = "700 18px sans-serif"
  ctx.textAlign = "center"
  ctx.textBaseline = "middle"
  ctx.fillText(label, 64, 100)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

function part(label: string, body: string, panel: string, position: [number, number, number]) {
  const map = partTexture(label, body, panel)
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.03, 0.03, 0.03),
    new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.62, metalness: 0.04, map }),
  )
  mesh.position.set(...position)
  return mesh
}

function buildSoFar(group: THREE.Group) {
  const x = 0.08
  const y = -0.39
  group.add(box([0.22, 0.08, 0.012], [x, y, 0.006], "#2a2e33", 0.55))
  group.add(part("BASE", "#1f8a3b", "#143322", [x - 0.06, y, 0.027]))
  group.add(part("COVER", "#c4322a", "#6e1c16", [x - 0.06, y, 0.057]))
  group.add(part("BASE", "#1f8a3b", "#143322", [x + 0.01, y, 0.027]))
  group.add(box([0.03, 0.03, 0.004], [x + 0.07, y, 0.014], "#14171a", 0.35))
}

export function packingCell() {
  const group = new THREE.Group()
  group.add(box([1.8, 1.5, 0.04], [0, -0.28, -0.18], "#8d8680", 0.95))
  group.add(box([0.78, 0.62, 0.02], [0, -0.2, -0.01], "#d7c4a3", 0.62))

  for (const [x, y] of [
    [-0.34, 0.06],
    [0.34, 0.06],
    [-0.34, -0.46],
    [0.34, -0.46],
  ] as const) {
    group.add(box([0.028, 0.028, 0.14], [x, y, -0.09], "#5c534c", 0.7))
  }

  const tote = "#2f4a62"
  const tx = 0.28
  const ty = -0.16
  const wall = 0.004
  const width = 0.09
  const depth = 0.07
  const height = 0.035
  group.add(box([width, depth, wall], [tx, ty, wall / 2], tote))
  group.add(box([wall, depth, height], [tx - width / 2, ty, height / 2], tote))
  group.add(box([wall, depth, height], [tx + width / 2, ty, height / 2], tote))
  group.add(box([width, wall, height], [tx, ty - depth / 2, height / 2], tote))
  group.add(box([width, wall, height], [tx, ty + depth / 2, height / 2], tote))

  buildSoFar(group)
  group.add(box([0.025, 0.02, 0.018], [tx - 0.015, ty, 0.013], "#a67c52", 0.7))
  group.add(box([0.02, 0.02, 0.02], [tx + 0.015, ty + 0.008, 0.014], "#3d4f6f", 0.6))

  const shelfY = 0.42
  const shelf = "#6e6258"
  group.add(box([0.02, 0.02, 0.42], [-0.18, shelfY, 0.05], shelf))
  group.add(box([0.02, 0.02, 0.42], [0.18, shelfY, 0.05], shelf))
  group.add(box([0.4, 0.12, 0.012], [0, shelfY, -0.02], shelf))
  group.add(box([0.4, 0.12, 0.012], [0, shelfY, 0.16], shelf))
  group.add(box([0.045, 0.04, 0.04], [-0.08, shelfY, 0], "#a67c52"))
  group.add(box([0.04, 0.035, 0.05], [0.06, shelfY, 0.005], "#3d4f6f"))

  const lamp = new THREE.PointLight("#ffe0b8", 1.4, 2.4, 2)
  lamp.position.set(0.05, -0.15, 0.72)
  group.add(lamp)
  return group
}
