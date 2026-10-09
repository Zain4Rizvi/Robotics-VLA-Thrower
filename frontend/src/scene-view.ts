import * as THREE from "three"

import type { MainModule, MjData, MjModel, MjvCamera } from "@mujoco/mujoco"

const PLANE = 0
const SPHERE = 2
const CAPSULE = 3
const ELLIPSOID = 4
const CYLINDER = 5
const BOX = 6
const MESH = 7

type GeomMesh = { id: number; mesh: THREE.Mesh }

function checkerTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 256
  canvas.height = 256
  const ctx = canvas.getContext("2d")
  if (!ctx) return null
  const cells = 8
  const cell = canvas.width / cells
  for (let y = 0; y < cells; y += 1) {
    for (let x = 0; x < cells; x += 1) {
      ctx.fillStyle = (x + y) % 2 === 0 ? "#3d5568" : "#243644"
      ctx.fillRect(x * cell, y * cell, cell, cell)
    }
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.repeat.set(10, 10)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

function read3(value: unknown): [number, number, number] {
  if (value && typeof value === "object" && "x" in value) {
    const vec = value as { x: number; y: number; z: number }
    return [vec.x, vec.y, vec.z]
  }
  const list = value as ArrayLike<number>
  return [Number(list[0]), Number(list[1]), Number(list[2])]
}

function setPose(object: THREE.Object3D, xpos: ArrayLike<number>, xmat: ArrayLike<number>, index: number) {
  const p = index * 3
  const r = index * 9
  const e = object.matrix.elements
  e[0] = xmat[r]
  e[1] = xmat[r + 3]
  e[2] = xmat[r + 6]
  e[3] = 0
  e[4] = xmat[r + 1]
  e[5] = xmat[r + 4]
  e[6] = xmat[r + 7]
  e[7] = 0
  e[8] = xmat[r + 2]
  e[9] = xmat[r + 5]
  e[10] = xmat[r + 8]
  e[11] = 0
  e[12] = xpos[p]
  e[13] = xpos[p + 1]
  e[14] = xpos[p + 2]
  e[15] = 1
  object.matrixWorldNeedsUpdate = true
}

function meshGeometry(model: MjModel, meshId: number, size: ArrayLike<number>) {
  const vertAdr = Number(model.mesh_vertadr[meshId])
  const vertNum = Number(model.mesh_vertnum[meshId])
  const faceAdr = Number(model.mesh_faceadr[meshId])
  const faceNum = Number(model.mesh_facenum[meshId])
  if (vertNum <= 0 || faceNum <= 0) return null
  const verts = model.mesh_vert as ArrayLike<number>
  const faces = model.mesh_face as ArrayLike<number>
  const positions = new Float32Array(vertNum * 3)
  for (let i = 0; i < vertNum; i += 1) {
    const src = (vertAdr + i) * 3
    positions[i * 3] = Number(verts[src]) * Number(size[0] || 1)
    positions[i * 3 + 1] = Number(verts[src + 1]) * Number(size[1] || 1)
    positions[i * 3 + 2] = Number(verts[src + 2]) * Number(size[2] || 1)
  }
  const index = new Uint32Array(faceNum * 3)
  const face0 = faceAdr * 3
  for (let i = 0; i < faceNum * 3; i += 1) index[i] = Number(faces[face0 + i])
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
  geometry.setIndex(new THREE.BufferAttribute(index, 1))
  geometry.computeVertexNormals()
  return geometry
}

function primitiveGeometry(type: number, size: ArrayLike<number>) {
  const x = Number(size[0]) || 0.01
  const y = Number(size[1]) || x
  const z = Number(size[2]) || x
  if (type === PLANE) {
    const width = x > 0 ? x * 2 : 12
    const height = y > 0 ? y * 2 : 12
    return new THREE.PlaneGeometry(width, height)
  }
  if (type === SPHERE) return new THREE.SphereGeometry(x, 24, 16)
  if (type === ELLIPSOID) {
    const geometry = new THREE.SphereGeometry(1, 24, 16)
    geometry.scale(x, y, z)
    return geometry
  }
  if (type === CYLINDER) {
    const geometry = new THREE.CylinderGeometry(x, x, y * 2, 20)
    geometry.rotateX(Math.PI / 2)
    return geometry
  }
  if (type === CAPSULE) {
    const geometry = new THREE.CapsuleGeometry(x, y * 2, 6, 16)
    geometry.rotateX(Math.PI / 2)
    return geometry
  }
  if (type === BOX) return new THREE.BoxGeometry(x * 2, y * 2, z * 2)
  return null
}

export class SceneView {
  readonly scene = new THREE.Scene()
  readonly freeCam = new THREE.PerspectiveCamera(45, 1, 0.01, 50)
  readonly fixedCam = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 50)
  private readonly hero: THREE.WebGLRenderer
  private readonly inset: THREE.WebGLRenderer
  private readonly insetCanvas = document.createElement("canvas")
  private readonly geoms: GeomMesh[] = []
  private readonly floor: THREE.Texture | null
  private model: MjModel | null = null
  private data: MjData | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.scene.background = new THREE.Color("#6d8ea3")
    this.floor = checkerTexture()
    this.scene.add(new THREE.AmbientLight("#ffffff", 0.72))
    const sun = new THREE.DirectionalLight("#ffffff", 1.15)
    sun.position.set(0.6, -1.1, 2.2)
    this.scene.add(sun)
    this.hero = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false })
    this.hero.outputColorSpace = THREE.SRGBColorSpace
    this.hero.toneMapping = THREE.NoToneMapping
    this.hero.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    this.insetCanvas.width = 256
    this.insetCanvas.height = 192
    this.inset = new THREE.WebGLRenderer({ canvas: this.insetCanvas, antialias: true, alpha: false })
    this.inset.outputColorSpace = THREE.SRGBColorSpace
    this.inset.toneMapping = THREE.NoToneMapping
    this.inset.setSize(256, 192, false)
    }

  mount(model: MjModel, data: MjData) {
    this.clear()
    this.model = model
    this.data = data
    const types = model.geom_type as ArrayLike<number>
    const groups = model.geom_group as ArrayLike<number>
    const sizes = model.geom_size as ArrayLike<number>
    const dataIds = model.geom_dataid as ArrayLike<number>
    const matIds = model.geom_matid as ArrayLike<number>
    const rgba = model.geom_rgba as ArrayLike<number>
    const matRgba = model.mat_rgba as ArrayLike<number>
    for (let id = 0; id < model.ngeom; id += 1) {
      if (Number(groups[id]) >= 3) continue
      const type = Number(types[id])
      const size = [sizes[id * 3], sizes[id * 3 + 1], sizes[id * 3 + 2]]
      const geometry = type === MESH ? meshGeometry(model, Number(dataIds[id]), [1, 1, 1]) : primitiveGeometry(type, size)
      if (!geometry) continue
      const mat = Number(matIds[id])
      const src = mat >= 0 ? matRgba : rgba
      const base = mat >= 0 ? mat * 4 : id * 4
      const color = new THREE.Color(Number(src[base]), Number(src[base + 1]), Number(src[base + 2]))
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: type === PLANE ? 0.95 : 0.55,
        metalness: 0.04,
        side: THREE.DoubleSide,
        map: type === PLANE ? this.floor : null,
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.matrixAutoUpdate = false
      mesh.frustumCulled = false
      this.scene.add(mesh)
      this.geoms.push({ id, mesh })
    }
  }

  resize(width: number, height: number) {
    if (width < 2 || height < 2) return
    this.hero.setSize(width, height, false)
    this.freeCam.aspect = width / height
    this.freeCam.updateProjectionMatrix()
  }

  sync(mujoco: MainModule, cam: MjvCamera) {
    const model = this.model
    const data = this.data
    if (!model || !data) return
    const xpos = data.geom_xpos as ArrayLike<number>
    const xmat = data.geom_xmat as ArrayLike<number>
    for (const item of this.geoms) setPose(item.mesh, xpos, xmat, item.id)
    this.scene.updateMatrixWorld(true)
    const gl = mujoco.mjv_camera2GLCamera(model, data, cam)
    const pos = read3(gl.pos)
    const forward = read3(gl.forward)
    const up = read3(gl.up)
    gl.delete()
    this.freeCam.up.set(up[0], up[1], up[2])
    this.freeCam.position.set(pos[0], pos[1], pos[2])
    this.freeCam.lookAt(pos[0] + forward[0], pos[1] + forward[1], pos[2] + forward[2])
    this.freeCam.fov = model.vis.global.fovy || 45
    this.freeCam.updateProjectionMatrix()
    this.hero.render(this.scene, this.freeCam)
  }

  paintColors(model: MjModel, rows: { name: string; rgba: number[] }[], nameOf: (name: string) => number) {
    const rgba = model.geom_rgba as ArrayLike<number> & { [index: number]: number }
    const matIds = model.geom_matid as ArrayLike<number>
    const painted = new Map<number, number[]>()
    for (const row of rows) {
      const id = nameOf(row.name)
      if (id < 0) continue
      const mat = Number(matIds[id])
      const untouched = row.rgba.every((value, index) => Math.abs(value - (index === 3 ? 1 : 0.5)) < 1e-3)
      if (mat >= 0 && untouched) continue
      rgba[id * 4] = row.rgba[0]
      rgba[id * 4 + 1] = row.rgba[1]
      rgba[id * 4 + 2] = row.rgba[2]
      rgba[id * 4 + 3] = row.rgba[3]
      painted.set(id, row.rgba)
    }
    for (const item of this.geoms) {
      const color = painted.get(item.id)
      if (!color) continue
      const material = item.mesh.material as THREE.MeshStandardMaterial
      material.color.setRGB(color[0], color[1], color[2])
    }
  }

  capture(cameraName: string, nameOf: (name: string) => number) {
    const model = this.model
    const data = this.data
    if (!model || !data) return
    const id = nameOf(cameraName)
    if (id < 0) return
    const xpos = data.cam_xpos as ArrayLike<number>
    const xmat = data.cam_xmat as ArrayLike<number>
    const r = id * 9
    const x = Number(xpos[id * 3])
    const y = Number(xpos[id * 3 + 1])
    const z = Number(xpos[id * 3 + 2])
    this.fixedCam.up.set(Number(xmat[r + 1]), Number(xmat[r + 4]), Number(xmat[r + 7]))
    this.fixedCam.position.set(x, y, z)
    this.fixedCam.lookAt(x - Number(xmat[r + 2]), y - Number(xmat[r + 5]), z - Number(xmat[r + 8]))
    this.fixedCam.fov = Number(model.cam_fovy[id]) || 55
    this.fixedCam.updateProjectionMatrix()
    this.inset.render(this.scene, this.fixedCam)
    return this.insetCanvas
  }

  clear() {
    for (const item of this.geoms) {
      this.scene.remove(item.mesh)
      item.mesh.geometry.dispose()
      const material = item.mesh.material as THREE.MeshStandardMaterial
      material.dispose()
    }
    this.geoms.length = 0
    this.model = null
    this.data = null
  }

  dispose() {
    this.clear()
    this.floor?.dispose()
    this.hero.dispose()
    this.inset.dispose()
  }
}
