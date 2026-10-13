import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import loadMujoco, { type MainModule, type MjData, type MjModel, type MjvCamera } from "@mujoco/mujoco"

import { SceneView } from "@/scene-view"
import { publishFeed } from "@/use-playground"

export type RobotId = "stack" | "openarm" | "peg"

export const STACK_PROMPTS = [
  "stack the red cube on the green cube",
  "stack the green cube on the red cube",
] as const

type Rgba = { name: string; rgba: number[] }
type BodyPose = { name: string; pos: number[]; quat: number[] }

type Clip = {
  id: string
  instruction: string
  seed: number
  file: string
  frames: number
  nq: number
  rgba: Rgba[]
  bodies: BodyPose[]
}

type RobotSpec = {
  xml: string
  fps: number
  cameras: [string, string]
  lookat: [number, number, number]
  distance: number
  azimuth: number
  elevation: number
  clips: Clip[]
}

type Catalog = Record<RobotId, RobotSpec>

let mujocoPromise: Promise<MainModule> | null = null
const fileCache = new Map<string, Uint8Array>()
const staged = new Set<string>()
const heldRobots = new Map<RobotId, { mujoco: MainModule; model: MjModel; data: MjData; cam: MjvCamera }>()
const heldLoads = new Map<RobotId, Promise<{ mujoco: MainModule; model: MjModel; data: MjData; cam: MjvCamera }>>()

type Held = { mujoco: MainModule; model: MjModel; data: MjData; cam: MjvCamera }

function loadHeld(id: RobotId, mj: MainModule, catalog: Catalog, files: Record<RobotId, string[]>): Promise<Held> {
  const ready = heldRobots.get(id)
  if (ready) return Promise.resolve(ready)
  const pending = heldLoads.get(id)
  if (pending) return pending
  const job = (async () => {
    try {
      const spec = catalog[id]
      const packed = await Promise.all(files[id].map(async (name) => [name, await loadBytes(`/mujoco/${name}`)] as const))
      for (const [name, bytes] of packed) stageFile(mj, name, bytes)
      const model = mj.MjModel.from_xml_path(spec.xml)
      const data = new mj.MjData(model)
      const cam = new mj.MjvCamera()
      mj.mjv_defaultFreeCamera(model, cam)
      const look = cam.lookat as ArrayLike<number> & { [index: number]: number }
      look[0] = spec.lookat[0]
      look[1] = spec.lookat[1]
      look[2] = spec.lookat[2]
      cam.distance = spec.distance
      cam.azimuth = spec.azimuth
      cam.elevation = spec.elevation
      const held = { mujoco: mj, model, data, cam }
      heldRobots.set(id, held)
      return held
    } catch (error) {
      heldLoads.delete(id)
      throw error
    }
  })()
  heldLoads.set(id, job)
  return job
}

function stageFile(mj: MainModule, name: string, bytes: Uint8Array) {
  if (staged.has(name)) return
  const slash = name.lastIndexOf("/")
  if (slash > 0) {
    const dir = name.slice(0, slash)
    if (!staged.has(dir)) {
      mj.FS.mkdirTree(dir, 0o777)
      staged.add(dir)
    }
  }
  mj.FS.writeFile(name, bytes)
  staged.add(name)
}

function mujoco() {
  mujocoPromise ??= loadMujoco()
  return mujocoPromise
}

void mujoco()

async function loadBytes(url: string) {
  const cached = fileCache.get(url)
  if (cached) return cached
  const response = await fetch(url)
  if (!response.ok) throw new Error(`missing ${url}`)
  const bytes = new Uint8Array(await response.arrayBuffer())
  fileCache.set(url, bytes)
  return bytes
}

function writeArray(target: ArrayLike<number> & { [index: number]: number }, index: number, values: number[], stride: number) {
  for (let i = 0; i < stride; i += 1) target[index * stride + i] = values[i]
}

export function usePlayback(canvasRef: RefObject<HTMLCanvasElement | null>, active: boolean) {
  const [robot, setRobot] = useState<RobotId>("stack")
  const [ready, setReady] = useState(false)
  const [message, setMessage] = useState("Loading")
  const [instruction, setInstructionText] = useState<string>(STACK_PROMPTS[0])
  const [seed, setSeed] = useState<number | null>(null)
  const robotRef = useRef(robot)
  const activeRef = useRef(active)
  const catalogRef = useRef<Catalog | null>(null)
  const clipIndex = useRef(0)
  const tapeRef = useRef<Float32Array | null>(null)
  const frameRef = useRef(0)
  const nudgeRef = useRef<(action: "orbit" | "pan" | "zoom", dx: number, dy: number) => void>(() => {})
  const openRef = useRef<(id: RobotId) => void>(() => {})
  const runtimeRef = useRef<Held | null>(null)
  const viewRef = useRef<SceneView | null>(null)

  const applyClip = useCallback(async (index: number) => {
    const spec = catalogRef.current?.[robotRef.current]
    const runtime = runtimeRef.current
    if (!spec || !runtime) return
    const clip = spec.clips[index]
    if (!clip) return
    const bytes = await loadBytes(`/${clip.file}`)
    const tape = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
    if (tape.length !== clip.frames * clip.nq || clip.nq !== runtime.model.nq) {
      throw new Error(`${clip.id} does not match the model`)
    }
    tapeRef.current = tape
    clipIndex.current = index
    frameRef.current = 0
    const { mujoco: mj, model } = runtime
    const bodyPos = model.body_pos as ArrayLike<number> & { [index: number]: number }
    const bodyQuat = model.body_quat as ArrayLike<number> & { [index: number]: number }
    for (const body of clip.bodies) {
      const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, body.name)
      if (id < 0) continue
      writeArray(bodyPos, id, body.pos, 3)
      writeArray(bodyQuat, id, body.quat, 4)
    }
    viewRef.current?.paintColors(model, clip.rgba, (name) => mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, name))
    setInstructionText(clip.instruction)
    setSeed(clip.seed)
  }, [])

  useEffect(() => {
    robotRef.current = robot
  }, [robot])

  useEffect(() => {
    activeRef.current = active
  }, [active])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let dead = false
    let frame = 0
    let last = 0
    let acc = 0
    let tick = 0
    const loading = Promise.all([
      mujoco(),
      fetch("/clips/catalog.json").then((response) => {
        if (!response.ok) throw new Error("clip catalog is missing")
        return response.json() as Promise<Catalog>
      }),
      fetch("/mujoco/files.json").then((response) => {
        if (!response.ok) throw new Error("scene files are missing")
        return response.json() as Promise<Record<RobotId, string[]>>
      }),
    ])
    const view = new SceneView(canvas)
    viewRef.current = view

    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const height = (event.currentTarget as HTMLElement).clientHeight || 1
      nudgeRef.current("zoom", 0, event.deltaY / height)
    }
    const scene = document.querySelector<HTMLElement>("[data-scene]")
    scene?.addEventListener("wheel", onWheel, { passive: false })

    const loop = (now: number) => {
      if (dead) return
      frame = window.requestAnimationFrame(loop)
      if (!activeRef.current) {
        last = 0
        return
      }
      const runtime = runtimeRef.current
      const tape = tapeRef.current
      const spec = catalogRef.current?.[robotRef.current]
      if (!runtime || !tape || !spec) return
      const dt = last ? now - last : 0
      last = now
      acc += dt
      const stepMs = 1000 / spec.fps
      const clip = spec.clips[clipIndex.current]
      while (acc >= stepMs) {
        acc -= stepMs
        frameRef.current = (frameRef.current + 1) % clip.frames
      }
      const qpos = runtime.data.qpos as Float64Array
      const start = frameRef.current * clip.nq
      for (let i = 0; i < clip.nq; i += 1) qpos[i] = tape[start + i]
      runtime.mujoco.mj_forward(runtime.model, runtime.data)
      const rect = canvas.getBoundingClientRect()
      view.resize(rect.width, rect.height)
      view.sync(runtime.mujoco, runtime.cam)
      tick += 1
      if (tick > 8 && tick % 8 === 0) {
        spec.cameras.forEach((name, index) => {
          const shot = view.capture(name, (camera) =>
            runtime.mujoco.mj_name2id(runtime.model, runtime.mujoco.mjtObj.mjOBJ_CAMERA.value, camera),
          )
          shot?.toBlob((blob) => {
            if (blob && !dead) publishFeed(index, blob)
          }, "image/jpeg", 0.72)
        })
      }
    }

    let loopOn = false
    const show = async (id: RobotId) => {
      try {
        const [mj, catalog, files] = await loading
        if (dead) return
        catalogRef.current = catalog
        const held = await loadHeld(id, mj, catalog, files)
        if (dead || robotRef.current !== id) return
        tapeRef.current = null
        view.mount(id, held.model, held.data)
        runtimeRef.current = held
        const { model, cam } = held
        nudgeRef.current = (action, dx, dy) => {
          const code =
            action === "orbit"
              ? mj.mjtMouse.mjMOUSE_ROTATE_V.value
              : action === "pan"
                ? mj.mjtMouse.mjMOUSE_MOVE_V.value
                : mj.mjtMouse.mjMOUSE_ZOOM.value
          mj.mjv_moveCamera(model, code, dx, dy, cam)
        }
        await applyClip(0)
        if (dead || robotRef.current !== id) return
        setReady(true)
        setMessage("")
        last = 0
        if (!loopOn) {
          loopOn = true
          frame = window.requestAnimationFrame(loop)
        }
      } catch (error) {
        if (dead || robotRef.current !== id) return
        setReady(false)
        setMessage(error instanceof Error ? error.message : "Could not open the scene")
      }
    }

    openRef.current = (id) => {
      void show(id)
    }
    return () => {
      dead = true
      window.cancelAnimationFrame(frame)
      scene?.removeEventListener("wheel", onWheel)
      runtimeRef.current = null
      tapeRef.current = null
      view.dispose()
      viewRef.current = null
      nudgeRef.current = () => {}
      openRef.current = () => {}
      setReady(false)
    }
  }, [applyClip, canvasRef])

  useEffect(() => {
    openRef.current(robot)
  }, [robot])

  const nextLayout = useCallback(() => {
    const spec = catalogRef.current?.[robotRef.current]
    if (!spec) return
    const current = spec.clips[clipIndex.current]
    if (!current) return
    const pool =
      robotRef.current === "stack"
        ? spec.clips.map((clip, index) => (clip.instruction === current.instruction ? index : -1)).filter((index) => index >= 0)
        : spec.clips.map((_clip, index) => index)
    const at = pool.indexOf(clipIndex.current)
    const next = pool[(at + 1) % pool.length]
    void applyClip(next).catch((error: unknown) => {
      setMessage(error instanceof Error ? error.message : "Could not open that layout")
    })
  }, [applyClip])

  const submitInstruction = useCallback(
    (text: string) => {
      const spec = catalogRef.current?.stack
      if (!spec || robotRef.current !== "stack") return false
      const wanted = text.trim().toLowerCase()
      if (!STACK_PROMPTS.includes(wanted as (typeof STACK_PROMPTS)[number])) return false
      const current = spec.clips[clipIndex.current]
      if (current && current.instruction === wanted) {
        frameRef.current = 0
        return true
      }
      const index = spec.clips.findIndex((clip) => clip.instruction === wanted)
      if (index < 0) return false
      void applyClip(index)
      return true
    },
    [applyClip],
  )

  const selectRobot = useCallback((id: RobotId) => {
    if (id === robotRef.current) return
    setReady(false)
    setMessage("Loading")
    setRobot(id)
  }, [])

  const nudge = useCallback((action: "orbit" | "pan" | "zoom", dx: number, dy: number) => {
    nudgeRef.current(action, dx, dy)
  }, [])

  return { robot, ready, message, instruction, seed, selectRobot, submitInstruction, nextLayout, nudge }
}
