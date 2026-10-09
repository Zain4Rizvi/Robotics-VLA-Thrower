import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import loadMujoco, { type MainModule, type MjData, type MjModel, type MjVFS, type MjvCamera } from "@mujoco/mujoco"

import { SceneView } from "@/scene-view"
import { publishFeed } from "@/use-playground"

export type RobotId = "stack" | "openarm" | "peg"

const RED = "stack the red cube on the green cube"
const GREEN = "stack the green cube on the red cube"

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

function mujoco() {
  mujocoPromise ??= loadMujoco()
  return mujocoPromise
}

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

export function usePlayback(canvasRef: RefObject<HTMLCanvasElement | null>) {
  const [robot, setRobot] = useState<RobotId>("stack")
  const [ready, setReady] = useState(false)
  const [message, setMessage] = useState("Loading")
  const [instruction, setInstructionText] = useState(RED)
  const [seed, setSeed] = useState<number | null>(null)
  const robotRef = useRef(robot)
  const catalogRef = useRef<Catalog | null>(null)
  const clipIndex = useRef(0)
  const tapeRef = useRef<Float32Array | null>(null)
  const frameRef = useRef(0)
  const nudgeRef = useRef<(action: "orbit" | "pan" | "zoom", dx: number, dy: number) => void>(() => {})
  const runtimeRef = useRef<{
    mujoco: MainModule
    model: MjModel
    data: MjData
    cam: MjvCamera
    view: SceneView
    vfs: MjVFS
  } | null>(null)

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
    runtime.view.paintColors(model, clip.rgba, (name) => mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, name))
    setInstructionText(clip.instruction)
    setSeed(clip.seed)
  }, [])

  useEffect(() => {
    robotRef.current = robot
  }, [robot])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let dead = false
    let frame = 0
    let last = 0
    let acc = 0
    let tick = 0
    const view = new SceneView(canvas)

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
      if (tick % 4 === 0) {
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

    const boot = async () => {
      try {
        const [mj, catalog, files] = await Promise.all([
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
        if (dead) return
        catalogRef.current = catalog
        const spec = catalog[robotRef.current]
        const names = files[robotRef.current]
        const vfs = new mj.MjVFS()
        await Promise.all(
          names.map(async (name) => {
            const bytes = await loadBytes(`/mujoco/${name}`)
            vfs.addBuffer(name, bytes.slice())
          }),
        )
        if (dead) {
          vfs.delete()
          return
        }
        const model = mj.MjModel.from_xml_path(spec.xml, vfs)
        const data = new mj.MjData(model)
        const cam = new mj.MjvCamera()
        if (dead) {
          data.delete()
          model.delete()
          cam.delete()
          vfs.delete()
          return
        }
        mj.mjv_defaultFreeCamera(model, cam)
        const look = cam.lookat as ArrayLike<number> & { [index: number]: number }
        look[0] = spec.lookat[0]
        look[1] = spec.lookat[1]
        look[2] = spec.lookat[2]
        cam.distance = spec.distance
        cam.azimuth = spec.azimuth
        cam.elevation = spec.elevation
        view.mount(model, data)
        runtimeRef.current = { mujoco: mj, model, data, cam, view, vfs }
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
        if (dead) return
        setReady(true)
        setMessage("")
        last = 0
        frame = window.requestAnimationFrame(loop)
      } catch (error) {
        if (dead) return
        setReady(false)
        setMessage(error instanceof Error ? error.message : "Could not open the scene")
      }
    }

    void boot()
    return () => {
      dead = true
      window.cancelAnimationFrame(frame)
      scene?.removeEventListener("wheel", onWheel)
      const runtime = runtimeRef.current
      runtimeRef.current = null
      tapeRef.current = null
      if (runtime) {
        runtime.data.delete()
        runtime.model.delete()
        runtime.cam.delete()
        runtime.vfs.delete()
      }
      view.dispose()
      nudgeRef.current = () => {}
      setReady(false)
    }
  }, [applyClip, canvasRef, robot])

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
      if (wanted !== RED && wanted !== GREEN) return false
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
