# TaystJK RT — working notes

Notes on the hardware ray-tracing (ray-query) work in `codemp/rd-vulkan/`.

## Known limitations

### The renderer requires an RT-capable GPU to load its shaders

**Issue:** once the ray-query shadow lands, the generic world fragment shader
(`shaders/glsl/gen_frag.tmpl`) embeds `#extension GL_EXT_ray_query`, so its compiled
SPIR-V declares the `RayQueryKHR` capability. Vulkan requires the device to support that
capability to create the shader module / graphics pipeline. On a GPU **without** hardware
ray tracing (no `VK_KHR_ray_query`), shader/pipeline creation fails and the Vulkan
renderer won't start.

**Why it affects every draw, not just RT ones:** we chose the *spec-constant* approach —
a `rt_shadow` specialization constant gates whether the trace **executes**, but it does
**not** remove the `RayQueryKHR` capability from the module. So `gen_frag` (used by all
generic geometry, including 2D/UI) carries the RT-device dependency even when the trace is
switched off.

**Impact / decision:** acceptable for this fork — target HW is an RTX 5090 (see
`CLAUDE.md`). Documented here so it's not a surprise: the `rd-vulkan` build can no longer
run on non-RT hardware. The other renderers (`rd-rend2`, `rd-taystjk`) are unaffected.

**If non-RT fallback is ever wanted (mitigations):**
- Compile two `gen_frag` sets — one with `GL_EXT_ray_query`, one without — and pick at
  runtime based on `vk.rayQuery`.
- Move the trace into a dedicated RT-only shader variant / pipeline instead of the shared
  generic shader.
- Keep `ray_query` out of `gen_frag` entirely and only enable RT shading through a
  separate world-shadow path.

## Build / environment assumptions

- **Hard-coded VS compiler path.** `shaders/compile_threaded.bat` calls `VsDevCmd.bat` from a
  hard-coded **VS2022 Professional** path
  (`C:\Program Files\Microsoft Visual Studio\2022\Professional\...`) to get `cl.exe` for building
  `compile_threaded.exe`. This is machine-specific — another box (Community/Enterprise, or a
  different VS version) needs this path adjusted. It also sets `VULKAN_SDK` to
  `C:\VulkanSDK\1.4.350.0` if unset. (`compile.bat` is stale/broken; use `compile_threaded.bat`.)

## Known shortcuts / TODO

- **Hard-coded light source.** `traceShadow()` in `gen_frag.tmpl` uses a literal light position
  `vec3(192, 0, 224)` — that's test-map "light 2". For any real map this must come from the
  engine's actual light data (and ideally loop over relevant lights) instead of a constant.
- **Shutdown / vid_restart leaks.** `vkDestroyDevice` reports ~21 leaked objects: the RT objects
  (BLAS, TLAS, their buffers/memory, the position/index buffers, the AS descriptor set) aren't
  freed on shutdown, and `R_Init` running twice (a vid_restart) reallocates the AS descriptor and
  orphans the first. Need to call `vk_release_world_rt` on shutdown/restart and free the descriptor
  + AS set layout (the `set_layout_as` teardown is already in `vk_shutdown`).
- **Spike scoping.** `rt_shadow` currently defaults on for *all* `gen_frag` draws; proper Step C
  scopes it to world-opaque pipelines only (constant_id 18, set per-pipeline) so 2D/UI/menu stop
  tracing.
