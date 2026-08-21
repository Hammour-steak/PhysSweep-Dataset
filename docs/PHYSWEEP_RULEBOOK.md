# PhysSweep Rulebook

## Architecture rule

Concrete objects and scenes belong in profiles and scene kits. Compatibility belongs in `compatibility.json`. Python may branch only on reusable behavior types; it must not branch on concrete ids. The active bundle and full boundary are documented in `PHYSWEEP_SAMPLING_ARCHITECTURE.md`.

## Camera rule

The active generic solver is `motion_structure_camera_v10`.

- Camera semantics have two independent parts. The motion observation intent selects the key temporal window; the structure context selects the physical anchors that explain the interaction.
- The only observation intents are `surface_travel`, `contact_event`, `ballistic_arc`, and `support_transition`. The only structure contexts are `horizontal_surface`, `inclined_surface`, `impact_boundary`, `edge_and_landing`, and `ramp_and_landing`.
- `expected_motion` contains physics acceptance semantics only. It must not contain camera fields. The compiled `camera_request.observation` is the sole camera contract.
- Surface travel observes the start and main travel interval. Contact events observe the approach, first required contact, and a bounded post-contact interval. Ballistic motion observes launch, apex, and first contact. Support transitions observe both sides of the structural boundary.
- A required structural anchor must remain visible: a horizontal support boundary, both ends of a slope, an impact surface, an edge plus landing point, or a ramp plus landing surface.
- `ramp_and_landing` uses the ramp high edge, ramp low edge, and the trajectory-local landing contact. The remote bounds of a large environment floor are never structural anchors.
- Camera minimum distance is derived from support size. Motion intent supplies the permitted elevation range, target blend, distance allowance, and partial-exit policy.
- The solver starts with the declared lens, normally 44 mm. Only when no pose satisfies every framing constraint may it try 40, 36, then 32 mm; focal fallback never relaxes object size, initial visibility, trajectory coverage, context, or occlusion thresholds.
- Object readability is evaluated over the observed interval, not only at frame zero. The median projected AABB span must meet the intent threshold; the initial AABB must still span at least 6%, remain at least 75% visible, and keep its center inside the 5% margin.
- At least 80% of observed trajectory centers remain inside the actual image. Full-trajectory center coverage is also measured against the exact `[0, 1]` image bounds; its motion-specific threshold controls permitted late exit. At least 90% of primary-motion samples and the declared fraction of the full trajectory remain unoccluded.
- Static geometry blocks the camera only when metadata sets `occludes_camera`. Visible supports, ramps, landing surfaces, cabinets, legs, impact walls, and rails are blockers; the environment floor is not.
- Environment walls, decor, set pieces, and mesh proxies are frozen before simulation. The camera solver must respect records marked `occludes_camera`; it may not delete, move, or disable a physical environment collider to improve composition.
- Partial exit after the primary interaction is allowed when the observation intent permits it. The solver does not pull back merely to retain an unimportant trajectory tail.
- Asset-proxy framing follows the same intent/structure contract and expands the observed path by the dynamic asset's canonical extent plus static-prop bounds.

The solver remains one implementation. Motion and structure change its declared objective and anchors; no concrete scene or asset receives a private camera branch.

## Temporal physics rule

- Videos are sampled at 24 fps, so accepted angular speed is capped at 60 rad/s. This stays below a conservative 80% of the rotational sampling Nyquist limit and prevents orientation aliasing between PyBullet state, point tracks, and rendered frames.
- The limit applies to the complete simulated trajectory, including post-contact spin. A sample that exceeds it is rejected and resampled; the renderer must not clamp or rewrite the trajectory.

## Object readability rule

- Physical scale is sampled before simulation and is immutable afterward; rendering may not enlarge an object independently.
- A sampled object below the readability floor is enlarged through a shape-level heuristic: sphere and cylinder use diameter, while cuboid uses its second-largest extent.
- The characteristic-extent target is 0.09 m and readability-only uplift is capped at 1.3x. Larger scale-bin samples remain valid.
- Metadata records sampled scale, readability floor, effective scale, characteristic rule, and whether adjustment occurred.

## Scene visual rule

- The support collision kit owns the primary interaction surface and global floor. The room profile owns paired visual context and static environment collision.
- Scene context and its `environment_binding` must be selected and frozen before simulation. PyBullet and Blender obey the same hashed placement exactly.
- Programmatic rooms provide the scalable controlled layer. Calibrated support and context GLBs provide the semantic-rich layer, targeting a 60/40 procedural/GLB split.
- A background GLB requires an asset hash, license, metric transform, support-plane alignment, camera-safe bounds, a reviewed first frame, and a validated static collision proxy before admission.
- Backdrop assets are filtered by both semantic theme and scene class. Their visible floor may remain, but near-horizontal collision faces in the reviewed floor band are removed so the analytic global floor is not duplicated.
- A real support mesh may replace a proxy visually only after its tabletop plane and proxy transform are measured; the PyBullet collider remains authoritative.
- Support-mesh scaling must match the final compiled proxy footprint and measured physical support plane. The maximum largest-to-smallest axis scale ratio is 1.9; an incompatible mesh is removed while sampling metadata, with a procedural support selected instead. Failed hash, bounds, texture, or plane checks still reject the render.
- Only the visible primary support may replace its analytic rendering. Environment walls, furniture, rails, and every collision record remain unchanged.
- Embedded PBR materials are preserved when audited. A geometry-only support may use the sampled Poly Haven support material declared in metadata.
- Procedural decor must remain outside the primary motion region and receives matching visible and collision boxes. Mesh environments use one reviewed static concave proxy.
- Changing only foreground or support appearance must preserve physics. Changing a scene visual profile may change incidental environment contacts because the environment is physical.
- A dynamic GLB may contain authored actions, NLA tracks, armatures, or modifiers. The renderer evaluates the asset at source frame 0, bakes the evaluated meshes, removes the source animation graph, and then applies only the immutable PhysSweep trajectory. Source animation must never be composed with simulated motion.

## Asset proxy rule

- Every curated Sketchfab asset has exactly one record in `asset_proxy_registry.json`; missing records are a build error.
- Proxy kinds are limited to dynamic rigid, static compound, support compound, static environment mesh, or explicit none.
- Dynamic collision is assembled only from reviewed analytic primitives declared in the record. A dynamic GLB triangle mesh is never used directly as runtime collision geometry. Audited zero-mass supports may use a frozen Blender-evaluated concave triangle mesh so visible holes, rails, and protrusions remain physical.
- Proxy primitives must match the backend geometry exactly: a sphere has one diameter on all three axes, and a cylinder has one shared x/y diameter plus height. Runtime type, dimensions, local position, and local orientation are compared for every child collider.
- Supports declare measured local usable surfaces. Whole-model bounds must not be substituted for a local tabletop or counter.
- Multi-component GLBs require an exact component partition. Every mesh is assigned a physical, visual-context, hidden-baked, or rejected role; an unclassified mesh is a render error.
- Support planes are selected from clustered upward-facing triangle area and continuity, not from the highest vertex. Concave supports require compound proxies and specialized motion profiles unless their full topology is represented by the generic sampler.
- A visual component without a physical proxy cannot remain in an interaction region. It must be hidden, moved to non-interacting context, or given its own proxy.
- Render-only and rejected records are not fallback candidates. Unsupported assets must be skipped visibly, never silently mapped to a generic table or cylinder.
- The same admitted record is shared by 1obj, 2obj, and 3obj pipelines.
- Trajectory QA compares discretely sampled distances with one backend-owned
  numerical tolerance. Free-fall lateral drift ends at the last pre-contact
  frame, so a physically valid collision impulse is never counted as airborne
  drift. Rebound-height tolerance is derived from gravity and output frame rate
  to account for a continuous peak falling between two stored frames.
- A proxy is admitted only after a deterministic physics probe and a three-state Blender overlay review.
- Asset identity does not imply scene compatibility. `asset_semantic_scene_rules.json` excludes game tables from generic object pairing and routes them to explicit billiards families.
- Support identity also constrains dynamic-object semantics. Curated support and
  support/prop entries select from named dynamic pools in
  `one_object_sampling_matrix.json`. Prop-bearing pools are specific to tray,
  tableware, or office-context semantics; the validation runner exercises every
  reachable support/prop/dynamic/profile combination rather than maintaining a
  second hard-coded candidate table.
- Every sampled static prop is instantiated as its declared zero-mass compound collider. Ordinary drop and push profiles route the moving object through a separate lane and reject any unplanned prop contact; a future intentional prop-impact profile must declare that contact explicitly.
- Prop-bearing environments currently admit straight clear-lane push only.
  Drop and diagonal motion remain available in generic and prop-free curated
  environments; they are excluded beside static props until a collision-aware
  path planner is part of the contract.
- A support capability is admitted per support/profile pair, never inferred from
  the presence of a usable plane. The twelve v2 support additions expose only
  `vertical_drop`; push, roll, and edge-exit remain disabled until a complete
  support/profile/dynamic compatibility probe passes.
- The billiards 1obj family contains one regulation cue ball. Its supported profiles are free rolling without rail contact and one rail impact followed by rebound. Static table, bed, and rails do not count as dynamic objects.
- The billiards v1 family uses three regulation-sized balls, requires a central ball-ball collision, and rejects rail or pocket contact because pocket sinking is not yet represented by the collision proxy.
- Asset scenes must bind a hashed HDRI plus hashed PBR floor and wall materials in metadata. A neutral fallback plane is not a valid background.
- Asset-scene metadata also binds color management, HDRI strength, light scaling, and every area light. The renderer may not inject private lights. One dominant key plus weak fill and rim lights preserves contact shadows without washing out light-colored supports.
- After material binding, the renderer measures the actual Base Color texture pixels for the dynamic object and visible support assets separately. This pre-render policy may adjust exposure, HDRI strength, and fill/rim energy for light-light, very-light-support, and dark-dark combinations. It may not inspect semantic names or use a rendered image, and every decision and measurement must be recorded in `render_record.json`.
- After the complete generic scene is bound, the renderer checks the metadata-declared first, middle, and final inspection frames in final display space. The rendered-frame policy shares its fixed luma, contrast, gradient, and clipping limits with the batch audit. It may adjust exposure only, by at most 0.35 EV per step and 0.70 EV total, with no more than two corrections. It may not change materials, lights, camera, physics, metadata, or asset selection, and it must record every probe. The retained probe frames replace the ordinary inspection renders, so a scene that passes immediately pays no additional frame-render count.
- Generic-scene key-light size and energy are derived from the dynamic object's physical footprint. Eevee contact-shadow bias is derived from physical thickness, so thin grounded objects are not erased by Blender's fixed default bias. The derived size, energy density, bias, search distance, and thickness are explicit metadata fields.

## Source Of Truth

- Generic matrix axes and sampling policy: `configs/one_object_sampling_rules.json`
- Engine, global contact calibration, and acceptance rules: `configs/pybullet_backend.json`
- Active PhysAssets identity, admitted visual, collision proxy, and nominal physical ranges: `configs/physassets_core_object_profiles.json`
- Full-candidate mesh, material, proxy, four-view, and visibility gate: `configs/object_visual_preflight.json`
- Per-object visual status and immutable source/admitted bindings: `configs/object_visual_curation.json`
- Reproducible visual repair recipes: `configs/object_visual_repairs.json`
- Support topology and dimensions: `configs/scene_kits.json`
- PBR material, HDRI, and texture-scale sampling: `configs/visual_sampling.json`
- Scene visual assets: `configs/scene_visual_profiles.json`
- Scene mesh backdrops and paired proxies: `configs/scene_mesh_profiles.json`
- Static environment collision manifest: `configs/visual_environment_collision_proxies.json`
- Real support visuals: `configs/support_mesh_profiles.json`
- Unified per-asset proxies: `configs/asset_proxy_registry.json`
- Per-asset component and scene decisions: `configs/asset_scene_composition.json`
- Foreground/support physical proxy catalog: `assets/proxies/catalog.json`
- Asset-to-scene semantics: `configs/asset_semantic_scene_rules.json`
- Decoupled motion/environment sampling: `configs/one_object_sampling_matrix.json`
- Active bundle: `configs/one_object_sampling_bundle.json`
- Runtime-validated supported scope: `configs/backend_capabilities.json`

Numerical ownership follows the thing being described: object profiles own per-object dimensions and nominal physical ranges, scene kits own support geometry, and `pybullet_backend.json` owns engine settings, global calibration, specialized initial states, and acceptance thresholds. Semantic files only declare admitted meanings and combinations.

The metadata JSON is the contract. Simulation may only read it. Camera solving and rendering happen after simulation and cannot change physical state.

Dataset manifests must hash both configuration dependencies and the implementation files declared by the active bundle or outer matrix. The active implementation boundary includes sampling, simulation, visual binding, and rendering. Each render record additionally stores the exact renderer hash. The bound manifest records the visual-binder and camera-rule hashes. Asset-registry summary counts must match the records before any scene is sampled.
The source manifest records the sampling entry point as `sampling_bundle_path`/`sampling_bundle_sha256` and the compiled camera rules as `rules_path`/`rules_sha256`. The visual binder reads and verifies only the frozen compiled-rule pair; it never substitutes a newer active rule file into an older batch.

## Decoupled Matrix

`one_object_sampling_matrix.json` samples motion before environment:

1. Allocate the eleven motion intents independently of asset coverage.
2. Allocate environment quotas independently.
3. Match each environment only to a compatible motion intent.
4. Sample a compatible physical proxy and then a visual asset.
5. Sample camera and lighting after the physical scene is fixed.

Changing environment weights must not change the sampled motion sequence.
Incompatible environments are excluded without changing the motion sequence.
The matcher uses a global compatibility-feasibility check, so it preserves exact
environment quotas whenever any legal assignment exists. The generic
environment admits every motion and receives the unmatched catch-all slots.

Identity-bearing environments keep narrow constraints. A billiards environment
admits only a regulation ball with free-roll or rail-rebound motion. The
workbench admits only reviewed clear-zone drop and long-axis push profiles.
Ordinary curated supports use named dynamic pools. These compatibility masks
prevent nonsensical Cartesian products without making asset coverage the owner
of the motion distribution.

1. Motion: eleven declared motion families and their subtypes.
2. Foreground object: 84 reviewed PhysAssets profiles plus specialized curated assets, each bound to an explicit rigid proxy. Upright objects use support-normal placement; declared rolling drums and rods use a side-on pose aligned with motion.
3. Support and interaction: four scene classes and twelve supports mapped to explicit floors, collision boxes, structures, trays, or ramps. Generic narrow tracks are excluded from v0.
4. Camera and framing: six view families and three framing profiles.
5. Appearance and lighting: curated Poly Haven materials, semantic contrast, room structure, and curated HDRIs.

Axes use deterministic balanced cycles. A 198-scene coverage batch visits every combination of eleven motion families and eighteen object profiles exactly once while balancing the remaining axes. A sample records every chosen axis value, seed, rule hash, backend hash, asset id, initial state, and expected-motion contract.

## Scene-Class Selection

The sampler first selects a scene class, then selects a compatible support within it:

- `ground_flat` (0.35): one continuous flush floor, never a floating floor patch.
- `raised_flat` (0.30): tables, benches, and counters with visible structure.
- `ground_feature` (0.20): ramps built from ground level.
- `raised_feature` (0.15): trays, pedestals, and raised ramps.

The weights are normalized only over classes that contain a support compatible with the selected motion. Drop and projectile motion use flat classes; slope motion uses feature classes; wall impact uses flat ground or raised supports; edge fall uses a raised flat support; and ramp-to-flat uses a ground ramp with an explicit landing surface. Sliding, rolling, and bouncing use flat classes plus declared raised features. This is a category-level conditional rule, never a scene-id exception.

## General Calculation Rules

- Simulation frequency is derived from object minimum extent and reference speed. One time step may cover at most 6% of that extent; frequency is clamped to 960-3840 Hz and rounded to an output-frame multiple. Generic, asset-proxy, and billiards branches use the same calculator.
- Collision detection runs before frame zero. Initial penetration may not exceed 0.5 mm.
- Runtime penetration is bounded by the stricter of 8 mm and 10% of the object's minimum extent. This prevents an absolute tolerance from accepting visibly deep overlap for plates, phones, magazines, and other thin objects.
- The solver uses 100 iterations. Integration parameters are scene-derived and recorded in immutable metadata; a renderer may not reinterpret them.
- Placement uses shape-aware support bounds and contact offsets.
- Static props are placed from their yaw-oriented physical AABB at a support
  edge. The dynamic lane is solved on the opposite edge from the oriented
  object footprint and must retain at least 2 cm of declared clearance. Dynamic
  yaw is sampled only from orientations that satisfy that same inequality, with
  zero yaw as the deterministic final candidate.
- Initial poses are placed above the exact support plane with a fixed clearance.
- Sliding speed is derived from target distance, friction, gravity, and target duration.
- Curated-asset push distance is a bounded fraction of the remaining shape-safe support distance. It is limited by the declared launch-speed cap; it is never a fixed distance reused across different support sizes.
- Pure sphere proxies use a rolling travel-time target rather than the Coulomb sliding-stop equation. Boxes and upright compound proxies keep the friction-derived sliding rule.
- Projectile speed is derived from ballistic flight time and target horizontal extent.
- Uphill speed is derived from slope angle, friction, gravity, and target climb distance.
- Downhill friction is bounded by the sampled ramp angle so the object can move.
- Rolling spheres and side-on cylinders receive coupled linear and angular velocity using `omega = normal cross tangent_velocity / radius`.
- Declared rolling motion must keep the measured linear/angular coupling ratio in [0.75, 1.35].
- Bounce restitution is sampled only for the declared bounce family.
- Wall impact uses an explicit wall collider and friction-compensated approach speed.
- Edge fall includes the object's directional footprint when calculating the distance and speed required to clear the support.
- Curated-asset edge exit uses the same principle: the launch speed is derived from remaining support distance, the yaw-conservative footprint radius, combined object/support friction, gravity, and a bounded safety margin. A fixed speed is not reused across different support widths.
- Ramp-to-flat geometry records an explicit landing collider whose top meets the computed low edge of the rotated ramp. A low ramp support is omitted when the ramp already rests near the floor.

Rules branch only on declared metadata categories. Concrete scene ids are never used as generation conditions. Visual profiles are selected with seeded randomness among compatible candidates while globally preferring the least-used profile, so small batches remain diverse and coverage batches cannot starve a legal profile.

The camera target first blends 70% primary motion with 30% of the complete simulated trajectory, then tries a primary-only target and progressively initial-biased targets when needed. The primary target is derived from the declared observation window, while the complete target always uses every simulated frame. It fits a local support patch, keeps at least 80% of primary trajectory centers inside the actual image, applies the motion-specific full-trajectory threshold, and permits limited late-motion exit. A horizontal support is anchored by the local contact region whose scale follows the observed motion span; a ramp transition is anchored by its high edge, low edge, and trajectory-local landing point. An edge transition is framed separately: the support edge, airborne path, first floor contact, and final settled pose are all mandatory, the view is 65 degrees oblique to the exit direction, and every fitted point keeps at least 7.5% image margin. The solver tests both sides of the exit and fails the sample when neither side satisfies the contract. Distant floor or tabletop boundaries remain soft context and are never mandatory framing anchors. The initial object must span at least 6% of the frame. Its sampled maximum starts at 38%, scales sublinearly for objects larger than the 0.18 m reference, and is capped at 50%; ground ballistic scenes use the stricter 18% cap. Distant legs and cabinets are not framing constraints. The declared focal length is preferred; 40, 36, and 32 mm are deterministic fallback candidates only when the current lens has no admissible pose.

## Acceptance

Every trajectory must remain finite, stay below speed and penetration limits, show the declared motion, and satisfy motion-specific checks such as support contact, ballistic apex, rolling coupling, downhill travel, uphill reversal, visible rebound, named wall contact, primary-support exit with floor contact, or named landing contact. The shared invariant audit also verifies that frame zero matches metadata; dynamic and static PyBullet parameters match their declarations; every primitive in a simple or compound collision proxy matches its declared type, dimensions, local position, and local orientation; and runtime principal inertia is finite and positive. Primitive inertia is additionally checked against its analytic value.

Unforced mechanical energy uses PyBullet's runtime principal inertia and vector gravity. Its numerical tolerance scales with object mass, characteristic extent, and initial energy rather than a fixed scene-independent floor. Airborne acceleration must fit gravity. Projectile positions are compared directly with `p(t) = p0 + v0*t + 0.5*g*t^2`, including three-dimensional RMS error, maximum error, and horizontal-velocity drift. Contact friction must remain inside the configured Coulomb cone, and declared kinetic-sliding cases must activate that limit. Bounce scenes compare the first observable rebound with the configured effective restitution. A terminal rest window is checked only when both linear and angular speeds are low throughout the window.

Curated asset scenes additionally enforce profile semantics:

- Drop profiles start airborne, make the declared support contact, move downward enough to be observable, and do not gain unexplained lateral drift.
- Resting and diagonal pushes begin on the support, retain sustained support contact, and travel in the declared projected direction. Diagonal motion must have displacement on both horizontal axes.
- Profiles that do not declare prop interaction must have zero static-prop contact frames.
- Edge exits require initial support, a sustained loss of support contact, subsequent ground contact, and measurable vertical drop.
- Workbench profiles must start inside a reviewed conservative interaction footprint. Long-axis pushes retain support contact; clear-zone drops use only the reviewed landing region.
- Billiards free roll may not touch a rail or reverse direction. Rail-rebound profiles must contact the named rail, reverse the normal velocity component, and retain a bounded rebound-speed ratio.

Leaving the support or frame after the primary interaction is allowed. Substitution, hidden forces after frame zero, and per-video repair are forbidden.

## Physics Scope

- Physics is authoritative for the declared primitive or compound proxy, not for every triangle of the render mesh. A visual asset is valid only after its proxy, usable support plane, scale, and component partition are reviewed.
- PyBullet evolves the state only from frame-zero position, orientation, velocity, gravity, contact, and material parameters. There are no post-frame-zero steering forces.
- In an isolated one-object scene under uniform gravity and Coulomb contact, changing mass alone normally does not change the trajectory. Mass becomes trajectory-identifiable only through mass-dependent forcing, drag, compliant coupling, or multi-object interaction. The sampler must not invent mass effects merely to create visible diversity.
