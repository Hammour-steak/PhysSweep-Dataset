# PhysSweep Rulebook

## Architecture rule

Concrete objects and scenes belong in profiles and scene kits. Compatibility belongs in `compatibility.json`. Python may branch only on reusable behavior types; it must not branch on concrete ids. The active bundle and full boundary are documented in `PHYSWEEP_SAMPLING_ARCHITECTURE.md`.

## Camera rule

The active generic solver is `motion_structure_camera_v12`.

- Camera semantics have two independent parts. The motion observation intent selects the key temporal window; the structure context selects the physical anchors that explain the interaction.
- The only observation intents are `surface_travel`, `contact_event`, `ballistic_arc`, and `support_transition`. The only structure contexts are `horizontal_surface`, `inclined_surface`, `impact_boundary`, `edge_and_landing`, and `ramp_and_landing`.
- `expected_motion` contains physics acceptance semantics only. It must not contain camera fields. The compiled `camera_request.observation` is the sole camera contract.
- Surface travel observes the start and main travel interval. Contact events observe the approach, first required contact, and a bounded post-contact interval. Ballistic motion observes launch, apex, and first contact. Support transitions observe both sides of the structural boundary.
- Structural anchors explain the interaction: a horizontal support boundary, both ends of a slope, an impact surface, an edge plus landing point, or a ramp plus landing surface. Their frame visibility and environment-occlusion thresholds are hard constraints selected by structure context. Horizontal, inclined, and edge structures require at least three quarters of their anchors unoccluded; ramp-to-landing requires four fifths; impact boundaries require one half. The synthetic camera target is a search aid, not a physical entity, and is diagnostic only.
- A solid wedge is represented by all four outer slope corners. `ramp_and_landing` adds the trajectory-local landing contact. The remote bounds of a large environment floor are never structural anchors.
- Camera minimum distance is derived from support size. Motion intent supplies the permitted elevation range, target blend, distance allowance, and partial-exit policy.
- `ground_flat` intersects the motion-specific elevation range with the shared 18-28 degree ground corridor. This preserves a wall, column, or environment boundary as scale context instead of filling the frame with an undifferentiated floor; raised supports and inclined structures retain their own ranges.
- The solver starts with the declared lens, normally 44 mm. Only when no pose satisfies every framing constraint may it try 40, 36, then 32 mm; focal fallback never relaxes object size, initial visibility, trajectory coverage, context, or occlusion thresholds.
- Object readability is evaluated over the observed interval, not only at frame zero. The median projected AABB span must meet the intent threshold; the initial AABB must still span at least 6%, remain fully visible, and keep its center inside the declared margin.
- At least 80% of observed trajectory centers remain inside the actual image. Full-trajectory center coverage is also measured against the exact `[0, 1]` image bounds; its motion-specific threshold controls permitted late exit. At least 90% of primary-motion samples and the declared fraction of the full trajectory remain unoccluded.
- Visible support geometry marked `occludes_camera` and every visible, collision-enabled environment box block the camera. This includes walls, cabinets, decor, legs, impact boundaries, and rails; the primary environment floor is not duplicated as an environment box.
- Environment walls, decor, set pieces, and mesh proxies are frozen before simulation. The camera solver must preserve them and keep the required trajectory fractions and structure anchors unoccluded; it may not delete, move, or disable a physical environment collider to improve composition.
- Partial exit after the primary interaction is allowed when the observation intent permits it. The solver does not pull back merely to retain an unimportant trajectory tail.
- Asset-proxy framing follows the same intent/structure contract and expands the observed path by the dynamic asset's canonical extent plus static-prop bounds.
- Generic and specialized cameras are separate mechanisms. The generic matrix searches geometry-derived poses and uses the scene seed to choose among near-equivalent admissible azimuth/elevation tiers. Specialized scene profiles such as billiards, edge exit, and workbench motion select from their own reviewed view pools in `visual_sampling.json`; passive pinball uses the geometry-bound front-oblique camera declared by its own backend configuration.
- A specialized view pool belongs to a reusable scene profile, never to a concrete asset id. Changing one profile cannot alter generic-matrix views or another specialized profile. Both mechanisms still enforce the same final framing, trajectory-visibility, structure-anchor, and occlusion contracts.
- When a static prop can occlude the moving object, the camera keeps the seed order inside the blocker-safe half of the profile pool. Occlusion safety filters admissible views; it does not replace seeded view selection with one extreme angle.
- Billiards profiles share the reviewed yaw pool `[-30, -15, 0, 15, 30]` degrees. A 15-degree minimum spacing removes visually redundant cross-profile angles while preserving left, frontal, and right observations; profile-specific elevation pools retain low, medium, and high views.
- Camera selection is deterministic: identical metadata, rules, implementation hashes, and seed produce the same view.

The generic solver remains one implementation. Motion and structure change its objective and anchors; specialized profiles provide bounded view choices without introducing concrete-scene patches.

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
- The passive-pinball family contains one unforced sphere and one exact analytic static fixture. Dense and offset profiles differ only through declared seeded launch offsets; both use the same board, rails, peg field, catch geometry, material rules, and camera mechanism. Active mechanisms and per-scene fixture edits are forbidden.
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

Numerical ownership follows the thing being described: object profiles own
per-object dimensions and nominal physical ranges, scene kits own support
geometry, `pybullet_backend.json` owns generic/asset/billiards engine rules, and
`passive_pinball_backend.json` owns the pinball fixture, material, engine,
camera, and acceptance rules. Semantic files only declare admitted meanings
and combinations.

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
passive-pinball environment admits only its declared one-ball gravity descent.
It replaces complete generic drop slots without changing the outer motion
quota. The workbench admits only reviewed clear-zone drop and long-axis push
profiles.
Ordinary curated supports use named dynamic pools. These compatibility masks
prevent nonsensical Cartesian products without making asset coverage the owner
of the motion distribution.

1. Motion: eleven declared motion families and their subtypes.
2. Foreground object: 84 reviewed PhysAssets profiles plus specialized curated assets, each bound to an explicit rigid proxy. Upright objects use support-normal placement; declared rolling drums and rods use a side-on pose aligned with motion.
3. Support and interaction: four scene classes and twenty-two supports mapped to explicit floors, collision boxes, structures, trays, corridors, or ramps. Generic narrow tracks are excluded from v0.
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

- Long-distance flat structures declare their admitted motions, visual
  environment categories, and maximum planar trajectory distance in the scene
  kit. The common sliding and rolling rules derive motion from those bounds;
  ballistic families retain their own height-and-flight-time range.
- Directional structures declare a motion axis. The sampler projects a sampled
  heading onto that axis before deriving the initial state and records the
  effective heading in metadata.
- `long_corridor` has two visible, camera-occluding PyBullet wall colliders.
  Corridor walls are physical structure, not render-only decoration. Its visual
  environment is restricted to the minimal shell so unrelated room set pieces
  cannot occupy the motion lane.
- Long tables are raised supports with explicit legs or cabinet structure and a
  separate environment floor. Their x-axis is the motion axis, so long motion
  uses the available tabletop length without inventing a diagonal path that
  immediately exits the narrow side. Wood-table, lab-bench, and kitchen-counter
  variants admit only semantically matching visual environments.
- Horizontal-motion cameras on long raised supports must keep five structural
  anchors visible: the support center, two long-axis anchors spanning the
  declared useful travel, and two short-axis anchors showing tabletop width.
  This preserves scene readability without forcing the entire table into frame.
- Support transitions use distinct physical colliders: ground ramps transition
  to `landing_surface` or the flush `environment_floor`, raised tables and the
  low pedestal transition to `environment_floor`. Platform-to-platform motion
  is not declared in v0 because it requires a separate lower-platform contact
  contract; a platform is never relabeled as a floor to reuse edge-fall code.
- Every admitted support transition compiles exactly one immutable
  `transition_contract`. It owns the source and destination collider ids,
  boundary point, outward horizontal direction, source and destination heights,
  height drop, intermediate contact phase, and required contact sequence.
  Motion derivation, camera focus, and trajectory QA read that same contract;
  they may not infer a second destination from motion names or scene ids.
- `raised_edge_to_floor` requires an airborne interval between source and
  destination contact. `incline_to_horizontal` requires continuous contact at
  a flush boundary. Both require source-before-destination contact, at least one
  destination-only frame, and no source recontact after the transition.
- Procedural room walls use a dynamic-clearance lower bound when initial motion
  points toward the wall. For ramp transitions, the post-slope speed bound is
  derived from projected initial speed plus the gravitational potential drop
  implied by slope angle and declared downhill travel. That speed times clip
  duration, the object's planar radius, and a fixed safety margin define the
  clearance. The wall, decor, and collision proxies move together.
- Large procedural environments declare a symmetric clear motion lane. Every
  side set piece must remain outside that lane after accounting for its physical
  half width. Warehouse, garage, and long-office shells use paired visible and
  collision geometry; they add context without becoming undeclared obstacles.
- Moving samples also define a conservative planar motion capsule from initial
  velocity, clip duration, object radius, and ramp energy gain. Any procedural
  side set piece intersecting that capsule is moved along the motion normal;
  its visible geometry and collider share the exact same shift.
- Camera context applies to both reviewed mesh environments and procedural room
  shells. Large rooms use a wider focal cap and a modest target offset toward
  the room interior so environment structure remains legible without shrinking
  the foreground object below the common readability floor.
- Non-spherical ramp transitions include a fixed exit-speed margin above the
  ideal Coulomb stopping calculation. The margin covers impact and rotational
  losses at the ramp-to-floor seam while preserving the declared target travel.
- Long ramp-to-floor structure requires an object characteristic extent of at
  least `0.12 m`. Spheres and cylinders use diameter; cuboids use their second
  largest extent. Incompatible small objects are resampled into shorter motion
  contexts rather than shrinking them below the common camera readability floor.

- Simulation frequency is derived from object minimum extent and reference speed. One time step may cover at most 6% of that extent; frequency is clamped to 960-3840 Hz and rounded to an output-frame multiple. Generic, asset-proxy, and billiards branches use this calculator. Passive pinball freezes its separately reviewed 3840 Hz rate in metadata.
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
- Sliding speed is derived from target distance and target duration. If the
  launch-speed bound activates, friction is recomputed from the bounded speed
  and target stopping distance rather than from the unreachable launch speed.
- Curated-asset push distance is a bounded fraction of the remaining shape-safe support distance. It is limited by the declared launch-speed cap; it is never a fixed distance reused across different support sizes.
- Pure sphere proxies use a rolling travel-time target rather than the Coulomb sliding-stop equation. Boxes and upright compound proxies keep the friction-derived sliding rule.
- Projectile speed is derived from ballistic flight time and target horizontal extent.
- Uphill speed is derived from slope angle, friction, gravity, and target climb distance.
- Downhill friction is bounded by the sampled ramp angle so the object can move.
- Ramp families have non-overlapping geometric semantics: long shallow ramps are 8-12 degrees, standard and channel ramps are 12-18 degrees, and short steep ramps are 20-30 degrees. Geometry compilation rejects a scene kit outside its declared family range.
- One semantic support may declare deterministic geometry variants. `ground_ramp_long_shallow` cycles through standard, extended-landing, and wide-gentle structures; the chosen dimensions, slope height, landing length, and variant ID are frozen before physics and camera derivation. Variants never create new motion semantics or bypass compatibility rules.
- Procedural ramps render as solid wedges. The inclined top uses `support_surface`; the bottom, high end, and triangular sides use `support_structure` so the physical slope remains visually distinguishable from a flat texture patch.
- Rolling spheres and side-on cylinders receive coupled linear and angular velocity using `omega = normal cross tangent_velocity / radius`.
- Declared rolling motion must keep the measured linear/angular coupling ratio in [0.75, 1.35].
- Bounce restitution is sampled only for the declared bounce family.
- Wall impact uses an explicit wall collider and friction-compensated approach speed.
- Edge fall includes the object's directional footprint when calculating the distance and speed required to clear the support.
- Curated-asset edge exit uses the same principle: the launch speed is derived from remaining support distance, the yaw-conservative footprint radius, combined object/support friction, gravity, and a bounded safety margin. A fixed speed is not reused across different support widths.
- Ramp-to-flat geometry records an explicit landing collider whose top meets the computed low edge of the rotated ramp. A low ramp support is omitted when the ramp already rests near the floor. The transition contract validates the declared destination height against the actual collider top.
- Ramp-to-flat motion must contact the ramp before the landing, reach a landing-only frame, and never contact the ramp again afterward. Its minimum post-transition travel is a bounded fraction of the declared landing length. Non-spherical launch speed is derived from that distance, contact friction, gravity, slope angle, and a shape-family transition-loss margin; it is not a scene-specific constant.

Rules branch only on declared metadata categories. Concrete scene ids are never used as generation conditions. Visual profiles are selected with seeded randomness among compatible candidates while globally preferring the least-used profile, so small batches remain diverse and coverage batches cannot starve a legal profile.

The camera target first blends 70% primary motion with 30% of the complete simulated trajectory, then tries a primary-only target and progressively initial-biased targets when needed. Long transition contexts additionally try the joint bounding center of primary trajectory samples and required structure anchors; this target is a framing fallback and does not alter physics. The primary target is derived from the declared observation window, while the complete target always uses every simulated frame. It fits a local support patch, keeps at least 80% of primary trajectory centers inside the actual image, applies the motion-specific full-trajectory threshold, and permits limited late-motion exit. A horizontal support is anchored by the local contact region whose scale follows the observed motion span; a ramp transition is anchored by all four wedge corners plus the trajectory-local landing point. An edge transition is framed separately: the support edge, airborne path, first floor contact, and final settled pose are mandatory, the view is 65 degrees oblique to the exit direction, and every fitted point keeps at least 7.5% image margin. The solver tests both sides of the exit and fails the sample when neither side satisfies the contract. Distant floor or tabletop boundaries remain soft context and are never mandatory framing anchors. The initial object must be fully visible and span at least 6% of the frame. Its sampled maximum starts at 38%, scales sublinearly for objects larger than the 0.18 m reference, and is capped at 50%; ground ballistic scenes use the stricter 18% cap. Distant legs and cabinets are not framing constraints. The declared focal length is preferred; 40, 36, 32, and 28 mm are deterministic fallback candidates only when wider framing is required and every object-size, trajectory, anchor, and occlusion gate still passes. Inclined and ramp-transition intents may search up to 4 m beyond their derived minimum, capped at 6 m, so long ramps can preserve every physical anchor without shrinking the object below the declared readability floor.

## Acceptance

Every trajectory must remain finite, stay below speed and penetration limits, show the declared motion, and satisfy motion-specific checks such as support contact, ballistic apex, rolling coupling, downhill travel, uphill reversal, visible rebound, named wall contact, or the shared support-transition contract. The shared invariant audit also verifies that frame zero matches metadata; dynamic and static PyBullet parameters match their declarations; every primitive in a simple or compound collision proxy matches its declared type, dimensions, local position, and local orientation; and runtime principal inertia is finite and positive. Primitive inertia is additionally checked against its analytic value.

Environment collision proxies always remain active. A motion may contact only the support or environment collider named by its metadata contract. For `ramp_to_flat`, contact with incidental walls, baseboards, decor, or set pieces rejects the candidate; deterministic slot retry keeps the requested motion distribution while replacing that candidate.

Unforced mechanical energy uses PyBullet's runtime principal inertia and vector gravity. Its numerical tolerance scales with object mass, characteristic extent, and initial energy rather than a fixed scene-independent floor. Airborne acceleration must fit gravity. Projectile positions are compared directly with `p(t) = p0 + v0*t + 0.5*g*t^2`, including three-dimensional RMS error, maximum error, and horizontal-velocity drift. Contact friction must remain inside the configured Coulomb cone, and declared kinetic-sliding cases must activate that limit. Bounce scenes compare the first observable rebound with the configured effective restitution. A terminal rest window is checked only when both linear and angular speeds are low throughout the window.

Curated asset scenes additionally enforce profile semantics:

- Drop profiles start airborne, make the declared support contact, move downward enough to be observable, and do not gain unexplained lateral drift.
- Resting and diagonal pushes begin on the support, retain sustained support contact, and travel in the declared projected direction. Diagonal motion must have displacement on both horizontal axes.
- Profiles that do not declare prop interaction must have zero static-prop contact frames.
- Edge exits require initial support, a sustained loss of support contact, subsequent ground contact, and measurable vertical drop.
- Workbench profiles must start inside a reviewed conservative interaction footprint. Long-axis pushes retain support contact; clear-zone drops use only the reviewed landing region.
- Billiards free roll may not touch a rail or reverse direction. Rail-rebound profiles must contact the named rail, reverse the normal velocity component, and retain a bounded rebound-speed ratio.
- Passive pinball must contact the declared minimum number of distinct pegs,
  enter the catch region, remain within fixture-local bounds, and satisfy hard
  penetration, speed, and unforced-energy limits. Visibility never substitutes
  for these physical checks.

Leaving the support or frame after the primary interaction is allowed. Substitution, hidden forces after frame zero, and per-video repair are forbidden.

## Physics Scope

- Physics is authoritative for the declared primitive or compound proxy, not for every triangle of the render mesh. A visual asset is valid only after its proxy, usable support plane, scale, and component partition are reviewed.
- PyBullet evolves the state only from frame-zero position, orientation, velocity, gravity, contact, and material parameters. There are no post-frame-zero steering forces.
- In an isolated one-object scene under uniform gravity and Coulomb contact, changing mass alone normally does not change the trajectory. Mass becomes trajectory-identifiable only through mass-dependent forcing, drag, compliant coupling, or multi-object interaction. The sampler must not invent mass effects merely to create visible diversity.
