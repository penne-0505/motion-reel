"""S4 'dimension' — Blender (Eevee) scene, beats 14–20 of the reel (180 frames @ 60 fps).

Run:  flatpak run --filesystem=<this dir> org.blender.Blender -b -P scene3d.py -- [--preview N]
Every transform is computed per frame in Python and keyed, so timing is exact to the beat grid.
"""
import bpy, math, sys, os
from mathutils import Vector

ROOT = os.path.dirname(os.path.abspath(__file__))
FPS, FRAMES = 60, 180
BEAT_F = 30  # frames per beat

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
preview = None
if '--preview' in args:
    preview = [int(x) for x in args[args.index('--preview') + 1].split(',')]


def srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hexlin(h, a=1.0):
    h = h.lstrip('#')
    return tuple(srgb_to_lin(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)) + (a,)


INK, PAPER = hexlin('16161E'), hexlin('EDEAE3')
SIGNAL, COBALT, WHITE = hexlin('FF4D1C'), hexlin('2B4BFF'), hexlin('F6F4EF')

# ── reset ──
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, FRAMES
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
sc.render.use_motion_blur = True
sc.render.motion_blur_shutter = 0.5
ee = sc.eevee
for attr, val in [('taa_render_samples', 48), ('use_raytracing', True), ('use_shadows', True),
                  ('shadow_ray_count', 2), ('shadow_step_count', 8), ('motion_blur_steps', 2),
                  ('use_fast_gi', True), ('fast_gi_distance', 2.0)]:
    if hasattr(ee, attr):
        try: setattr(ee, attr, val)
        except Exception as e: print('skip', attr, e)

world = bpy.data.worlds.new('W'); sc.world = world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = hexlin('EDEAE3')
bg.inputs[1].default_value = 0.32


def material(name, col, rough=0.38, coat=0.25, spec=0.5):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = col
    p.inputs['Roughness'].default_value = rough
    if 'Coat Weight' in p.inputs: p.inputs['Coat Weight'].default_value = coat
    if 'Specular IOR Level' in p.inputs: p.inputs['Specular IOR Level'].default_value = spec
    return m


# ── cyclorama ──
def cyclorama():
    import bmesh
    me = bpy.data.meshes.new('cyc'); ob = bpy.data.objects.new('cyc', me)
    sc.collection.objects.link(ob)
    bm = bmesh.new()
    prof = []
    for k in range(40): prof.append((-30 + k * (38 / 39), 0.0))           # floor y -30 → 8
    R = 6.0
    for k in range(1, 25):
        a = k / 24 * math.pi / 2
        prof.append((8 + math.sin(a) * R, R - math.cos(a) * R))            # curve up
    for k in range(1, 6): prof.append((8 + R, R + k * 4))                  # wall
    xs = [-40 + i * 4 for i in range(21)]
    grid = [[bm.verts.new((x, y, z)) for (y, z) in prof] for x in xs]
    for i in range(len(xs) - 1):
        for j in range(len(prof) - 1):
            bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
    bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    ob.data.materials.append(material('cycm', PAPER, rough=0.85, coat=0.0, spec=0.2))
    return ob


cyclorama()


pivots = []


def finish(ob, half, mat):
    """Mesh origin stays at its centre (rotation); an empty at its base carries squash & position."""
    ob.data.materials.append(mat)
    for p in ob.data.polygons: p.use_smooth = True
    base = bpy.data.objects.new(ob.name + '_base', None); sc.collection.objects.link(base)
    ob.parent = base
    ob.location = (0, 0, half)
    pivots.append(base)
    return ob


objs = []
bpy.ops.mesh.primitive_cube_add(size=0.9)
cube = bpy.context.object
bev = cube.modifiers.new('bev', 'BEVEL'); bev.width = 0.09; bev.segments = 5
objs.append(finish(cube, 0.45, material('cobalt', COBALT)))
bpy.ops.mesh.primitive_torus_add(major_radius=0.42, minor_radius=0.18, major_segments=96, minor_segments=32)
objs.append(finish(bpy.context.object, 0.18, material('ink', INK, rough=0.3, coat=0.6)))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.52, segments=96, ring_count=48)
objs.append(finish(bpy.context.object, 0.52, material('signal', SIGNAL, rough=0.32, coat=0.35)))
bpy.ops.mesh.primitive_cone_add(radius1=0.48, depth=1.05, vertices=96)
cone = bpy.context.object
objs.append(finish(cone, 0.525, material('white', WHITE, rough=0.45)))
bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=0.9, vertices=96)
cyl = bpy.context.object
b2 = cyl.modifiers.new('bev', 'BEVEL'); b2.width = 0.05; b2.segments = 4; b2.limit_method = 'ANGLE'
objs.append(finish(cyl, 0.45, material('signal2', SIGNAL, rough=0.32, coat=0.35)))
for ob in (cube, cone, cyl):
    if hasattr(ob.data, 'set_sharp_from_angle'):
        ob.data.set_sharp_from_angle(angle=math.radians(35))

XS = [-2.6, -1.3, 0.0, 1.3, 2.6]

# ── lights ──
def area(name, loc, rot, energy, size, col=(1, 1, 1)):
    d = bpy.data.lights.new(name, 'AREA'); d.energy = energy; d.size = size; d.color = col
    o = bpy.data.objects.new(name, d); sc.collection.objects.link(o)
    o.location = loc; o.rotation_euler = [math.radians(a) for a in rot]
    return o


area('key', (-5, -5, 7), (48, 0, -42), 700, 5, (1.0, 0.97, 0.93))
area('fill', (6, -4, 3), (70, 0, 55), 160, 7, (0.9, 0.94, 1.0))
area('rim', (0, 6, 6), (-50, 0, 180), 420, 4)

# ── camera ──
cam_d = bpy.data.cameras.new('cam'); cam_d.lens = 50
cam_d.dof.use_dof = True; cam_d.dof.aperture_fstop = 4.0
cam = bpy.data.objects.new('cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam
tgt = bpy.data.objects.new('tgt', None); sc.collection.objects.link(tgt)
tc = cam.constraints.new('TRACK_TO'); tc.target = tgt; tc.track_axis = 'TRACK_NEGATIVE_Z'; tc.up_axis = 'UP_Y'
cam_d.dof.focus_object = tgt

# ── choreography (all in beats, lb = local beat 0..6) ──
def clamp01(x): return max(0.0, min(1.0, x))
def seg(b, a, c): return clamp01((b - a) / (c - a))
def e_io_cubic(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def e_in_expo(x): return 0 if x <= 0 else 2 ** (10 * x - 10)
def e_out_cubic(x): return 1 - (1 - x) ** 3
def lerp(a, b, t): return a + (b - a) * t


G = 81.6            # units / beat²
LAND = [0.25 + i * 0.4 for i in range(5)]
DROP_H = 5.0


def drop_state(lb, tl):
    """Height, vertical velocity and list of (impact_time, impact_speed)."""
    t0 = tl - math.sqrt(2 * DROP_H / G)
    if lb < t0: return None
    if lb < tl:
        dt = lb - t0
        return DROP_H - 0.5 * G * dt * dt, -G * dt, []
    v = G * (tl - t0)
    impacts, t, e = [(tl, v)], tl, 0.32
    while True:
        v *= e
        dur = 2 * v / G
        if v < 1.0 or lb < t + dur:
            if v < 1.0: return 0.0, 0.0, impacts
            dt = lb - t
            return v * dt - 0.5 * G * dt * dt, v - G * dt, impacts
        t += dur
        impacts.append((t, v))


def squash_from(impacts, lb, vref=28.6):
    s = 0.0
    for ti, vi in impacts:
        dt = lb - ti
        if dt >= 0: s += 0.38 * (vi / vref) * math.exp(-dt * 9) * math.cos(2 * math.pi * 2.3 * dt)
    return s


JUMP_T, JUMP_LAND, JUMP_H = 3.0, 4.0, 1.9


def jump_state(lb, i):
    d = i * 0.035
    t0, t1 = JUMP_T + d, JUMP_LAND + d
    if lb < t0 - 0.45: return 0.0, 0.0, 0.0
    if lb < t0:  # anticipation squash
        u = seg(lb, t0 - 0.45, t0)
        return 0.0, 0.28 * math.sin(u * math.pi / 2) ** 2, 0.0
    if lb < t1:
        u = (lb - t0) / (t1 - t0)
        z = 4 * JUMP_H * u * (1 - u)
        v = 4 * JUMP_H * (1 - 2 * u)  # per unit u
        return z, -0.10 * abs(v) / 4, u   # negative squash = stretch
    dt = lb - t1
    return 0.0, 0.34 * math.exp(-dt * 8) * math.cos(2 * math.pi * 2.2 * dt), 1.0


def hop(lb, i):
    t0 = 4.45 + i * 0.13
    if t0 <= lb < t0 + 0.32:
        u = (lb - t0) / 0.32
        return 0.38 * 4 * u * (1 - u)
    return 0.0


SPIN = [  # (axis, turns) during the jump
    ('X', 0.25), ('X', 0.5), ('Z', 1.0), ('Z', 1.0), ('X', 0.5)]


def pose(i, lb):
    ob, pv = objs[i], pivots[i]
    st = drop_state(lb, LAND[i])
    if st is None:
        pv.location = (XS[i], 0, 40); pv.scale = (1, 1, 1); return
    z, vz, impacts = st
    s = squash_from(impacts, lb)
    if lb < LAND[i]:
        s = -min(0.22, abs(vz) * 0.0075)   # stretch in free fall
    jz, js, ju = jump_state(lb, i)
    z += jz + hop(lb, i)
    s += js
    sz = 1 - s
    sxy = 1 / math.sqrt(max(sz, 0.3))
    pv.location = (XS[i], 0, z)
    pv.scale = (sxy, sxy, sz)
    ax, turns = SPIN[i]
    ang = e_io_cubic(ju) * turns * 2 * math.pi
    base_rz = [0.35, 0, 0, 0, 0][i]
    if ax == 'X': ob.rotation_euler = (ang, 0, base_rz)
    else: ob.rotation_euler = (0, 0, base_rz + ang)


def camera(lb):
    push = e_io_cubic(seg(lb, 0, 4.2))
    dist = lerp(12.4, 9.6, push) + 1.6 * math.sin(math.pi * seg(lb, 2.9, 4.2))
    orbit = math.radians(lerp(-10, 26, e_io_cubic(seg(lb, 3.6, 5.6))))
    h = lerp(2.3, 1.25, e_io_cubic(seg(lb, 2.5, 5.6)))
    cam.location = (math.sin(orbit) * dist, -math.cos(orbit) * dist, h)
    jf = math.sin(math.pi * seg(lb, 2.95, 4.1))
    ty = 0.55 + 1.15 * jf  # follow the jump
    # whip pan: throw the target sideways, fast
    w = e_in_expo(seg(lb, 5.5, 6.0))
    tx = 0.35 * e_io_cubic(seg(lb, 3.6, 5.6))
    tgt.location = (tx + lerp(0, 16, w) * math.cos(orbit), lerp(0, 16, w) * math.sin(orbit), ty)
    cam.data.dof.aperture_fstop = lerp(5.6, 3.2, push)


for f in range(1, FRAMES + 1):
    lb = (f - 1) / BEAT_F
    sc.frame_set(f)
    for i in range(5):
        pose(i, lb)
        pivots[i].keyframe_insert('location', frame=f)
        pivots[i].keyframe_insert('scale', frame=f)
        objs[i].keyframe_insert('rotation_euler', frame=f)
    camera(lb)
    cam.keyframe_insert('location', frame=f)
    tgt.keyframe_insert('location', frame=f)
    cam_d.dof.keyframe_insert('aperture_fstop', frame=f)

sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'
out = os.path.join(ROOT, 'render3d' if preview is None else 'render3d_preview')
os.makedirs(out, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, 'scene3d.blend'))
if preview is None:
    sc.render.filepath = os.path.join(out, '')
    bpy.ops.render.render(animation=True)
else:
    for f in preview:
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, f'{f:04d}.png')
        bpy.ops.render.render(write_still=True)
