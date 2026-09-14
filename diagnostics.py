"""On-demand solver diagnostics inspired by Keyframe's FMM explanation.

Diagnostic tree is a spatial partition, not a claim about CUDA exact traversal.
Force errors use a deterministic sample against all sources, O(sample_count*N).
"""
import math


def force_sample(sim, count=64):
    n = len(sim.bodies)
    if not n:
        return [], 0.0
    chosen = sorted(set(round(i*(n-1)/max(1, min(n, count)-1)) for i in range(min(n, count))))
    approx = sim.accelerations()
    points = []
    error2 = norm2 = 0.0
    for i in chosen:
        b = sim.bodies[i]
        ax = ay = 0.0
        for j, source in enumerate(sim.bodies):
            if i == j:
                continue
            dx, dy = source.x-b.x, source.y-b.y
            r2 = dx*dx+dy*dy+sim.softening**2
            f = sim.G*source.mass/(r2*math.sqrt(r2))
            ax += f*dx
            ay += f*dy
        if sim.background:
            bx, by = sim.background.acceleration(b.x, b.y, sim.G)
            ax += bx
            ay += by
        px, py = approx[i]
        e = math.hypot(px-ax, py-ay)
        norm = math.hypot(ax, ay)
        error2 += e*e
        norm2 += norm*norm
        points.append((b.x, b.y, px, py, e/max(norm, 1e-20)))
    return points, math.sqrt(error2/max(norm2, 1e-40))


def tree_boxes(sim, max_depth=6):
    root = sim._build_quadtree()
    stack = [root] if root else []
    boxes = []
    while stack:
        node = stack.pop()
        boxes.append((node.cx-node.half, node.cy-node.half,
                      node.cx+node.half, node.cy+node.half, node.depth))
        if node.children and node.depth < max_depth:
            stack.extend(c for c in node.children if c.mass or c.indices)
    return boxes
