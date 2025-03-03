import numpy as np
from plotly import graph_objects as go


def plot_connection(x, y, color="white", name="conn"):
    return [
        go.Scatter3d(
            x=[x[i, 0], y[i, 0]],
            y=[x[i, 1], y[i, 1]],
            z=[x[i, 2], y[i, 2]],
            mode="lines",
            line={"color": color, "width": 2},
        )
        for i in range(x.shape[0])
    ]


def plot_mesh(mesh, color="lightblue", opacity=1.0, name="mesh"):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
    )


def plot_hand(verts, faces, color="lightpink", opacity=1.0):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
    )


def plot_contact_points(pts, grad, color="lightpink"):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        u=-grad[:, 0],
        v=-grad[:, 1],
        w=-grad[:, 2],
        anchor="tip",
        colorscale=[(0, color), (1, color)],
        sizemode="absolute",
        sizeref=0.2,
        opacity=0.5,
    )


def plot_point_cloud(pts, color="lightblue", name="pc", **kwargs):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        name=name,
        marker={"color": color, "size": 3},
        **kwargs,
    )


def plot_point_cloud_occ(pts, color_levels=None, opacity=1.0, size=3.0, cmap_fn=None):
    if cmap_fn is None:
        cmap_fn = lambda levels: [
            (f"rgb({int(255)},{int(255)},{int(255)})" if x <= 1e-12 else f"rgb({int(0)},{int(0)},{int(0)})") for x in levels.tolist()
        ]
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker={"color": cmap_fn(color_levels), "size": size, "opacity": opacity},
    )


def plot_point_cloud_cmap(pts, color_levels=None, name="pc"):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        name=name,
        marker={"color": color_levels, "size": 2, "opacity": 1},
    )
