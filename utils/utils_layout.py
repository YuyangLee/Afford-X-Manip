import numpy as np
import torch
from tqdm import trange, tqdm
import trimesh as tm


def optimize_layout_from_meshes(
    object_mesh_list: list[str],
    x_bound,
    y_bound,
    batch_size: int = 128,
    bound_expansion: float = 0.1,
    n_step_optim: int = 2000,
    device="cuda:0",
) -> np.array:
    meshes = [tm.load(m) for m in object_mesh_list]
    bounds = [m.bounds for m in meshes]
    all_bounds = np.array(bounds)[:, :, :2]
    return optimize_layout(all_bounds, x_bound, y_bound, batch_size, bound_expansion, n_step_optim, device)


def optimize_layout(
    all_bounds: np.array,
    x_bound,
    y_bound,
    batch_size: int = 128,
    bound_expansion: float = 0.1,
    n_step_optim: int = 2000,
    device="cuda:0",
) -> np.array:
    def denormalize_pos(pos):
        x, y = pos[..., 0], pos[..., 1]  # B x N_o
        x, y = (
            x * (x_bound[1] - x_bound[0]) + x_bound[0],
            y * (y_bound[1] - y_bound[0]) + y_bound[0],
        )  # , (theta % 1) * np.pi * 2
        return torch.stack([x, y], dim=-1)  # , theta

    def normalize_pos(pos):
        x, y = pos[..., 0], pos[..., 1]  # B x N_o
        x, y = (x - x_bound[0]) / (x_bound[1] - x_bound[0]), (y - y_bound[0]) / (y_bound[1] - y_bound[0])  # , (theta / (np.pi * 2) + 1) % 1
        return torch.stack([x, y], dim=-1)

    n_objects = len(all_bounds)

    obj_bounds = (
        torch.tensor(all_bounds, dtype=torch.float32, device=device)[..., :2].unsqueeze(0).tile([batch_size, 1, 1, 1])
    )  # B x N_obj x 2 x 2 (lo, up)
    object_pos_normalized = torch.rand(
        [batch_size, n_objects, 2],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    optimizer = torch.optim.Adam([object_pos_normalized], lr=1e-4)

    for i_optim in trange(n_step_optim):
        optimizer.zero_grad()

        # We don't consider the z-axis rotation for simplicity.
        object_pos = denormalize_pos(object_pos_normalized)  # B x N_o x 2
        curr_bound = obj_bounds + object_pos.unsqueeze(-2)

        # Check collision
        collision = torch.zeros([batch_size], dtype=torch.float32, device=device)
        for i_obj in range(n_objects):
            for j_obj in range(i_obj + 1, n_objects):
                bb_i, bb_j = curr_bound[:, i_obj], curr_bound[:, j_obj]

                xmin1, ymin1 = bb_i[:, 0, 0], bb_i[:, 0, 1]
                xmax1, ymax1 = bb_i[:, 1, 0], bb_i[:, 1, 1]
                xmin2, ymin2 = bb_j[:, 0, 0], bb_j[:, 0, 1]
                xmax2, ymax2 = bb_j[:, 1, 0], bb_j[:, 1, 1]

                inter_xmin = torch.max(xmin1, xmin2)
                inter_ymin = torch.max(ymin1, ymin2)
                inter_xmax = torch.min(xmax1, xmax2)
                inter_ymax = torch.min(ymax1, ymax2)

                inter_width = torch.relu(inter_xmax - inter_xmin + 0.01)
                inter_height = torch.relu(inter_ymax - inter_ymin + 0.01)

                ij_collision = inter_width * inter_height

                collision = collision + ij_collision

        bound = (
            torch.relu(x_bound[0] - curr_bound[..., 0, 0] + 0.01)
            + torch.relu(curr_bound[..., 1, 0] - x_bound[1] + 0.01)
            + torch.relu(y_bound[0] - curr_bound[..., 0, 1] + 0.01)
            + torch.relu(curr_bound[..., 1, 1] - y_bound[1] + 0.01)
        )
        bound = bound.sum(dim=-1)
        loss = (collision * 1.0 + bound * n_objects * 1.0).mean()

        loss.backward()
        optimizer.step()

        selected_idx = torch.where((collision < 1e-5) * (bound < 1e-3))[0]
        if i_optim % 50 == 49:
            tqdm.write(
                f"Optimizing: {i_optim}  Avg pntr: {collision.mean().item():.4f}  Avg bound: {bound.mean().item():.4f}  {len(selected_idx)}/{batch_size} accepted"
            )

        if len(selected_idx) >= 1:
            break

    object_pos = object_pos.detach().cpu().numpy()

    if len(selected_idx) == 0:
        return False, object_pos[torch.argmin(collision + bound)]

    selected_idx = selected_idx[0]
    object_pos = object_pos[selected_idx]

    return True, object_pos
