from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from kornia.utils import tensor_to_image

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.util.geometry import canonicalise_222_rotation, canonicalise_origin
from crystalsizer3d.util.utils import init_tensor

device = torch.device('cuda')

TEST_CRYSTALS = {
    'alpha': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [
            0.3887779414653778, 1.024718999862671, 0.745987057685852, 0.7828164100646973, 1.0540395975112915,
            0.8920794725418091, 0.29156485199928284, 0.9565051794052124, 0.9716991782188416, 0.9008828401565552,
            0.8119459748268127, 0.8231415152549744, 0.8070471882820129, 1.0333278179168701, 0.9691855311393738,
            0.7162739038467407, 1.1065247058868408, 1.1049960851669312, 1.1667091846466064, 0.9211059808731079,
        ],
        'scale': 0.8415803909301758,
        'rotation': [0.006333562545478344, -0.023195751011371613, -4.9023756980896],
        "origin": [
            0.0222465842962265,
            -0.03906448185443878,
            0.25648394227027893
        ],
        'material_ior': 1.725118637084961,
        'material_roughness': 0.1283971518278122
    },
}


def random_axis_angle():
    v = torch.randn(3)
    v = v / v.norm() * torch.rand(1).sqrt() * torch.pi
    return v


def compose_axis_angle_rotations(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Return axis-angle of Ra ∘ Rb (apply Rb, then Ra)."""
    return rotation_matrix_to_axis_angle(
        (axis_angle_to_rotation_matrix(Ra[None]) @ axis_angle_to_rotation_matrix(Rb[None]))[0]
    )


def canonicalise_crystal(crystal: Crystal):
    """
    Canonicalise the crystal's rotation and distances.
    """
    re_init = False
    R, dists = canonicalise_222_rotation(
        crystal.rotation, crystal.distances, crystal.miller_indices, crystal.rotation_mode
    )
    if not torch.allclose(R, crystal.rotation):
        print("Canonical rotation found:", R)
        crystal.rotation.data = R
        crystal.distances.data = dists
        re_init = True
    else:
        print("No rotation canonicalisation needed.")

    origin, dists = canonicalise_origin(
        origin=crystal.origin,
        distances=crystal.distances,
        lattice_unit_cell=crystal.lattice_unit_cell,
        lattice_angles=crystal.lattice_angles,
        miller_indices=crystal.miller_indices,
        scale=crystal.scale,
        rotation=crystal.rotation,
        rotation_mode=crystal.rotation_mode,
    )
    if not torch.allclose(origin, crystal.origin) or not torch.allclose(dists, crystal.distances):
        print("Canonical origin found:", origin)
        crystal.origin.data = origin
        crystal.distances.data = dists
        re_init = True
    else:
        print("No origin canonicalisation needed.")

    if re_init:
        crystal._init()


def show_projected_image(which='alpha'):
    """
    Show the projected crystal wireframe.
    """
    image_size = (1000, 1000)
    assert which in TEST_CRYSTALS
    crystal_spec = TEST_CRYSTALS[which]
    pi_z = torch.tensor([0., 0., torch.pi])
    R0 = random_axis_angle() * 10
    R = compose_axis_angle_rotations(R0, pi_z)

    print("Original rotation:", R)
    print("Original origin:", crystal_spec['origin'])
    crystal_spec['rotation'] = R.tolist()
    crystal = Crystal(**crystal_spec)
    crystal.to(device)
    zoom = 0.5

    projector = Projector(crystal, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    im1 = tensor_to_image(projector.image)

    crystal2 = crystal.clone()
    crystal2.to(device)
    canonicalise_crystal(crystal2)

    projector = Projector(crystal2, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    im2 = tensor_to_image(projector.image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(im1)
    axes[0].set_title('Original Crystal')
    axes[0].axis('off')
    axes[1].imshow(im2)
    axes[1].set_title('Canonicalised Crystal')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Print the miller indices and distances
    print("Distances:")
    distances = [f"{d1:.3f} -> {d2:.3f}{'*' if d1 != d2 else ''}"
                 for d1, d2 in zip(crystal.distances.tolist(), crystal2.distances.tolist())]
    pprint(distances)
    print("Rotations:")
    rotations = [f"{r1:.3f} -> {r2:.3f}" for r1, r2 in zip(crystal.rotation.tolist(), crystal2.rotation.tolist())]
    pprint(rotations)

    # Check that the vertices are the same
    vertices1 = crystal.vertices
    vertices2 = crystal2.vertices
    v_dists = torch.cdist(vertices1, vertices2)
    if v_dists.amin(dim=0).max() > 1e-3 or v_dists.amin(dim=1).max() > 1e-3:
        print("Vertices are different.")
        print("Minimum distance between vertices 1->2:")
        pprint(v_dists.amin(dim=0).tolist())
        print("Minimum distance between vertices 2->1:")
        pprint(v_dists.amin(dim=1).tolist())
    else:
        print("Vertices are the same.")


def check_adjust_origin(which='alpha'):
    """
    Check that adjusting the origin works correctly.
    """
    image_size = (1000, 1000)
    assert which in TEST_CRYSTALS
    crystal_spec = TEST_CRYSTALS[which]
    pi_z = torch.tensor([0., 0., torch.pi])
    # pi_z = torch.tensor([0., 0., 0.])
    R0 = random_axis_angle() * 10
    # R0 = torch.zeros(3)
    R = compose_axis_angle_rotations(R0, pi_z)

    print("Original rotation:", R)
    print("Original origin:", crystal_spec['origin'])
    crystal_spec['rotation'] = R.tolist()
    crystal = Crystal(**crystal_spec, dtype=torch.float64, merge_vertices=True)
    crystal.to(device)
    zoom = 0.5

    projector = Projector(crystal, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    im1 = tensor_to_image(projector.image)

    crystal2 = crystal.clone()
    crystal2.to(device)
    new_origin = crystal2.origin + torch.randn(3).to(device).to(torch.float64) / 10
    print("Adjusting origin to:", new_origin)
    print(f"Origin change magnitude: {(new_origin - crystal2.origin).norm():.3f}")
    crystal2.adjust_origin(new_origin=new_origin, verify=False)
    crystal2.build_mesh()

    projector = Projector(crystal2, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    im2 = tensor_to_image(projector.image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(im1)
    axes[0].set_title('Original Crystal')
    axes[0].axis('off')
    axes[1].imshow(im2)
    axes[1].set_title('Canonicalised Crystal')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Print the miller indices and distances
    print("Distances:")
    distances = [f"{d1:.3f} -> {d2:.3f}{'*' if d1 != d2 else ''}"
                 for d1, d2 in zip(crystal.distances.tolist(), crystal2.distances.tolist())]
    pprint(distances)
    print("Rotations:")
    rotations = [f"{r1:.3f} -> {r2:.3f}" for r1, r2 in zip(crystal.rotation.tolist(), crystal2.rotation.tolist())]
    pprint(rotations)
    print("Origins:")
    origins = [f"{o1:.3f} -> {o2:.3f}{'*' if o1 != o2 else ''}"
               for o1, o2 in zip(crystal.origin.tolist(), crystal2.origin.tolist())]
    pprint(origins)

    # Check that the vertices are the same
    vertices1 = crystal.vertices
    vertices2 = crystal2.vertices
    v_dists = torch.cdist(vertices1, vertices2)
    if v_dists.amin(dim=0).max() > 1e-3 or v_dists.amin(dim=1).max() > 1e-3:
        print("Vertices are different.")
        print("Minimum distance between vertices 1->2:")
        pprint(v_dists.amin(dim=0).tolist())
        print("Minimum distance between vertices 2->1:")
        pprint(v_dists.amin(dim=1).tolist())
    else:
        print("Vertices are the same.")


if __name__ == '__main__':
    # show_projected_image('alpha')
    check_adjust_origin('alpha')
