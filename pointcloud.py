import numpy as np
from PIL import Image


def depth_to_pointcloud(
    depth,
    image=None,
    mask=None,
    stride=4
):
    """
    Converte mapa de profundidade em nuvem de pontos 3D.
    """

    # Depth precisa ser uma matriz 2D
    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise ValueError(
            f"Formato de depth inesperado: {depth.shape}"
        )

    height, width = depth.shape

    # Grid de pixels
    y, x = np.mgrid[
        0:height:stride,
        0:width:stride
    ]

    z = depth[
        ::stride,
        ::stride
    ]

    # Pontos válidos
    valid = np.isfinite(z)

    # ==========================================
    # MÁSCARA DO OBJETO
    # ==========================================

    if mask is not None:

        if isinstance(mask, Image.Image):

            mask_image = mask

        else:

            mask_array = np.asarray(mask)

            # Se vier RGB/RGBA, converte corretamente
            if mask_array.ndim == 3:

                mask_array = mask_array[:, :, 0]

            mask_image = Image.fromarray(
                mask_array.astype(np.uint8)
            )

        # Sempre transforma em escala de cinza
        mask_image = mask_image.convert("L")

        # Mesmo tamanho da profundidade
        mask_image = mask_image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        mask_array = np.asarray(
            mask_image
        )

        print(
            "DEBUG máscara:",
            mask_array.shape,
            flush=True
        )

        object_mask = (
            mask_array[
                ::stride,
                ::stride
            ] > 128
        )

        print(
            "DEBUG object_mask:",
            object_mask.shape,
            flush=True
        )

        print(
            "DEBUG valid:",
            valid.shape,
            flush=True
        )

        valid = (
            valid
            & object_mask
        )

    # ==========================================
    # VERIFICA PONTOS
    # ==========================================

    if not np.any(valid):

        raise ValueError(
            "Nenhum ponto 3D válido foi encontrado. "
            "A máscara pode ter removido todo o objeto."
        )

    # ==========================================
    # FILTRA PONTOS
    # ==========================================

    x = x[valid].astype(
        np.float32
    )

    y = y[valid].astype(
        np.float32
    )

    z = z[valid].astype(
        np.float32
    )

    # ==========================================
    # NORMALIZA PROFUNDIDADE
    # ==========================================

    z_min = np.min(z)
    z_max = np.max(z)

    if z_max - z_min > 1e-8:

        z = (
            z - z_min
        ) / (
            z_max - z_min
        )

    else:

        z[:] = 0.5

    # ==========================================
    # CENTRALIZA X/Y
    # ==========================================

    x = (
        x - width / 2
    ) / width

    y = -(
        y - height / 2
    ) / height

    # ==========================================
    # ESCALA Z
    # ==========================================

    z = z * 2.0

    # ==========================================
    # XYZ
    # ==========================================

    points = np.stack(
        [
            x,
            y,
            z
        ],
        axis=1
    )

    # ==========================================
    # CORES
    # ==========================================

    colors = None

    if image is not None:

        image = image.convert("RGB")

        image = image.resize(
            (width, height)
        )

        image_array = np.asarray(
            image
        )

        sampled_colors = (
            image_array[
                ::stride,
                ::stride
            ]
        )

        colors = sampled_colors[
            valid
        ]

    return points, colors


def save_pointcloud_ply(
    filename,
    points,
    colors=None
):
    """
    Salva a nuvem de pontos no formato PLY.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):

        raise ValueError(
            "points precisa ter formato (N, 3)."
        )

    if colors is not None:

        colors = np.asarray(colors)

        if len(colors) != len(points):

            raise ValueError(
                "A quantidade de cores precisa "
                "ser igual à quantidade de pontos."
            )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("ply\n")
        file.write("format ascii 1.0\n")

        file.write(
            f"element vertex {len(points)}\n"
        )

        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        if colors is not None:

            file.write(
                "property uchar red\n"
            )

            file.write(
                "property uchar green\n"
            )

            file.write(
                "property uchar blue\n"
            )

        file.write("end_header\n")

        if colors is not None:

            for point, color in zip(
                points,
                colors
            ):

                x, y, z = point

                r, g, b = color

                file.write(
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{z:.6f} "
                    f"{int(r)} "
                    f"{int(g)} "
                    f"{int(b)}\n"
                )

        else:

            for point in points:

                x, y, z = point

                file.write(
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{z:.6f}\n"
                )