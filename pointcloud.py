import numpy as np
from PIL import Image


# =====================================================
# DEPTH → NUVEM DE PONTOS
# =====================================================

def depth_to_pointcloud(
    depth,
    image=None,
    mask=None,
    stride=4
):
    """
    Converte um mapa de profundidade em uma
    nuvem de pontos 3D.

    depth:
        Matriz 2D de profundidade.

    image:
        Imagem RGB utilizada para colorir os pontos.

    mask:
        Máscara do objeto.
        Branco = objeto.
        Preto = fundo.

    stride:
        Quantidade de pixels pulados.
        stride=1 utiliza todos os pixels.
        stride=4 utiliza 1 a cada 4 pixels.
    """

    # =================================================
    # 1 — PREPARAR DEPTH
    # =================================================

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    depth = np.squeeze(
        depth
    )

    if depth.ndim != 2:

        raise ValueError(
            "Depth precisa ser 2D. "
            f"Formato recebido: {depth.shape}"
        )

    height, width = depth.shape

    # =================================================
    # 2 — VALIDAR STRIDE
    # =================================================

    stride = int(stride)

    if stride < 1:

        stride = 1

    # =================================================
    # 3 — GRID DE PIXELS
    # =================================================

    y, x = np.mgrid[
        0:height:stride,
        0:width:stride
    ]

    z = depth[
        ::stride,
        ::stride
    ]

    # =================================================
    # 4 — PONTOS VÁLIDOS
    # =================================================

    valid = np.isfinite(
        z
    )

    # =================================================
    # 5 — APLICAR MÁSCARA
    # =================================================

    if mask is not None:

        # ---------------------------------------------
        # CONVERTE PARA PIL
        # ---------------------------------------------

        if isinstance(
            mask,
            Image.Image
        ):

            mask_image = mask

        else:

            mask_array = np.asarray(
                mask
            )

            # -----------------------------------------
            # RGB / RGBA → GRAYSCALE
            # -----------------------------------------

            if mask_array.ndim == 3:

                mask_array = (
                    mask_array[:, :, 0]
                )

            mask_image = Image.fromarray(
                mask_array.astype(
                    np.uint8
                )
            )

        # ---------------------------------------------
        # ESCALA DE CINZA
        # ---------------------------------------------

        mask_image = mask_image.convert(
            "L"
        )

        # ---------------------------------------------
        # MESMO TAMANHO DO DEPTH
        # ---------------------------------------------

        mask_image = mask_image.resize(
            (
                width,
                height
            ),
            Image.Resampling.LANCZOS
        )

        mask_array = np.asarray(
            mask_image,
            dtype=np.uint8
        )

        # ---------------------------------------------
        # REDUZ PARA O MESMO STRIDE
        # ---------------------------------------------

        object_mask = (
            mask_array[
                ::stride,
                ::stride
            ] > 128
        )

        # ---------------------------------------------
        # GARANTE MESMO FORMATO
        # ---------------------------------------------

        if object_mask.shape != valid.shape:

            raise ValueError(
                "Máscara e depth possuem "
                "dimensões incompatíveis: "
                f"mask={object_mask.shape}, "
                f"depth={valid.shape}"
            )

        # ---------------------------------------------
        # APLICA
        # ---------------------------------------------

        valid = (
            valid
            & object_mask
        )

    # =================================================
    # 6 — VERIFICAR RESULTADO
    # =================================================

    if not np.any(
        valid
    ):

        raise ValueError(
            "Nenhum ponto 3D válido foi encontrado. "
            "Verifique a máscara do objeto."
        )

    # =================================================
    # 7 — FILTRAR XYZ
    # =================================================

    x = x[
        valid
    ].astype(
        np.float32
    )

    y = y[
        valid
    ].astype(
        np.float32
    )

    z = z[
        valid
    ].astype(
        np.float32
    )

    # =================================================
    # 8 — NORMALIZAR PROFUNDIDADE
    # =================================================

    z_min = np.min(
        z
    )

    z_max = np.max(
        z
    )

    if (
        z_max - z_min
        > 1e-8
    ):

        z = (
            z - z_min
        ) / (
            z_max - z_min
        )

    else:

        z[:] = 0.5

    # =================================================
    # 9 — CENTRALIZAR X
    # =================================================

    x = (
        x - width / 2.0
    ) / width

    # =================================================
    # 10 — CENTRALIZAR Y
    # =================================================

    y = -(
        y - height / 2.0
    ) / height

    # =================================================
    # 11 — ESCALA Z
    # =================================================

    z = z * 2.0

    # =================================================
    # 12 — MONTAR XYZ
    # =================================================

    points = np.stack(
        [
            x,
            y,
            z
        ],
        axis=1
    )

    # =================================================
    # 13 — CORES
    # =================================================

    colors = None

    if image is not None:

        # ---------------------------------------------
        # GARANTE PIL
        # ---------------------------------------------

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.fromarray(
                np.asarray(image)
            )

        image = image.convert(
            "RGB"
        )

        # ---------------------------------------------
        # MESMO TAMANHO
        # ---------------------------------------------

        image = image.resize(
            (
                width,
                height
            ),
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            image,
            dtype=np.uint8
        )

        # ---------------------------------------------
        # MESMO STRIDE
        # ---------------------------------------------

        sampled_colors = image_array[
            ::stride,
            ::stride
        ]

        # ---------------------------------------------
        # APLICA MESMA MÁSCARA
        # ---------------------------------------------

        colors = sampled_colors[
            valid
        ]

        colors = np.asarray(
            colors,
            dtype=np.uint8
        )

    # =================================================
    # DEBUG
    # =================================================

    print(
        "PointCloud:",
        f"{len(points)} pontos",
        f"| depth={depth.shape}",
        f"| stride={stride}",
        flush=True
    )

    if colors is not None:

        print(
            "PointCloud:",
            f"{len(colors)} cores",
            flush=True
        )

    # =================================================
    # RETORNO
    # =================================================

    return (
        points,
        colors
    )


# =====================================================
# SALVAR PLY
# =====================================================

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

    # =================================================
    # VALIDAR PONTOS
    # =================================================

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):

        raise ValueError(
            "points precisa ter formato (N, 3)."
        )

    # =================================================
    # VALIDAR CORES
    # =================================================

    if colors is not None:

        colors = np.asarray(
            colors,
            dtype=np.uint8
        )

        if len(colors) != len(points):

            raise ValueError(
                "A quantidade de cores precisa "
                "ser igual à quantidade de pontos."
            )

        if (
            colors.ndim != 2
            or colors.shape[1] != 3
        ):

            raise ValueError(
                "colors precisa ter formato (N, 3)."
            )

    # =================================================
    # CRIAR ARQUIVO
    # =================================================

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        # ---------------------------------------------
        # HEADER
        # ---------------------------------------------

        file.write(
            "ply\n"
        )

        file.write(
            "format ascii 1.0\n"
        )

        file.write(
            f"element vertex {len(points)}\n"
        )

        file.write(
            "property float x\n"
        )

        file.write(
            "property float y\n"
        )

        file.write(
            "property float z\n"
        )

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

        file.write(
            "end_header\n"
        )

        # ---------------------------------------------
        # PONTOS + CORES
        # ---------------------------------------------

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

        # ---------------------------------------------
        # SOMENTE PONTOS
        # ---------------------------------------------

        else:

            for point in points:

                x, y, z = point

                file.write(
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{z:.6f}\n"
                )