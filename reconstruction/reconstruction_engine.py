import numpy as np


def limpar_pointcloud(
    points,
    colors=None,
    z_min_percent=0.0,
    z_max_percent=1.0
):
    """
    Limpa a nuvem de pontos e remove valores inválidos.

    points:
        Array (N, 3).

    colors:
        Cores correspondentes aos pontos.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Formato inválido de points: {points.shape}"
        )

    # Remove NaN e infinito
    valid = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[valid]

    if colors is not None:

        colors = np.asarray(
            colors,
            dtype=np.uint8
        )

        if len(colors) != len(valid):
            raise ValueError(
                "Quantidade de cores diferente "
                "da quantidade de pontos."
            )

        colors = colors[valid]

    if len(points) == 0:
        raise ValueError(
            "A nuvem não possui pontos válidos."
        )

    # ---------------------------------------------
    # FILTRO DE PROFUNDIDADE
    # ---------------------------------------------

    z = points[:, 2]

    limite_min = np.quantile(
        z,
        z_min_percent
    )

    limite_max = np.quantile(
        z,
        z_max_percent
    )

    filtro = (
        (z >= limite_min)
        &
        (z <= limite_max)
    )

    points = points[filtro]

    if colors is not None:
        colors = colors[filtro]

    print(
        f"Reconstrução: {len(points)} pontos após limpeza.",
        flush=True
    )

    return points, colors


def centralizar_pointcloud(
    points
):
    """
    Centraliza a nuvem de pontos
    no centro do objeto.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        raise ValueError(
            "Nuvem de pontos vazia."
        )

    centro = np.mean(
        points,
        axis=0
    )

    points = points - centro

    return points


def normalizar_pointcloud(
    points,
    tamanho=2.0
):
    """
    Normaliza o tamanho da nuvem.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        raise ValueError(
            "Nuvem de pontos vazia."
        )

    maior_dimensao = np.max(
        np.ptp(
            points,
            axis=0
        )
    )

    if maior_dimensao < 1e-8:
        return points

    escala = (
        tamanho /
        maior_dimensao
    )

    return points * escala


def preparar_pointcloud(
    points,
    colors=None
):
    """
    Pipeline inicial de preparação
    da nuvem de pontos.
    """

    points, colors = limpar_pointcloud(
        points,
        colors
    )

    points = centralizar_pointcloud(
        points
    )

    points = normalizar_pointcloud(
        points
    )

    return points, colors