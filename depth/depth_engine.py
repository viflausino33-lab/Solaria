import numpy as np


def preparar_depth(depth):
    """
    Normaliza o resultado de profundidade
    para um array 2D (altura, largura).
    """

    if depth is None:
        raise ValueError("Depth não foi fornecido.")

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    # Remove dimensões extras
    depth = np.squeeze(depth)

    # Garante que temos uma imagem 2D
    if depth.ndim != 2:
        raise ValueError(
            f"Formato de depth inválido: {depth.shape}"
        )

    return depth


def normalizar_depth(depth):
    """
    Normaliza a profundidade para valores entre 0 e 1.
    """

    depth = preparar_depth(depth)

    minimo = np.nanmin(depth)
    maximo = np.nanmax(depth)

    if not np.isfinite(minimo) or not np.isfinite(maximo):
        raise ValueError(
            "A profundidade contém valores inválidos."
        )

    intervalo = maximo - minimo

    if intervalo < 1e-8:
        return np.zeros_like(depth)

    depth = (
        depth - minimo
    ) / intervalo

    return depth.astype(np.float32)