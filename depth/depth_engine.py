import numpy as np

from PIL import Image


# =====================================================
# PREPARAR DEPTH
# =====================================================

def preparar_depth(depth):
    """
    Converte o resultado de profundidade
    para um array 2D (altura, largura).
    """

    if depth is None:

        raise ValueError(
            "Depth não foi fornecido."
        )

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    # Remove dimensões extras
    depth = np.squeeze(depth)

    # Garante que seja 2D
    if depth.ndim != 2:

        raise ValueError(
            f"Formato de depth inválido: "
            f"{depth.shape}"
        )

    return depth


# =====================================================
# NORMALIZAR DEPTH
# =====================================================

def normalizar_depth(depth):
    """
    Normaliza a profundidade
    para valores entre 0 e 1.
    """

    depth = preparar_depth(
        depth
    )

    minimo = np.nanmin(
        depth
    )

    maximo = np.nanmax(
        depth
    )

    if (
        not np.isfinite(minimo)
        or
        not np.isfinite(maximo)
    ):

        raise ValueError(
            "A profundidade contém "
            "valores inválidos."
        )

    intervalo = (
        maximo - minimo
    )

    if intervalo < 1e-8:

        return np.zeros_like(
            depth
        )

    depth = (
        depth - minimo
    ) / intervalo

    return depth.astype(
        np.float32
    )


# =====================================================
# VALIDAR DEPTH + IMAGEM
# =====================================================

def validar_depth_e_imagem(
    depth,
    imagem
):
    """
    Garante que a profundidade
    e a imagem tenham dimensões
    compatíveis.
    """

    # ---------------------------------------------
    # PREPARA DEPTH
    # ---------------------------------------------

    depth = preparar_depth(
        depth
    )

    # ---------------------------------------------
    # GARANTE PIL
    # ---------------------------------------------

    if not isinstance(
        imagem,
        Image.Image
    ):

        imagem = Image.fromarray(
            np.asarray(imagem)
        )

    imagem = imagem.convert(
        "RGB"
    )

    # ---------------------------------------------
    # DIMENSÕES DO DEPTH
    # ---------------------------------------------

    altura, largura = (
        depth.shape
    )

    # ---------------------------------------------
    # AJUSTA IMAGEM
    # ---------------------------------------------

    if imagem.size != (
        largura,
        altura
    ):

        imagem = imagem.resize(
            (
                largura,
                altura
            ),
            Image.Resampling.LANCZOS
        )

    return (
        depth,
        imagem
    )