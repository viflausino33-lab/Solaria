import numpy as np


def estimar_normais(points):
    """
    Estima uma direção normal aproximada
    para cada ponto da nuvem.

    Nesta primeira versão usamos uma
    aproximação simples baseada na profundidade.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Formato inválido de points: {points.shape}"
        )

    if len(points) < 3:
        raise ValueError(
            "São necessários pelo menos 3 pontos."
        )

    # -------------------------------------------------
    # CENTRO DA NUVEM
    # -------------------------------------------------

    centro = np.mean(
        points,
        axis=0
    )

    # -------------------------------------------------
    # VETORES DO CENTRO PARA OS PONTOS
    # -------------------------------------------------

    vetores = (
        points - centro
    )

    # -------------------------------------------------
    # NORMALIZAÇÃO
    # -------------------------------------------------

    comprimentos = np.linalg.norm(
        vetores,
        axis=1,
        keepdims=True
    )

    comprimentos[
        comprimentos < 1e-8
    ] = 1.0

    normais = (
        vetores /
        comprimentos
    )

    return normais


def criar_superficie_frontal(
    points
):
    """
    Cria uma representação inicial da
    superfície frontal do objeto.

    Esta função ainda não cria o verso.

    O objetivo é transformar a nuvem
    em uma estrutura organizada que
    poderemos utilizar posteriormente
    para reconstrução.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        raise ValueError(
            "A nuvem de pontos está vazia."
        )

    # -------------------------------------------------
    # ORDENA POR ALTURA
    # -------------------------------------------------

    ordem = np.argsort(
        points[:, 1]
    )

    pontos_ordenados = points[
        ordem
    ]

    return pontos_ordenados


def estimar_dimensoes(
    points
):
    """
    Calcula largura, altura e profundidade
    aproximadas do objeto.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        raise ValueError(
            "A nuvem de pontos está vazia."
        )

    minimo = np.min(
        points,
        axis=0
    )

    maximo = np.max(
        points,
        axis=0
    )

    dimensoes = (
        maximo - minimo
    )

    return {
        "largura": float(
            dimensoes[0]
        ),
        "altura": float(
            dimensoes[1]
        ),
        "profundidade": float(
            dimensoes[2]
        )
    }


def preparar_geometria(
    points
):
    """
    Pipeline inicial de geometria.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        raise ValueError(
            "Nuvem de pontos vazia."
        )

    normais = estimar_normais(
        points
    )

    dimensoes = estimar_dimensoes(
        points
    )

    superficie = criar_superficie_frontal(
        points
    )

    return {
        "points": superficie,
        "normals": normais,
        "dimensions": dimensoes
    }
