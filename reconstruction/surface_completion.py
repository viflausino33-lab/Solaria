import numpy as np


def criar_camada_verso(
    points,
    profundidade=0.35
):
    """
    Cria uma primeira estimativa do verso
    do objeto a partir da nuvem frontal.

    Esta é uma etapa inicial da reconstrução.
    O objetivo é gerar uma segunda superfície
    que posteriormente será refinada.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Formato inválido de points: {points.shape}"
        )

    if len(points) == 0:
        raise ValueError(
            "A nuvem de pontos está vazia."
        )

    # ---------------------------------------------
    # EXTREMOS DE PROFUNDIDADE
    # ---------------------------------------------

    z_min = np.min(
        points[:, 2]
    )

    z_max = np.max(
        points[:, 2]
    )

    espessura = (
        z_max - z_min
    )

    # Caso a profundidade seja muito pequena,
    # utiliza uma espessura mínima.
    if espessura < 1e-5:
        espessura = profundidade

    # ---------------------------------------------
    # CRIA O VERSO
    # ---------------------------------------------

    verso = points.copy()

    verso[:, 2] = (
        verso[:, 2]
        - espessura
        - profundidade
    )

    return verso


def unir_frente_e_verso(
    frente,
    verso
):
    """
    Une as duas superfícies em uma única
    nuvem de pontos.
    """

    frente = np.asarray(
        frente,
        dtype=np.float32
    )

    verso = np.asarray(
        verso,
        dtype=np.float32
    )

    if frente.ndim != 2 or frente.shape[1] != 3:
        raise ValueError(
            "Frente precisa ter formato (N, 3)."
        )

    if verso.ndim != 2 or verso.shape[1] != 3:
        raise ValueError(
            "Verso precisa ter formato (N, 3)."
        )

    pontos = np.concatenate(
        [
            frente,
            verso
        ],
        axis=0
    )

    return pontos


def completar_superficie(
    points,
    profundidade=0.35
):
    """
    Pipeline inicial de preenchimento
    da superfície do objeto.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        raise ValueError(
            "Nuvem vazia."
        )

    frente = points.copy()

    verso = criar_camada_verso(
        frente,
        profundidade=profundidade
    )

    completa = unir_frente_e_verso(
        frente,
        verso
    )

    return {
        "frente": frente,
        "verso": verso,
        "completa": completa
    }
