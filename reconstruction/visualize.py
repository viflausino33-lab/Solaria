import numpy as np
import matplotlib.pyplot as plt

from reconstruction_engine import preparar_pointcloud
from geometry import preparar_geometria
from surface_completion import completar_superficie


def visualizar_reconstrucao(points):
    """
    Mostra a reconstrução 3D da nuvem
    de pontos.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    # ---------------------------------------------
    # PREPARAÇÃO
    # ---------------------------------------------

    points, colors = preparar_pointcloud(
        points
    )

    # ---------------------------------------------
    # GEOMETRIA
    # ---------------------------------------------

    geometria = preparar_geometria(
        points
    )

    frente = geometria["points"]

    # ---------------------------------------------
    # COMPLETAR SUPERFÍCIE
    # ---------------------------------------------

    resultado = completar_superficie(
        frente
    )

    frente = resultado["frente"]
    verso = resultado["verso"]

    # ---------------------------------------------
    # VISUALIZAÇÃO
    # ---------------------------------------------

    figura = plt.figure(
        figsize=(10, 8)
    )

    eixo = figura.add_subplot(
        111,
        projection="3d"
    )

    # Frente
    eixo.scatter(
        frente[:, 0],
        frente[:, 1],
        frente[:, 2],
        s=1,
        label="Frente"
    )

    # Verso
    eixo.scatter(
        verso[:, 0],
        verso[:, 1],
        verso[:, 2],
        s=1,
        label="Verso"
    )

    eixo.set_xlabel("X")
    eixo.set_ylabel("Y")
    eixo.set_zlabel("Z")

    eixo.set_title(
        "Reconstrução 3D - Solaria"
    )

    eixo.legend()

    plt.tight_layout()

    plt.show()


# =====================================================
# TESTE
# =====================================================

if __name__ == "__main__":

    print(
        "Criando objeto de teste..."
    )

    largura = 120
    altura = 180

    y, x = np.mgrid[
        0:altura,
        0:largura
    ]

    cx = largura / 2
    cy = altura / 2

    dx = x - cx
    dy = y - cy

    distancia = np.sqrt(
        dx ** 2 +
        dy ** 2
    )

    mascara = (
        distancia < 55
    )

    x_obj = x[mascara]
    y_obj = y[mascara]

    z = (
        1.0 -
        (
            distancia[mascara] / 55
        ) ** 2
    )

    points = np.stack(
        [
            x_obj,
            -y_obj,
            z * 40
        ],
        axis=1
    ).astype(
        np.float32
    )

    print(
        f"Objeto criado: {len(points)} pontos"
    )

    visualizar_reconstrucao(
        points
    )