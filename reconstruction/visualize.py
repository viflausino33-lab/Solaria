import numpy as np
import plotly.graph_objects as go

from reconstruction_engine import preparar_pointcloud
from geometry import preparar_geometria
from surface_completion import completar_superficie


def visualizar_reconstrucao(points):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    # ==========================================
    # PREPARAÇÃO
    # ==========================================

    points, colors = preparar_pointcloud(
        points
    )

    # ==========================================
    # GEOMETRIA
    # ==========================================

    geometria = preparar_geometria(
        points
    )

    frente = geometria["points"]

    # ==========================================
    # COMPLETAR SUPERFÍCIE
    # ==========================================

    resultado = completar_superficie(
        frente
    )

    frente = resultado["frente"]
    verso = resultado["verso"]

    print(
        f"Frente: {len(frente)} pontos"
    )

    print(
        f"Verso: {len(verso)} pontos"
    )

    # ==========================================
    # GRÁFICO 3D
    # ==========================================

    figura = go.Figure()

    # Frente
    figura.add_trace(
        go.Scatter3d(
            x=frente[:, 0],
            y=frente[:, 1],
            z=frente[:, 2],
            mode="markers",
            marker=dict(
                size=2
            ),
            name="Frente"
        )
    )

    # Verso
    figura.add_trace(
        go.Scatter3d(
            x=verso[:, 0],
            y=verso[:, 1],
            z=verso[:, 2],
            mode="markers",
            marker=dict(
                size=2
            ),
            name="Verso"
        )
    )

    figura.update_layout(
        title="Solaria - Reconstrução 3D",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=1000,
        height=800
    )

    # ==========================================
    # SALVAR HTML
    # ==========================================

    arquivo = "reconstrucao_3d.html"

    figura.write_html(
        arquivo
    )

    print()
    print(
        "================================"
    )

    print(
        "VISUALIZAÇÃO 3D CRIADA"
    )

    print(
        "================================"
    )

    print(
        f"Arquivo: {arquivo}"
    )


# ==============================================
# TESTE
# ==============================================

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