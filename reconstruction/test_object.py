import numpy as np

from reconstruction_engine import preparar_pointcloud
from geometry import preparar_geometria
from surface_completion import completar_superficie


def criar_objeto_teste():

    largura = 120
    altura = 180

    y, x = np.mgrid[
        0:altura,
        0:largura
    ]

    # Centro do objeto
    cx = largura / 2
    cy = altura / 2

    # Distância do centro
    dx = x - cx
    dy = y - cy

    distancia = np.sqrt(
        dx ** 2 +
        dy ** 2
    )

    # Cria uma forma oval
    mascara = distancia < 55

    x_obj = x[mascara]
    y_obj = y[mascara]

    # Cria profundidade curva
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

    return points


print(
    "Criando objeto de teste..."
)

points = criar_objeto_teste()

print(
    f"Objeto criado: {len(points)} pontos"
)


print(
    "\nPreparando point cloud..."
)

points, colors = preparar_pointcloud(
    points
)

print(
    f"Pontos preparados: {len(points)}"
)


print(
    "\nPreparando geometria..."
)

geometria = preparar_geometria(
    points
)

points = geometria[
    "points"
]

print(
    "Dimensões:"
)

for nome, valor in geometria[
    "dimensions"
].items():

    print(
        f"  {nome}: {valor:.4f}"
    )


print(
    "\nCompletando superfície..."
)

resultado = completar_superficie(
    points
)

print(
    f"Frente: "
    f"{len(resultado['frente'])}"
)

print(
    f"Verso: "
    f"{len(resultado['verso'])}"
)

print(
    f"Completa: "
    f"{len(resultado['completa'])}"
)


print(
    "\n================================"
)

print(
    "OBJETO DE TESTE CONCLUÍDO"
)

print(
    "================================"
)