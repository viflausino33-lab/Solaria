import numpy as np

from reconstruction_engine import preparar_pointcloud
from geometry import preparar_geometria
from surface_completion import completar_superficie


# =====================================================
# CRIA UMA NUVEM DE TESTE
# =====================================================

print("Criando nuvem de teste...")

points = np.random.rand(
    1000,
    3
).astype(
    np.float32
)

print(
    f"Pontos originais: {len(points)}"
)


# =====================================================
# ETAPA 1
# LIMPEZA
# =====================================================

print("\nPreparando point cloud...")

points, colors = preparar_pointcloud(
    points
)

print(
    f"Pontos após preparação: {len(points)}"
)


# =====================================================
# ETAPA 2
# GEOMETRIA
# =====================================================

print("\nPreparando geometria...")

geometria = preparar_geometria(
    points
)

pontos_geometria = geometria[
    "points"
]

normais = geometria[
    "normals"
]

dimensoes = geometria[
    "dimensions"
]

print(
    f"Pontos da geometria: "
    f"{len(pontos_geometria)}"
)

print(
    f"Normais: "
    f"{len(normais)}"
)

print(
    "Dimensões:"
)

print(
    f"  Largura: "
    f"{dimensoes['largura']:.4f}"
)

print(
    f"  Altura: "
    f"{dimensoes['altura']:.4f}"
)

print(
    f"  Profundidade: "
    f"{dimensoes['profundidade']:.4f}"
)


# =====================================================
# ETAPA 3
# COMPLETAR SUPERFÍCIE
# =====================================================

print(
    "\nCriando superfície completa..."
)

resultado = completar_superficie(
    pontos_geometria
)

frente = resultado[
    "frente"
]

verso = resultado[
    "verso"
]

completa = resultado[
    "completa"
]

print(
    f"Frente: "
    f"{len(frente)} pontos"
)

print(
    f"Verso: "
    f"{len(verso)} pontos"
)

print(
    f"Total: "
    f"{len(completa)} pontos"
)


# =====================================================
# RESULTADO
# =====================================================

print(
    "\n================================"
)

print(
    "TESTE DE RECONSTRUÇÃO CONCLUÍDO"
)

print(
    "================================"
)