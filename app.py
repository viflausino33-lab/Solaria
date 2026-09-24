import gradio as gr
import spaces
import torch
import tempfile

from PIL import Image

from marigoldv2_inference import (
    BASE_MODEL_URI,
    MODEL_URI,
    MarigoldV2,
)

from segmentation import (
    gerar_mascara,
)

from pointcloud import (
    depth_to_pointcloud,
    save_pointcloud_ply,
)


# =====================================================
# CONFIGURAÇÃO
# =====================================================

model = None


# =====================================================
# CARREGAR MARIGOLD V2
# =====================================================

def carregar_modelo():

    global model

    if model is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Carregando Marigold V2 em {device}...",
            flush=True
        )

        model = MarigoldV2(
            BASE_MODEL_URI,
            MODEL_URI,
            device
        )

        print(
            "Marigold V2 carregado.",
            flush=True
        )

    return model


# =====================================================
# SEGMENTAÇÃO
# =====================================================

def preparar_mascara(imagem):

    if imagem is None:

        return (
            None,
            "Envie uma imagem."
        )

    try:

        print(
            "Iniciando segmentação...",
            flush=True
        )

        mascara = gerar_mascara(
            imagem
        )

        print(
            "Segmentação concluída.",
            flush=True
        )

        return (
            mascara,
            "Objeto identificado com sucesso."
        )

    except Exception as e:

        print(
            f"ERRO NA SEGMENTAÇÃO: {e}",
            flush=True
        )

        return (
            None,
            f"Erro na segmentação:\n{str(e)}"
        )


# =====================================================
# RECORTAR OBJETO
# =====================================================

def criar_imagem_do_objeto(
    imagem,
    mascara
):

    imagem = imagem.convert("RGB")
    mascara = mascara.convert("L")

    # Garante mesmo tamanho
    mascara = mascara.resize(
        imagem.size,
        Image.Resampling.LANCZOS
    )

    # -------------------------------------------------
    # ENCONTRA A ÁREA DO OBJETO
    # -------------------------------------------------

    bbox = mascara.getbbox()

    if bbox is None:

        raise ValueError(
            "Não foi possível encontrar "
            "o objeto na máscara."
        )

    # -------------------------------------------------
    # ADICIONA UMA PEQUENA MARGEM
    # -------------------------------------------------

    largura, altura = imagem.size

    margem_x = int(largura * 0.03)
    margem_y = int(altura * 0.03)

    x1 = max(
        0,
        bbox[0] - margem_x
    )

    y1 = max(
        0,
        bbox[1] - margem_y
    )

    x2 = min(
        largura,
        bbox[2] + margem_x
    )

    y2 = min(
        altura,
        bbox[3] + margem_y
    )

    # -------------------------------------------------
    # RECORTA
    # -------------------------------------------------

    imagem_cortada = imagem.crop(
        (x1, y1, x2, y2)
    )

    mascara_cortada = mascara.crop(
        (x1, y1, x2, y2)
    )

    # -------------------------------------------------
    # CRIA IMAGEM COM FUNDO PRETO
    # -------------------------------------------------

    fundo = Image.new(
        "RGB",
        imagem_cortada.size,
        (0, 0, 0)
    )

    objeto = Image.composite(
        imagem_cortada,
        fundo,
        mascara_cortada
    )

    return (
        objeto,
        mascara_cortada
    )


# =====================================================
# ETAPA 2 — 3D
# =====================================================

@spaces.GPU(duration=180)
def gerar_3d(
    imagem,
    mascara
):

    if imagem is None:

        return (
            None,
            None,
            None,
            "Envie uma imagem."
        )

    if mascara is None:

        return (
            None,
            None,
            None,
            "A máscara do objeto não foi gerada."
        )

    try:

        # =================================================
        # 1. RECORTAR OBJETO
        # =================================================

        print(
            "Preparando objeto...",
            flush=True
        )

        objeto, mascara_objeto = (
            criar_imagem_do_objeto(
                imagem,
                mascara
            )
        )

        print(
            f"Objeto preparado: {objeto.size}",
            flush=True
        )

        # =================================================
        # 2. CARREGAR MARIGOLD
        # =================================================

        print(
            "Carregando Marigold V2...",
            flush=True
        )

        modelo = carregar_modelo()

        # =================================================
        # 3. PROFUNDIDADE
        # =================================================

        print(
            "Calculando profundidade do objeto...",
            flush=True
        )

        resultados = modelo(
            objeto
        )

        print(
            "Resultado do Marigold recebido.",
            flush=True
        )

        # =================================================
        # 4. DEPTH VISUAL
        # =================================================

        depth_image = resultados.get(
            "Depth"
        )

        # =================================================
        # 5. RAW DEPTH
        # =================================================

        raw_depth = resultados.get(
            "raw_depth"
        )

        if raw_depth is None:

            raise ValueError(
                "O Marigold não retornou "
                "'raw_depth'."
            )

        print(
            "DEBUG raw_depth:",
            type(raw_depth),
            getattr(
                raw_depth,
                "shape",
                None
            ),
            flush=True
        )

        # =================================================
        # 6. NUVEM DE PONTOS
        # =================================================

        print(
            "Gerando nuvem de pontos...",
            flush=True
        )

        points, colors = (
            depth_to_pointcloud(

                raw_depth,

                objeto,

                mask=mascara_objeto,

                stride=4
            )
        )

        print(
            f"Nuvem criada: {len(points)} pontos.",
            flush=True
        )

        # =================================================
        # 7. SALVAR PLY
        # =================================================

        arquivo_ply = (
            tempfile.NamedTemporaryFile(
                suffix=".ply",
                delete=False
            )
        )

        arquivo_ply.close()

        save_pointcloud_ply(
            arquivo_ply.name,
            points,
            colors
        )

        print(
            f"PLY salvo em: {arquivo_ply.name}",
            flush=True
        )

        # =================================================
        # 8. STATUS
        # =================================================

        status = (
            "Processamento concluído!\n\n"
            f"Tamanho do objeto: {objeto.size}\n"
            f"Pontos 3D: {len(points):,}\n\n"
            "Fluxo utilizado:\n"
            "✓ Segmentação\n"
            "✓ Recorte do objeto\n"
            "✓ Marigold V2\n"
            "✓ Máscara aplicada\n"
            "✓ Nuvem de pontos"
        )

        return (
            mascara_objeto,
            depth_image,
            arquivo_ply.name,
            status
        )

    except Exception as e:

        print(
            "ERRO NA GERAÇÃO 3D:",
            repr(e),
            flush=True
        )

        return (
            None,
            None,
            None,
            f"Erro durante a geração 3D:\n\n{str(e)}"
        )


# =====================================================
# INTERFACE
# =====================================================

with gr.Blocks(
    title="Solaria 1.0 - 2D para 3D"
) as app:

    gr.Markdown(
        """
        # 🚀 Solaria 1.0

        ## Imagem 2D → Objeto → Profundidade → 3D

        O Solaria identifica o objeto principal,
        remove o fundo, calcula a profundidade
        com Marigold V2 e cria uma nuvem de pontos
        somente do objeto.
        """
    )

    # =================================================
    # IMAGEM + MÁSCARA
    # =================================================

    with gr.Row():

        with gr.Column():

            imagem_input = gr.Image(
                type="pil",
                label="Imagem 2D"
            )

        with gr.Column():

            mascara_output = gr.Image(
                label="Máscara do objeto"
            )

    # =================================================
    # BOTÃO
    # =================================================

    botao = gr.Button(
        "Gerar 3D",
        variant="primary"
    )

    # =================================================
    # PROFUNDIDADE
    # =================================================

    depth_output = gr.Image(
        label="Mapa de profundidade"
    )

    # =================================================
    # VISUALIZAÇÃO 3D
    # =================================================

    gr.Markdown(
        """
        ## Visualização 3D
        """
    )

    modelo_3d = gr.Model3D(
        label="Nuvem de pontos 3D",
        display_mode="point_cloud"
    )

    # =================================================
    # STATUS
    # =================================================

    resultado = gr.Textbox(
        label="Status",
        lines=8
    )

    # =================================================
    # FLUXO
    # =================================================

    etapa_mascara = botao.click(
        fn=preparar_mascara,
        inputs=imagem_input,
        outputs=[
            mascara_output,
            resultado
        ]
    )

    etapa_mascara.then(
        fn=gerar_3d,
        inputs=[
            imagem_input,
            mascara_output
        ],
        outputs=[
            mascara_output,
            depth_output,
            modelo_3d,
            resultado
        ]
    )


# =====================================================
# INICIAR
# =====================================================

app.queue().launch()
