import numpy as np

from PIL import Image
from rembg import remove, new_session


# Mantém a sessão carregada para não baixar/carregar
# o modelo novamente a cada imagem.
_session = None


def carregar_segmentador():
    global _session

    if _session is None:
        _session = new_session("isnet-general-use")

    return _session


def gerar_mascara(imagem: Image.Image):
    """
    Gera uma máscara do objeto principal da imagem.

    Branco = objeto
    Preto = fundo
    """

    if imagem is None:
        raise ValueError("Nenhuma imagem foi enviada.")

    imagem = imagem.convert("RGB")

    session = carregar_segmentador()

    mascara = remove(
        imagem,
        session=session,
        only_mask=True,
        post_process_mask=True
    )

    if not isinstance(mascara, Image.Image):
        mascara = Image.fromarray(
            np.asarray(mascara)
        )

    mascara = mascara.convert("L")

    # Garante que a máscara tenha exatamente
    # o mesmo tamanho da imagem original.
    mascara = mascara.resize(
        imagem.size,
        Image.Resampling.LANCZOS
    )

    return mascara