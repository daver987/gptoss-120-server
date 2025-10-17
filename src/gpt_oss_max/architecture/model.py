from max.pipelines.architectures.gpt_oss.model import GptOssModel as _GptOssModel


class GptOssModel(_GptOssModel):
    """
    Wrapper around Modular's stock GPT-OSS pipeline.
    Retaining a dedicated subclass keeps the extension point for MXFP4
    kernels without diverging from the upstream architecture.
    """

    pass
