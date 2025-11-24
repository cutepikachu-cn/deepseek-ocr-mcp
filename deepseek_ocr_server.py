import os
import logging
import base64
import mimetypes
from typing import Optional, Literal
from fastmcp import FastMCP
from openai import OpenAI, APIError
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 FastMCP 服务
mcp = FastMCP(
    name="DeepSeek-OCR",
    description="利用 DeepSeek-VL2/OCR 模型提供强大的多语言光学字符识别服务，支持 Markdown 格式输出、公式识别及版面分析。"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepSeek-OCR")

# 定义场景与官方提示词的映射
SCENE_PROMPTS = {
    "document": "<|grounding|>Convert the document to markdown.",
    "ocr_grounded": "<|grounding|>OCR this image.",
    "ocr_free": "Free OCR.",
    "figure": "Parse the figure.",
    "description": "Describe this image in detail.",
}

def get_client() -> OpenAI:
    """
    构造 OpenAI 客户端，严格从环境变量读取配置
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_BASE")
    
    # 强制校验 API_KEY
    if not api_key:
        raise ValueError("API Key is missing. Set API_KEY environment variable.")
    
    # 强制校验 API_BASE (防止默认连接到 OpenAI 官方)
    if not base_url:
        raise ValueError("API Base URL is missing. Set API_BASE environment variable (e.g., https://api.deepseek.com).")

    return OpenAI(api_key=api_key, base_url=base_url)

def encode_local_image(image_path: str) -> str:
    """
    读取本地图片并转换为 Base64 Data URI 格式
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Local image file not found: {image_path}")
        
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg" # 默认 fallback
        
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
    return f"data:{mime_type};base64,{base64_encoded}"

def process_image_input(image_input: str) -> str:
    """
    处理图片输入：严格限制为 HTTP/HTTPS URL 或 有效的本地文件路径
    """
    image_input = image_input.strip()
    
    # 1. 检查是否为网络 URL
    if image_input.startswith("http://") or image_input.startswith("https://"):
        return image_input
        
    # 2. 检查是否为本地文件
    # os.path.isfile 会检查路径是否存在且是一个常规文件
    if os.path.isfile(image_input):
        try:
            return encode_local_image(image_input)
        except Exception as e:
            raise ValueError(f"Failed to read local image file '{image_input}': {str(e)}")

    # 3. 如果都不是，抛出错误，拒绝处理 Base64 字符串以节省 Token
    error_msg = (
        f"Invalid image input: '{image_input[:50]}...'. "
        "Input must be a valid HTTP/HTTPS URL or an absolute local file path."
    )
    logger.warning(error_msg)
    raise ValueError(error_msg)

@mcp.tool()
def recognize_image(
    image: str,
    scene: Literal["document", "ocr_grounded", "ocr_free", "figure", "description"] = "document",
    instruction: Optional[str] = None,
    detail: Literal["auto", "low", "high"] = "auto"
) -> str:
    """
    调用 DeepSeek-OCR 能力识别图片内容。

    Args:
        image: 图片输入。必须是以下两种格式之一：
               1. HTTP/HTTPS URL (e.g., "https://example.com/a.png")
               2. 本地文件绝对路径 (e.g., "/Users/user/Desktop/a.png")
               注意：不支持直接传入 Base64 字符串。
        scene: 识别场景，自动匹配官方提示词。默认为 "document"。
        instruction: 自定义提示词。如果提供，将覆盖 scene 对应的默认提示词。
        detail: 图片细节水平 (auto, low, high)。
    
    Returns:
        识别后的文本内容。
    """
    
    # 从环境变量获取配置 (不再提供默认值)
    target_model = os.getenv("MODEL")
    if not target_model:
        return "Error: Configuration missing. Set MODEL environment variable."
    
    # 从环境变量获取 temperature，默认为 0.0
    try:
        final_temperature = float(os.getenv("TEMPERATURE", "0.0"))
    except ValueError:
        final_temperature = 0.0
    
    # 确定最终使用的提示词
    final_instruction = instruction if instruction else SCENE_PROMPTS.get(scene, SCENE_PROMPTS["document"])
    
    logger.info(f"Starting OCR task. Model: {target_model} | Scene: {scene} | Temp: {final_temperature}")

    # 处理图片输入
    try:
        final_image_url = process_image_input(image)
    except ValueError as ve:
        # 捕获预期的输入错误并返回给 LLM
        return f"Error: {str(ve)}"
    except Exception as e:
        return f"Unexpected error processing image: {str(e)}"

    try:
        client = get_client()
        
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": final_image_url,
                                "detail": detail
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096, 
            temperature=final_temperature,
            stream=False
        )

        content = response.choices[0].message.content
        logger.info("OCR task completed successfully.")
        return content

    except APIError as e:
        # 不再硬编码厂商名称，使用通用错误描述
        error_msg = f"API Request Error: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()