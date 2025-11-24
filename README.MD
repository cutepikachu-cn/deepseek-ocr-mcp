# DeepSeek-OCR MCP Server

这是一个基于 FastMCP 构建的 Model Context Protocol (MCP) 服务器，旨在为 LLM（如 Claude Desktop, Cursor, Visual Studio Code 等）提供调用 DeepSeek-OCR（视觉语言模型）的能力。

## 特性

- 🖼️ 高效图片处理：仅支持 HTTP/HTTPS 链接 或 本地文件路径。
  - 注意：为了避免消耗大量 Token，本工具不支持直接传递 Base64 字符串。
- 🎯 官方提示词集成：内置 DeepSeek 官方推荐的 5 种标准场景提示词。
- 🔒 安全配置：敏感信息（API Key, Base URL）仅通过环境变量配置。
- 📝 Markdown 友好：默认模式完美支持 Markdown 表格和文档结构提取。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量 (必须)

请确保在运行环境或 `.env` 文件中设置了以下变量：

```bash
API_KEY=sk-xxxxxxxxxxxxxxxx
API_BASE=https://api.deepseek.com/v1
MODEL=deepseek-ocr
TEMPERATURE=0.0  # (可选) 默认为 0.0
```


3. Claude Desktop 配置

```json
{
  "mcpServers": {
    "deepseek-ocr": {
      "command": "python",
      "args": ["/绝对路径/to/your/deepseek_ocr_server.py"],
      "env": {
        "API_KEY": "你的API密钥",
        "API_BASE": "[https://api.deepseek.com/v1](https://api.deepseek.com/v1)",
        "MODEL": "deepseek-chat",
        "TEMPERATURE": "0.0"
      }
    }
  }
}
```

## 工具使用说明

### 核心参数
- image (必填):
    -✅ 远程链接: https://example.com/doc.png
    -✅ 本地路径: /Users/username/Desktop/scan.jpg (服务器会自动读取)
    - ❌ 不支持 Base64 字符串直接输入
- scene (选填, 默认 document):
    - document: 文档转 Markdown (推荐)
    - ocr_grounded: 通用 OCR
    - figure: 图表分析
    - ocr_free: 纯文本提取
- detail (选填, 默认 auto): 图片细节模式 (auto, low, high)。
- instruction: 自定义提示词 (可选)。

### 使用示例

场景 1：本地文件识别

```
"帮我识别桌面上这个发票的内容：/Users/me/Desktop/invoice.png"
```

场景 2：远程表格转 Markdown

```
"把这个在线图表转成数据表：https://www.google.com/search?q=https://example.com/chart.png"
```