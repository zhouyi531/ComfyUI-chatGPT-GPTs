import os
import base64
import torch
import openai
from PIL import Image
import io
from openai import OpenAI
from openai.types.chat import ChatCompletion
import sys
import boto3
from botocore.exceptions import ClientError
import time
from datetime import datetime
import hashlib
import yaml
from pathlib import Path

import nodes

sys.path.append(os.path.dirname(__file__))  # 添加当前目录到Python路径

print("[GPTs Node] 插件初始化中...")  # 在控制台查看是否输出

CONFIG_DIR = Path(__file__).parent / "configs"

class GPTsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai_config": ("OPENAI_CONFIG",),
                "s3_config": ("S3_CONFIG",),
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                # "custom_instructions": ("STRING", {"default": "你是一个专业的图像分析助手", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_response",)
    FUNCTION = "call_gpts"
    CATEGORY = "AI Tools/GPTs"

    def upload_to_s3(self, image_bytes, s3_config):
        try:
            # 修复后的打印语句
            print(f"""\
[S3 Debug] 配置参数:
    endpoint: {s3_config['endpoint_url']}
    bucket: {s3_config['bucket_name']}
    custom_domain: {s3_config['custom_domain']}
    path_style: {s3_config['force_path_style']}
    file_path: {s3_config['file_path']}\
        """)
            
            # 添加详细的客户端配置
            session = boto3.session.Session()
            config = boto3.session.Config(
                s3={'addressing_style': 'path' if s3_config['force_path_style'] == 'enable' else 'auto'},
                signature_version='s3v4'  # 强制使用V4签名
            )
            
            s3 = session.client(
                's3',
                aws_access_key_id=s3_config['aws_access_key'],
                aws_secret_access_key=s3_config['aws_secret_key'],
                endpoint_url=s3_config['endpoint_url'],
                config=config,
                region_name='us-east-1'  # 即使第三方服务也需要设置一个虚拟区域
            )
            
            # 打印请求前信息
            print(f"[S3 Debug] 准备上传文件，大小: {len(image_bytes)} bytes")
            print(f"[S3 Debug] 客户端配置: {config}")
            print(f"[S3 Debug] 完整访问密钥: {s3_config['aws_access_key'][:3]}...{s3_config['aws_access_key'][-3:]}")
            print(f"[S3 Debug] 密钥前三位: {s3_config['aws_secret_key'][:3]}...")

            # 生成带路径的文件名
            import uuid
            base_name = f"{uuid.uuid4()}.png"
            
            # 替换路径中的动态变量
            md5_hash = hashlib.md5(image_bytes).hexdigest()
            path_vars = {
                '{year}': datetime.now().strftime('%Y'),
                '{month}': datetime.now().strftime('%m'),
                '{day}': datetime.now().strftime('%d'),
                '{md5}': md5_hash,
                '{extName}': 'png'
            }
            
            file_path = s3_config['file_path']
            for var, value in path_vars.items():
                file_path = file_path.replace(var, value)
            
            filename = f"{file_path.strip('/')}/{base_name}" if file_path else base_name
            print(f"[S3 Debug] 最终文件路径: {filename}")  # 添加路径调试
            
            s3.put_object(
                Bucket=s3_config['bucket_name'],
                Key=filename,
                Body=image_bytes,
                ContentType='image/png'
            )
            
            # 修改后的URL生成逻辑
            if s3_config['custom_domain']:
                # 处理自定义域名格式
                domain = s3_config['custom_domain'].strip()
                # 移除可能存在的协议头
                if domain.startswith(('http://', 'https://')):
                    domain = domain.split('://', 1)[1]
                # 构建完整URL
                url = f"https://{domain}/{filename}"
                print(f"[S3 Debug] 生成自定义域名URL: {url}")  # 添加调试输出
            else:
                # 生成预签名URL（兼容第三方S3）
                url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': s3_config['bucket_name'], 'Key': filename},
                    ExpiresIn=3600
                )
            return {
                "url": url,
                "bucket": s3_config['bucket_name'],
                "key": filename  # 新增返回key
            }
        except ClientError as e:
            # 添加详细错误解析
            error_info = {
                'code': e.response['Error']['Code'],
                'message': e.response['Error']['Message'],
                'status_code': e.response['ResponseMetadata']['HTTPStatusCode'],
                'request_id': e.response['ResponseMetadata']['RequestId']
            }
            print(f"[S3 Error] 完整错误响应: {error_info}")
            print(f"[S3 Error] 建议排查步骤:")
            print("1. 检查访问密钥和密钥是否正确")
            print("2. 确认endpoint_url是否包含协议（http/https）")
            print("3. 检查服务器时间是否同步（时间偏差会导致签名错误）")
            print("4. 尝试切换force_path_style设置")
            return None

    def delete_from_s3(self, s3_config, file_info):
        try:
            session = boto3.session.Session(
                aws_access_key_id=s3_config['aws_access_key'],
                aws_secret_access_key=s3_config['aws_secret_key']
            )
            s3 = session.client(
                's3',
                endpoint_url=s3_config['endpoint_url'],
                config=boto3.session.Config(
                    s3={'addressing_style': 'path' if s3_config['force_path_style'] == 'enable' else 'auto'}
                )
            )
            
            # 添加删除前检查
            print(f"[S3 Cleanup] 验证删除权限...")
            response = s3.head_object(
                Bucket=file_info['bucket'],
                Key=file_info['key']
            )
            print(f"[S3 Cleanup] 文件存在验证: {response['ResponseMetadata']['HTTPStatusCode']}")
            
            # 执行删除
            delete_response = s3.delete_object(
                Bucket=file_info['bucket'],
                Key=file_info['key']
            )
            print(f"[S3 Cleanup] 删除响应: {delete_response}")
            
            # 添加删除后验证
            try:
                s3.head_object(Bucket=file_info['bucket'], Key=file_info['key'])
                print("[S3 Cleanup] 警告: 文件仍然存在")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print("[S3 Cleanup] 文件已确认删除")
                else:
                    raise
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"[S3 Cleanup Error] 删除失败 ({error_code}): {e.response['Error']['Message']}")
        except Exception as e:
            print(f"[S3 Cleanup Error] 意外错误: {str(e)}")

    def call_gpts(
        self,
        openai_config: dict,
        s3_config: dict,
        image,                       # torch.Tensor (C,H,W) or (1,C,H,W)
        prompt: str = "",            # 保留，但当前不使用
        custom_instructions: str = "", # This will be passed from INPUT_TYPES if connected
        temperature: float = 0.7,
    ):
        file_info = None
        try:
            # ---- 基础配置检查 ---------------------------------------------------
            if not openai_config.get("api_key"):
                return ("Error: OpenAI API key is missing.",)
            if not openai_config.get("assistant_id"):
                return ("Error: Assistant ID is missing.",)
            if not s3_config.get("bucket_name"):
                return ("Error: S3 bucket name is missing.",)

            # ---- 图片转字节流并上传 S3 ------------------------------------------
            tensor = image.cpu()
            print(f"[Debug] Original tensor shape: {tensor.shape}")
            
            # Handle different tensor formats from ComfyUI
            if tensor.dim() == 4:  # (batch, height, width, channels) or (batch, channels, height, width)
                tensor = tensor[0]  # Remove batch dimension
            elif tensor.dim() == 3:
                pass  # Keep as is, might be (height, width, channels) or (channels, height, width)
            elif tensor.dim() == 2:
                # Might be grayscale (height, width), add channel dimension
                tensor = tensor.unsqueeze(-1)
            else:
                return (f"Error: Unsupported tensor dimensions: {tensor.shape}",)
            
            print(f"[Debug] After batch removal tensor shape: {tensor.shape}")
            
            # Ensure we have a valid image tensor shape
            if len(tensor.shape) != 3:
                return (f"Error: Expected 3D tensor after processing, got shape: {tensor.shape}",)
            
            # Check if this looks like a valid image tensor
            h, w, c = tensor.shape
            if c > w or c > h:  # Likely (channels, height, width) format
                if tensor.shape[0] in [1, 3, 4]:  # Common channel counts
                    tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
                    print(f"[Debug] Permuted CHW to HWC: {tensor.shape}")
                else:
                    return (f"Error: Tensor shape {tensor.shape} doesn't appear to be a valid image format",)
            
            # Ensure tensor values are in valid range and convert to uint8
            tensor = tensor.clamp(0, 1)  # Clamp to [0, 1] first
            tensor = (tensor * 255).byte()  # Convert to [0, 255] uint8
            
            # Convert to numpy array
            numpy_array = tensor.numpy()
            print(f"[Debug] Final numpy array shape for PIL: {numpy_array.shape}")
            
            # Handle grayscale vs RGB
            if numpy_array.shape[2] == 1:
                numpy_array = numpy_array.squeeze(-1)  # Remove single channel dimension for grayscale
                pil_image = Image.fromarray(numpy_array, mode='L')
            elif numpy_array.shape[2] == 3:
                pil_image = Image.fromarray(numpy_array, mode='RGB')
            elif numpy_array.shape[2] == 4:
                pil_image = Image.fromarray(numpy_array, mode='RGBA')
            else:
                return (f"Error: Unsupported number of channels: {numpy_array.shape[2]}",)
            
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            file_info = self.upload_to_s3(buf.getvalue(), s3_config)
            if not file_info or not file_info.get("url"):
                return ("Error: S3 upload failed.",)

            # ---- OpenAI 客户端、助手 -------------------------------------------
            client = OpenAI(
                api_key=openai_config["api_key"],
                default_headers={"OpenAI-Beta": "assistants=v2"},
            )
            print(f"[OpenAI] SDK 版本: {openai.__version__}")

            # 只检索，不创建
            assistant = client.beta.assistants.retrieve(
                assistant_id=openai_config["assistant_id"], timeout=30
            )

            # ---- 构造消息：仅 image_url -----------------------------------------
            messages_payload = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": file_info["url"],  # 预签名 / 公开 URL
                                "detail": "auto",
                            },
                        }
                    ],
                }
            ]

            # ---- 创建线程并运行 --------------------------------------------------
            thread = client.beta.threads.create(messages=messages_payload)
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id, # Corrected from openai_config[\"assistant_id\"] to assistant.id
                temperature=temperature,
            )

            # ---- 轮询运行状态 ----------------------------------------------------
            for attempt in range(30):  # 最长约 60 s
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id, run_id=run.id
                )
                print(f"[OpenAI] 运行状态检查 #{attempt+1}: {run.status}")
                if run.status == "completed":
                    break
                if run.status in {"failed", "cancelled", "expired"}:
                    detail = (
                        f"{run.last_error.code} - {run.last_error.message}"
                        if run.last_error
                        else "no detail"
                    )
                    return (f"Error: Run {run.status}. {detail}",)
                time.sleep(2)
            else:
                return ("Error: Request timed out.",)

            # ---- 获取最新助手回复（文本） - Aligned with hello_world example ----
            msgs = client.beta.threads.messages.list(
                thread_id=thread.id, order="desc", limit=1 # Fetch only the latest message
            )
            
            response_text = ""
            if msgs.data and msgs.data[0]: # Check if data exists and has at least one message
                # Directly access the first (and only expected) message
                latest_message = msgs.data[0]
                if latest_message.role == "assistant": # Double check it's from assistant
                    response_text = next((
                        block.text.value
                        for block in latest_message.content
                        if block.type == "text" and hasattr(block, 'text') and hasattr(block.text, "value")
                    ), "")

            if not response_text.strip():
                # If the run failed, provide that error instead of just "empty response"
                if run.status != "completed": # Check run status if response is empty
                    detail = (
                        f"{run.last_error.code} - {run.last_error.message}"
                        if run.last_error
                        else "no detail for run status: " + run.status
                    )
                    return (f"Error: Run {run.status} but no text response. {detail}",)
                return ("Error: Empty response from assistant or text content not found.",)

            # 简单清洗
            clean = "\\n".join(line.strip() for line in response_text.splitlines() if line.strip())
            
            # 打印assistant的回复到终端
            print(f"\n[Assistant Reply] =====================================")
            print(clean)
            print(f"===============================================\n")
            
            return (clean,)

        except Exception as e:
            import traceback # openai import is already global
            traceback.print_exc()
            # Check if it's an OpenAI APIError to potentially get more specific info
            if isinstance(e, openai.APIError):
                 return (f"Error: OpenAI APIError {e.status_code} - {e.message}",)
            return (f"Error: {type(e).__name__} - {str(e)}",)

        finally:
            # ---- S3 清理 --------------------------------------------------------
            if file_info and file_info.get("bucket") and file_info.get("key"):
                try:
                    self.delete_from_s3(s3_config, file_info)
                except Exception as e:
                    print(f"[S3 Cleanup Error] {str(e)}")

class S3ConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        config_files = [f.stem for f in CONFIG_DIR.glob("s3_*.yaml")]
        return {
            "required": {
                "config_source": (["file", "manual"], {"default": "file"}),
                "config_name": (config_files, {"default": "s3_config"}),
            },
            "optional": {
                "aws_access_key": ("STRING", {"default": ""}),
                "aws_secret_key": ("STRING", {"default": ""}),
                "endpoint_url": ("STRING", {"default": "https://s3.amazonaws.com"}),
                "bucket_name": ("STRING", {"default": "my-bucket"}),
                "custom_domain": ("STRING", {"default": ""}),
                "file_path": ("STRING", {"default": ""}),
                "force_path_style": (["enable", "disable"], {"default": "disable"}),
            }
        }

    RETURN_TYPES = ("S3_CONFIG",)
    RETURN_NAMES = ("s3_config",)
    CATEGORY = "AI Tools/Config"
    FUNCTION = "get_config"

    def get_config(self, config_source, config_name, **kwargs):
        try:
            print(f"[Config Debug] 正在加载S3配置，来源: {config_source}, 文件名: {config_name}.yaml")
            config_path = CONFIG_DIR / f"{config_name}.yaml"
            print(f"[Config Debug] 配置文件路径: {config_path.absolute()}")
            
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path) as f:
                raw_content = f.read()
                print(f"[Config Debug] 原始文件内容:\n{raw_content}")
                configs = yaml.safe_load(raw_content)
                        
            # 直接获取default配置
            config = configs.get('default', {})
            
            # 验证必要字段
            required_fields = ['aws_access_key', 'aws_secret_key', 'bucket_name']
            for field in required_fields:
                if not config.get(field):
                    raise ValueError(f"缺少必要字段: {field}")
            
            print(f"[S3 Config] 加载配置: {config}")
            return (config,)  # 返回元组包裹的字典
        except Exception as e:
            print(f"[Config Error] 加载S3配置失败: {str(e)}")
            return (self.get_empty_config(),)  # 返回空配置元组

    def get_empty_config(self):
        return {
            'aws_access_key': '',
            'aws_secret_key': '',
            'endpoint_url': 'https://s3.amazonaws.com',
            'bucket_name': '',
            'custom_domain': '',
            'file_path': '',
            'force_path_style': 'disable'
        }

class OpenAIConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        config_files = [f.stem for f in CONFIG_DIR.glob("openai_*.yaml")]
        return {
            "required": {
                "config_source": (["file", "manual"], {"default": "file"}),
                "config_name": (config_files, {"default": "openai_config"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "assistant_id": ("STRING", {"default": "asst_xxxxxxxx"}),
            }
        }

    RETURN_TYPES = ("OPENAI_CONFIG",)
    RETURN_NAMES = ("openai_config",)
    CATEGORY = "AI Tools/Config"
    FUNCTION = "get_config"

    def get_config(self, config_source, config_name, **kwargs):
        config_data = {}
        if config_source == "file":
            config_file = CONFIG_DIR / f"{config_name}.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_yaml = yaml.safe_load(f)
                    if isinstance(loaded_yaml, dict) and 'default' in loaded_yaml and isinstance(loaded_yaml['default'], dict):
                        config_data = loaded_yaml['default']
                    else:
                        config_data = loaded_yaml
            else:
                print(f"[Config Node] 警告: 配置文件 {config_file} 未找到，将使用空配置。")
                return (self.get_empty_config(),)

        final_config = {}
        final_config.update(config_data)

        # Override with UI inputs if they are provided (non-empty)
        if kwargs.get('api_key'):
            final_config['api_key'] = kwargs['api_key']
        if kwargs.get('assistant_id'):
            final_config['assistant_id'] = kwargs['assistant_id']

        # If keys are still missing, but were present in UI (even if empty), use UI's (potentially empty) value.
        # This ensures UI's default empty strings are passed if no config file value exists.
        if 'api_key' not in final_config and 'api_key' in kwargs:
             final_config['api_key'] = kwargs['api_key']
        if 'assistant_id' not in final_config and 'assistant_id' in kwargs:
             final_config['assistant_id'] = kwargs['assistant_id']

        if not final_config.get('api_key'):
            print("[Config Node] 错误: API Key 未在配置文件或手动输入中提供。")
            return ({"error": "API Key is required", "api_key": "", "assistant_id": final_config.get('assistant_id', "")},)

        if not final_config.get('assistant_id'):
            print("[Config Node] 错误: Assistant ID 未在配置文件或手动输入中提供。")
            return ({"error": "Assistant ID is required", "api_key": final_config.get('api_key', ""), "assistant_id": ""},)
            
        print(f"[Config Node] OpenAI配置加载成功: api_key={'*' * (len(final_config.get('api_key', '')) - 3) + final_config.get('api_key', '')[-3:] if final_config.get('api_key') else '未设置'}, assistant_id={final_config.get('assistant_id')}")
        return (final_config,)

    def get_empty_config(self):
        return {
            "api_key": "",
            "assistant_id": ""
        }

NODE_CLASS_MAPPINGS = {
    "GPTsNode": GPTsNode,
    "S3Config": S3ConfigNode,
    "OpenAIConfig": OpenAIConfigNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTsNode": "🔮 GPTs调用节点",
    "S3Config": "☁️ S3存储配置",
    "OpenAIConfig": "🔑 OpenAI配置"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 