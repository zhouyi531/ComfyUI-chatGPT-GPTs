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
                "custom_instructions": ("STRING", {"default": "你是一个专业的图像分析助手", "multiline": True}),
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

    def call_gpts(self, openai_config, s3_config, prompt, image, custom_instructions="", temperature=0.7):
        file_info = None  # 用于存储上传文件信息
        try:
            print(f"[Debug] openai_config类型: {type(openai_config)}, 内容: {openai_config}")
            print(f"[Debug] s3_config类型: {type(s3_config)}, 内容: {s3_config}")
            # 添加配置验证
            if not openai_config.get('api_key'):
                raise ValueError("OpenAI API密钥不能为空")
            if not openai_config.get('assistant_id'):
                raise ValueError("助手ID不能为空")
            if not s3_config.get('bucket_name'):
                raise ValueError("S3存储桶名称不能为空")
            
            # 转换图像为字节流
            tensor = image.cpu()
            pil_image = Image.fromarray(tensor.mul(255).clamp(0, 255).byte().numpy()[0])
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # 上传到S3（修改返回值处理）
            file_info = self.upload_to_s3(img_byte_arr, s3_config)
            if not file_info or not file_info.get('url'):
                return ("Error: S3上传失败",)
            
            # 创建OpenAI客户端时添加版本配置
            client = OpenAI(
                api_key=openai_config['api_key'],
                default_headers={"OpenAI-Beta": "assistants=v2"}  # 显式指定API版本
            )
            
            # 检查SDK版本
            print(f"[OpenAI] SDK版本: {openai.__version__}")  # 添加版本信息输出
            
            # 创建或获取现有助手时添加版本参数
            assistant = client.beta.assistants.retrieve(
                assistant_id=openai_config['assistant_id'],
                timeout=30  # 增加超时时间
            )
            
            # 构建消息内容
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": file_info['url']}  # 使用S3 URL
                    }
                ]
            }]

            # 创建Thread并运行
            thread = client.beta.threads.create(messages=messages)
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions=custom_instructions,
                temperature=temperature
            )
            
            # 添加超时和状态检查
            max_retries = 30  # 30次尝试
            wait_seconds = 5  # 每次等待5秒
            current_attempt = 0
            
            print(f"[OpenAI] 开始监控运行状态 (ID: {run.id})")
            
            while current_attempt < max_retries:
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                print(f"[OpenAI] 运行状态检查 #{current_attempt+1}: {run.status}")
                
                if run.status == "completed":
                    break
                elif run.status in ["failed", "cancelled", "expired"]:
                    print(f"[OpenAI] 运行异常终止，状态: {run.status}")
                    if run.last_error:
                        print(f"错误详情: {run.last_error.code} - {run.last_error.message}")
                    return (f"Error: 运行失败 ({run.status})",)
                
                # 添加等待间隔避免频繁请求
                time.sleep(wait_seconds)
                current_attempt += 1
            else:
                print(f"[OpenAI] 错误: 超过最大等待时间 ({max_retries*wait_seconds}秒)")
                return ("Error: 请求超时",)
            
            # 获取响应时添加排序
            messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                order="asc"  # 改为正序获取最新消息
            )

            # 添加更安全的响应解析
            response_text = ""
            if messages and hasattr(messages, 'data'):
                for message in reversed(messages.data):
                    # 添加空值检查
                    if not message or not hasattr(message, 'content'):
                        continue
                        
                    if message.role == "assistant":
                        for content in message.content:
                            # 添加内容类型检查
                            if not content:
                                continue
                                
                            if content.type == "text" and hasattr(content.text, 'value'):
                                # 新增文本清洗逻辑
                                clean_text = content.text.value
                                # 替换所有代码块标记
                                clean_text = clean_text.replace('```', '')
                                # 去除首尾空白和多余空行
                                clean_text = '\n'.join([line.strip() for line in clean_text.split('\n') if line.strip()])
                                response_text += clean_text + "\n"
                            elif content.type == "image_file":
                                print("[OpenAI] 检测到图像文件响应")
            
            # 添加最终响应验证
            if not response_text.strip():
                print("[OpenAI] 警告: 未收到有效文本响应")
                return ("Error: 空响应",)
            
            return (response_text.strip(),)
            
        except Exception as e:
            print(f"[GPTs Error] 调用失败: {str(e)}")
            return (f"Error: {str(e)}",)
        finally:
            # 仅保留finally中的删除逻辑
            if file_info and file_info.get('bucket') and file_info.get('key'):
                print(f"[S3 Cleanup] 开始删除文件: {file_info['key']}")
                try:
                    self.delete_from_s3(s3_config, file_info)
                except Exception as e:
                    print(f"[S3 Cleanup Error] 最终清理失败: {str(e)}")

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
        try:
            print(f"[Config Debug] 正在加载OpenAI配置，来源: {config_source}, 文件名: {config_name}.yaml")
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
            
            # 打印解析后的配置结构
            print(f"[Config Debug] 解析后的配置结构: {config}")
            
            # 验证必要字段
            if not config.get('api_key'):
                raise ValueError("OpenAI API密钥不能为空")
            if not config.get('assistant_id'):
                raise ValueError("助手ID不能为空")
            
            print(f"[OpenAI Config] 加载配置: {config}")
            return (config,)  # 返回元组包裹的字典
        except Exception as e:
            print(f"[Config Error] 加载OpenAI配置失败: {str(e)}")
            return ({'api_key': '', 'assistant_id': ''},)  # 返回空配置元组

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