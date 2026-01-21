# from API_server.aliyun_model_utils import *
from openai import OpenAI
import os
import time
import uuid
import json
import re
import pandas as pd


def get_response_from_aliyunAPI(
          messages,  # 聊天消息
          yun_model_if_stream=True, 
          yun_model_api_id=os.getenv("ALIYUN_ACCESS_API_ID"), 
          yun_model_api_key=os.getenv("ALIYUN_ACCESS_API_KEY"), 
          yun_model_name="qwen3-235b-a22b",
          stream_options={"include_usage": True},
          modalities=["text"],
          temperature=None,
          top_p=None,
          top_k=None,
          presence_penalty=None,
          response_format=None,
          max_tokens=None,
          n=1,
          enable_thinking=True,
          thinking_budget=None,
          seed=None,
          stop=None,
          tools=None,  # 工具调用参数，默认为None
          tool_choice=None,
          parallel_tool_calls=None,
          # translation_options=None,
          enable_search=True,
          forced_search=None,
          search_strategy=None,
          ):
    """
    使用方法：https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.bdd41d1cky7RPB#a75fcbc1dchyc

    先定义  messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': '你是谁？'}]

    调用get_response_from_aliyunAPI, completion = get_response_from_aliyunAPI(*****)

    
    "qwen3-235b-a22b" 
    1.普通：直接通过.model_dump_json()获取json格式的response 打印complection.model_dump_json()

    2.流式：for循环获取chunk： for chunk in complection得到chunk, 然后逐个得到chunk.model_dump_json()，可以for循环打印chunk.model_dump_json()

    
    "deepseek-r1"
    1.普通：通过reasoning_content字段打印思考过程print(completion.choices[0].message.reasoning_content)；# 通过content字段打印最终答案print(completion.choices[0].message.content)

    2.流式：for循环获取chunk： for chunk in complection得到chunk, chunk.choices[0].delta里面包含.reasoning_content和.content，有时当前chunk中content为空，可能正在思考中，就先打印chunk.choices[0].delta.reasoning_content，等到content不为空时，才打印chunk.choices[0].delta.content

    其他参考资料：

    https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.bdd41d1cky7RPB#f0b9c155ad0e0

    https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.59b625625N88kI

    https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.0.0.23b51d1cBV7hr1#dad2dbe656yhp

    https://help.aliyun.com/zh/model-studio/deep-thinking?spm=a2c4g.11186623.0.0.23b51d1cBV7hr1#62c72012bc2sw


    Args:
        messages: array of dicts, 消息列表，每个消息包含角色（role）和内容（content）,eg [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': '你是谁？'}]
        yun_model_if_stream: 是否使用流式输出
        yun_model_api_id: 阿里云API ID
        yun_model_api_key: 阿里云API Key
        yun_model_name: 模型名称, eg "qwen3-235b-a22b" 或 "deepseek-r1" 或 "qwen-plus-latest"
        stream_options: 当启用流式输出时，可通过将本参数设置为{"include_usage": true}，在输出的最后一行显示所使用的Token数。
        modalities: array （可选）默认值为["text"]。["text","audio"]：输出文本与音频；;["text"]：输出文本。
        temperature: 采样温度，控制模型生成文本的多样性。temperature越高，生成的文本更多样，反之，生成的文本更确定。取值范围： [0, 2)不建议更改
        top_p: 核采样的概率阈值，控制模型生成文本的多样性。top_p越高，生成的文本更多样。反之，生成的文本更确定。取值范围：（0,1.0]由于temperature与top_p均可以控制生成文本的多样性，因此建议您只设置其中一个值。也不建议更改
        top_k: 生成过程中采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个Token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。取值为None或当top_k大于100时，表示不启用top_k策略，此时仅有top_p策略生效。
        presence_penalty: 控制模型生成文本时的内容重复度。取值范围：[-2.0, 2.0]。正数会减少重复度，负数会增加重复度。适用场景：较高的presence_penalty适用于要求多样性、趣味性或创造性的场景，如创意写作或头脑风暴。较低的presence_penalty适用于要求一致性或专业术语的场景，如技术文档或其他正式文档。
        response_format:  默认值为{"type": "text"}。返回内容的格式。可选值：{"type": "text"}或{"type": "json_object"}。设置为{"type": "json_object"}时会输出标准格式的JSON字符串。使用方法请参见：结构化输出https://help.aliyun.com/zh/model-studio/json-mode?spm=a2c4g.11186623.0.0.418e2117VwLbyq
        max_tokens: 本次请求返回的最大 Token 数。
        n: 默认值为1。生成响应的个数，取值范围是1-4。对于需要生成多个响应的场景（如创意写作、广告文案等），可以设置较大的 n 值。
        enable_thinking: 默认值为False。是否启用思考模式，如果启用，则模型将生成一个额外的响应，用于描述模型正在思考的进度。
        thinking_budget: 思考过程的最大长度，只在enable_thinking为true时生效。适用于 Qwen3 的商业版与开源版模型。详情请参见限制思考长度https://help.aliyun.com/zh/model-studio/deep-thinking?spm=a2c4g.11186623.0.0.418e2117VwLbyq#e7c0002fe4meu
        seed: 随机种子，取值范围是0-4294967295。设置随机种子可以使模型生成的结果可复现。
        stop: string 或 array停止符，使用stop参数后，当模型生成的文本即将包含指定的字符串或token_id时，将自动停止生成。您可以在stop参数中传入敏感词来控制模型的输出。
        tools: array，可供模型调用的工具数组，可以包含一个或多个工具对象。一次Function Calling流程模型会从中选择一个工具（开启parallel_tool_calls可以选择多个工具）。属性包括：    - type: 工具类型，取值为"function"。    - function: 工具的具体定义，包括名称（name）、描述（description）和参数（parameters）。
        tool_choice: "auto"表示由大模型进行工具策略的选择。"none"如无论输入都不会进行工具调用。{"type": "function", "function": {"name": "the_function_to_call"}}如果您希望对于某一类问题，Function Calling 能够强制调用某个工具，可以设定tool_choice参数为{"type": "function", "function": {"name": "the_function_to_call"}}，其中the_function_to_call是您指定的工具函数名称。
        parallel_tool_calls: 是否开启并行工具调用。参数为true时开启，为false时不开启。并行工具调用详情请参见：并行工具调用。https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.0.0.418e2117VwLbyq#cb6b5c484bt4x
        # translation_options: 当您使用翻译模型时需要配置的翻译参数。
        enable_search: 是否启用网络搜索功能。默认值为false。启用后，模型可以通过网络搜索获取最新信息。
        forced_search: 是否强制开启搜索。
        search_strategy: 搜索策略。"standard"：在请求时搜索5条互联网信息；"pro"：在请求时搜索10条互联网信息。
    """
    if yun_model_name == "deepseek-r1":
            print("DeepSeek-R1 类模型不支持的功能:Function Calling、JSON Output、对话前缀续写、上下文硬盘缓存、网络搜索;不支持的参数:temperature、top_p、presence_penalty、frequency_penalty、logprobs、top_logprobs,设置这些参数都不会生效，即使没有输出错误提示。")

    client = OpenAI(
        api_key=yun_model_api_key,
        # api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请用阿里云百炼API Key将本行替换为：api_key="sk-xxx"
        base_url=yun_model_api_id,
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    
    search_strategy = None
    if forced_search is not None and search_strategy is not None:
        search_options = {"forced_search": forced_search, "search_strategy": search_strategy, "enable_citation": True,}
    else:
        search_options = None

    extra_body={}
    if enable_thinking is not None:    extra_body["enable_thinking"] = enable_thinking
    if thinking_budget is not None: extra_body["thinking_budget"] = thinking_budget
    if enable_search is not None:
        extra_body["enable_search"] = enable_search
        if search_options is not None:
            extra_body["search_options"] = search_options
    if top_k is not None: extra_body["top_k"] = top_k

    params = {
        "model": yun_model_name,
        "messages": messages,
        "stream": yun_model_if_stream,
    }
    if extra_body is not {}: params["extra_body"] = extra_body
    if stream_options is not None and params.get("stream") is True: params["stream_options"] = stream_options
    if modalities is not None: params["modalities"] = modalities
    if temperature is not None: params["temperature"] = temperature
    if top_p is not None: params["top_p"] = top_p
    if presence_penalty is not None: params["presence_penalty"] = presence_penalty
    if response_format is not None: params["response_format"] = response_format
    if max_tokens is not None: params["max_tokens"] = max_tokens
    if n is not None: params["n"] = n
    # if enable_thinking is not None: params["enable_thinking"] = enable_thinking   # 通过 Python SDK 调用时，请通过extra_body配置。配置方式为：extra_body={"enable_thinking": xxx}。
    # if thinking_budget is not None: params["thinking_budget"] = thinking_budget   # 通过 Python SDK 调用时，请通过extra_body配置。配置方式为：extra_body={"enable_thinking": xxx}。
    if seed is not None: params["seed"] = seed
    if stop is not None: params["stop"] = stop
    if tools is not None: params["tools"] = tools
    if tool_choice is not None: params["tool_choice"] = tool_choice
    if parallel_tool_calls is not None: params["parallel_tool_calls"] = parallel_tool_calls
    # if translation_options is not None: params["translation_options"] = translation_options

    completion = client.chat.completions.create(**params)

    return completion





# def def_get_aliyun_response(aliyuns_access_if_stream=True):
#     """
#     https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.help-menu-2400256.d_2_1_0.418e2117VwLbyq&scm=20140722.H_2712576._.OR_help-T_cn~zh-V_1
    
#     aliyuns_access_if_stream=True，则返回流式输出，否则返回普通输出

#     普通输出使用方法：
#     # if __name__ == '__main__':
#         #     print(get_response(***).model_dump_json())
#         # # 运行代码可以获得以下结果：
#         # {
#         #     "id": "chatcmpl-xxx",
#         #     "choices": [
#         #         {
#         #             "finish_reason": "stop",
#         #             "index": 0,
#         #             "logprobs": null,
#         #             "message": {
#         #                 "content": "我是来自阿里云的超大规模预训练模型，我叫通义千问。",
#         #                 "role": "assistant",
#         #                 "function_call": null,
#         #                 "tool_calls": null
#         #             }
#         #         }
#         #     ],
#         #     "created": 1716430652,
#         #     "model": "qwen-plus",
#         #     "object": "chat.completion",
#         #     "system_fingerprint": null,
#         #     "usage": {
#         #         "completion_tokens": 18,
#         #         "prompt_tokens": 22,
#         #         "total_tokens": 40
#         #     }
#         # }

#     # 流式输出使用方法：
#     # # 调用方式
#         # if __name__ == '__main__':
#         #     for chunk in get_response(***):
#         #        print(chunk.model_dump_json())
#         # # 运行代码可以获得以下结果：
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"","function_call":null,"role":"assistant","tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"我是","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"来自","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"阿里","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"云的大规模语言模型","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"，我叫通义千问。","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"","function_call":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#         # {"id":"chatcmpl-xxx","choices":[],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":{"completion_tokens":16,"prompt_tokens":22,"total_tokens":38}}

#     """

#     if not aliyuns_access_if_stream:

#         # 普通输出
#         from openai import OpenAI
#         import os
#         def get_response(messages, aliyuns_access_api_id, aliyuns_access_api_key, model="qwen-plus"):
#             """
#             messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
#                         {'role': 'user', 'content': '你是谁？'}]

#             return completion, 使用时一般是complection.model_dump_json()
#                 普通输出使用方法：
#                 # if __name__ == '__main__':
#                     #     print(get_response(***).model_dump_json())
#                     # # 运行代码可以获得以下结果：
#                     # {
#                     #     "id": "chatcmpl-xxx",
#                     #     "choices": [
#                     #         {
#                     #             "finish_reason": "stop",
#                     #             "index": 0,
#                     #             "logprobs": null,
#                     #             "message": {
#                     #                 "content": "我是来自阿里云的超大规模预训练模型，我叫通义千问。",
#                     #                 "role": "assistant",
#                     #                 "function_call": null,
#                     #                 "tool_calls": null
#                     #             }
#                     #         }
#                     #     ],
#                     #     "created": 1716430652,
#                     #     "model": "qwen-plus",
#                     #     "object": "chat.completion",
#                     #     "system_fingerprint": null,
#                     #     "usage": {
#                     #         "completion_tokens": 18,
#                     #         "prompt_tokens": 22,
#                     #         "total_tokens": 40
#                     #     }
#                     # }

#             """
#             client = OpenAI(
#                 api_key=aliyuns_access_api_key,
#                 # api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请用阿里云百炼API Key将本行替换为：api_key="sk-xxx"
#                 base_url=aliyuns_access_api_id,
#                 # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
#                 )
#             completion = client.chat.completions.create(
#                 model=model,  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#                 # messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
#                 #         {'role': 'user', 'content': '你是谁？'}]
#                 messages=messages,
#                 )
#             # print(completion.model_dump_json())
#             return completion
        


#     else:
#         # 流式输出
#         from openai import OpenAI
#         import os
#         def get_response(messages, aliyuns_access_api_id, aliyuns_access_api_key, model="qwen-plus"):
#             """
#             messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
#                         {'role': 'user', 'content': '你是谁？'}]
            
#             return completion, 使用时一般是for chunk in complection得到chunk, 然后逐个chunk.model_dump_json()
#                 # 使用方法
#                 #     if __name__ == '__main__':
#                 #         for chunk in get_response(***):
#                 #            print(chunk.model_dump_json())
#                 # # 运行代码可以获得以下结果：
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"","function_call":null,"role":"assistant","tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"我是","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"来自","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"阿里","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"云的大规模语言模型","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"，我叫通义千问。","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"","function_call":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}
#                 # {"id":"chatcmpl-xxx","choices":[],"created":1719286190,"model":"qwen-plus","object":"chat.completion.chunk","system_fingerprint":null,"usage":{"completion_tokens":16,"prompt_tokens":22,"total_tokens":38}}
#             """
#             client = OpenAI(
#                 api_key=aliyuns_access_api_key,
#                 # api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请用阿里云百炼API Key将本行替换为：api_key="sk-xxx"
#                 base_url=aliyuns_access_api_id,
#                 # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
#             )
#             completion = client.chat.completions.create(
#                 model=model,  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#                 # messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
#                 #         {'role': 'user', 'content': '你是谁？'}]
#                 messages=messages,
#                 stream=True,
#                 # 通过以下设置，在流式输出的最后一行展示token使用信息
#                 stream_options={"include_usage": True}
#                 # 当启用流式输出时，可通过将本参数设置为{"include_usage": true}，在输出的最后一行显示所使用的Token数。
#                 )
#             # for chunk in completion:
#             #     print(chunk.model_dump_json())
#             return completion
        
#     return get_response




def stream_manual_extract_think_chunks_generate(response, data):
    """
    生成流式输出的chunk，适用于自己微调的模型，会利用<think>和</think>标签来区分思考部分和回答部分。
    """
    # 发送角色信息
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": data.get('model', "local-model"),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # 检查响应是否包含思考部分
    if ("<think>" in response) and ("</think>" in response):
        # 提取思考部分
        thinking_part = response.split("<think>", 1)[1].split("</think>")[0].strip()
            # <think>后面的第一行删掉
        thinking_part = thinking_part.split('\n', 1)[-1].strip()  # 删除思考部分的第一行

        # 提取回答部分
        answer_part = response.split("</think>", 1)[1].strip()
        
        # 发送思考部分，使用特殊格式标记
        think_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get('model', "local-model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        # "content": "",
                        "reasoning": thinking_part  # 添加思考部分作为单独属性
                        # "content": f"<thinking>\n{thinking_part}\n</thinking>\n\n"
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(think_chunk)}\n\n"
        
        # 发送回答部分
        answer_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get('model', "local-model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": answer_part
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(answer_chunk)}\n\n"
    else:
        # 没有思考部分，直接发送内容
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get('model', "local-model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": response
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # 发送完成信号
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": data.get('model', "local-model"),
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # 结束标记
    yield "data: [DONE]\n\n"




def stream_manual_no_think_chunks_generate(response, data):
    """
    生成流式输出的chunk，适用于手动编制的回答，会直接把编制的字符串以流式形式返回，不会 使用<think>和</think>标签来区分思考部分和回答部分。
    """
    # 发送角色信息
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": data.get('model', "local-model"),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # 没有思考部分，直接发送内容
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": data.get('model', "local-model"),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": response
                },
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # 发送完成信号
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": data.get('model', "local-model"),
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # 结束标记
    yield "data: [DONE]\n\n"





def stream_Yun_chunks_generate_4_2Process(process, shared_dict, data, model_name, heartbeat_interval=50):
    """
    生成流式输出的chunk，适用于阿里云模型的API调用，默认阿里云的回答已经符合OpenAI格式，会直接遍历其中的chunk输出。也 不会 使用<think>和</think>标签来区分思考部分和回答部分。

    chat响应chunk对象（流式输出）：https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.59b625625N88kI
    chat响应对象（非流式输出）：https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.59b625625N88kI

    需要遍历他这个chunk，依次yield

    args:
    - process: 进程对象，用于检查是否需要发送心跳信号。
    - shared_dict: 共享字典，用于存储子进程的结果。
    - data: 包含模型相关信息的字典，通常包含模型名称等。
    - model_name: 模型名称，用于标识当前使用的模型。
    - heartbeat_interval: 心跳间隔时间，单位为秒，默认值为20秒。
    """
    ### 下面是Qwen 生成的OpenAI格式的流式输出chunk：
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":"assistant","tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"我是通义千问","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":17,"prompt_tokens":22,"total_tokens":39,"completion_tokens_details":null,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":0}}}

    last_hb = time.time()

    # 子进程存活且结果未写入时持续循环发心跳
    while process.is_alive() and shared_dict['response'] is None:
        if time.time() - last_hb > heartbeat_interval:
            yield ": ping\n\n"     # SSE 注释／心跳
            # 如果response为空，说明模型训练正在进行，此次tool调用只是客户端因为较长时间没得到响应，重复询问、调用了同样的tool，因此不需执行、没结果
            # # 发送角色信息
            # chunk = {
            #     "id": f"chatcmpl-{uuid.uuid4()}",
            #     "object": "chat.completion.chunk",
            #     "created": int(time.time()),
            #     "model": data.get('model', "local-model"),
            #     "choices": [
            #         {
            #             "index": 0,
            #             "delta": {
            #                 "role": "assistant"
            #             },
            #             "finish_reason": None
            #         }
            #     ]
            # }
            # yield f"data: {json.dumps(chunk)}\n\n"
            
            # # 发送思考部分，使用特殊格式标记
            # think_chunk = {
            #     "id": f"chatcmpl-{uuid.uuid4()}",
            #     "object": "chat.completion.chunk",
            #     "created": int(time.time()),
            #     "model": data.get('model', "local-model"),
            #     "choices": [
            #         {
            #             "index": 0,
            #             "delta": {
            #                 # "content": "",
            #                 "reasoning": "working..."
            #                 # "content": f"<thinking>\n{thinking_part}\n</thinking>\n\n"
            #             },
            #             "finish_reason": None
            #         }
            #     ]
            # }
            # yield f"data: {json.dumps(think_chunk)}\n\n"
            
            last_hb = time.time()
        time.sleep(5)  # 每5秒检查一次子进程状态

    process.join()  # 确保子进程退出

    # 如果写入了结果，就真正返回给客户端
    if shared_dict['response'] is not None:
        response = shared_dict['response']
        for chunk in response:
            # 将chunk对象转换为JSON格式并发送
            # yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {chunk.model_dump_json()}\n\n"
        # 响应结束
        yield "data: [DONE]\n\n"
    else:
        # 异常备用分支
        yield "data: {\"status\":\"error\",\"message\":\"子进程异常退出\"}\n\n"





def stream_Yun_chunks_generate(response, data, model_name):
    """
    生成流式输出的chunk，适用于阿里云模型的API调用，默认阿里云的回答已经符合OpenAI格式，会直接遍历其中的chunk输出。也 不会 使用<think>和</think>标签来区分思考部分和回答部分。

    chat响应chunk对象（流式输出）：https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.59b625625N88kI
    chat响应对象（非流式输出）：https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.59b625625N88kI

    需要遍历他这个chunk，依次yield
    """
    ### 下面是Qwen 生成的OpenAI格式的流式输出chunk：
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":"assistant","tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"我是通义千问","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":17,"prompt_tokens":22,"total_tokens":39,"completion_tokens_details":null,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":0}}}
    if response != []:
        for chunk in response:
            # 将chunk对象转换为JSON格式并发送
            # yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {chunk.model_dump_json()}\n\n"
        # 响应结束
        yield "data: [DONE]\n\n"
    else:
        # 如果response为空，说明模型训练正在进行，此次tool调用只是客户端因为较长时间没得到响应，重复询问、调用了同样的tool，因此不需执行、没结果
        # 发送角色信息
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get('model', "local-model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # 发送思考部分，使用特殊格式标记
        think_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get('model', "local-model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        # "content": "",
                        "reasoning": "该操作正在执行中，模型训练耗时较长，请耐心等待。"
                        # "content": f"<thinking>\n{thinking_part}\n</thinking>\n\n"
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(think_chunk)}\n\n"





def non_stream_json_generate(response, data, model_name):
    """
    示例: chat响应对象(非流式输出): https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.59b625625N88kI
    """
    normal_json = {
                "id": f"chatcmpl-{uuid.uuid4()}",  # 唯一标识符
                "object": "chat.completion",        # 对象类型
                "created": int(time.time()),        # 创建时间戳
                "model": data.get('model', model_name),  # 模型名称
                "system_fingerprint": None,           # 系统指纹
                "choices": [
                    {
                        "index": 0,                 # 选择索引
                        "message": {
                            "role": "assistant",    # 消息角色
                            "content": response # 生成的回复内容
                        },
                        "finish_reason": "stop"     # 结束原因
                    }
                ],
                # "usage": {
                #     "prompt_tokens": prompt_length,    # 提示使用的标记数
                #     "completion_tokens": len(generated_tokens), # 回复使用的标记数
                #     "total_tokens": prompt_length + len(generated_tokens) # 总标记数
                # }
                }
    return normal_json





def get_ADfile_path_from_history(data):
    """
    从历史对话里面提取出异常检测文件的路径
    """
    pattern = r'/data/[\w./ -]*\.csv'
    if isinstance(data, str):
        # 直接是字符串格式的历史对话
        matches = re.findall(pattern, data)
        if matches:
            return matches[-1].strip()
        else:
            return "File Path not Provided: The path of the working condition identification, anomaly detection or fault diagnosis file was not found. Please check whether this question or historical conversation content contains this information."
    else:
        # Cherry Studio/OpenAI格式 - 处理对话历史
        messages = data.get('messages', [])
        for message in messages[::-1]:
            role = message.get('role', '')
            if role == 'user':
                content = message.get('content', '')
                # 提取csv文件路径
                match = re.search(pattern, content)
                csv_path = match.group(0).strip() if match else None
                if csv_path is not None:
                    return csv_path
        return "File Path not Provided: The path of the working condition identification, anomaly detection or fault diagnosis file was not found. Please check whether this question or historical conversation content contains this information."


def get_user_time_from_question(question):
    """
    从用户提问中提取时间字符串
    假设用户提问里面有：<2024/10/18 18:42:01 - 2024/10/18 18:43:01>
    提取出来开始和结束时间
    也有可能是 ['2024-10-18 19:24:12', '2024-10-18 19:24:17'] 
    
    """
    start_time = None
    end_time = None

    # for pattern in [
    #     r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})',
    #     r'(\d{4}/\d{2}/\d{2}\d{2}:\d{2}:\d{2})',
    #     r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
    #     r'(\d{4}-\d{2}-\d{2}\d{2}:\d{2}:\d{2})'
    # ]:
    #     match = re.search(pattern, question)
    #     if match:
    #         start_time_str = match.group(1)
    #         start_time = pd.to_datetime(start_time_str)
    #         break

    # for pattern in [
    #     r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) -',
    #     r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})-',
    #     r'(\d{4}/\d{2}/\d{2}\d{2}:\d{2}:\d{2})-',
    #     r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) -',
    #     r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})-',
    #     r'(\d{4}-\d{2}-\d{2}\d{2}:\d{2}:\d{2})-'
    # ]:
    #     match = re.search(pattern, question)
    #     if match:
    #         start_time_str = match.group(1)
    #         # 去掉后面的横杠
    #         start_time_str = start_time_str.strip('-').strip()
    #         # 转化为时间对象
    #         start_time = pd.to_datetime(start_time_str)
    #         break

    # for pattern in [
    #     r'- (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})',
    #     r'-(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})',
    #     r'-(\d{4}/\d{2}/\d{2}\d{2}:\d{2}:\d{2})',
    #     r'- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
    #     r'-(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
    #     r'-(\d{4}-\d{2}-\d{2}\d{2}:\d{2}:\d{2})'
    # ]:
    #     match = re.search(pattern, question)
    #     if match:
    #         end_time_str = match.group(1)
    #         # 去掉前面的横杠
    #         end_time_str = end_time_str.strip('-').strip()
    #         # 转化为时间对象
    #         end_time = pd.to_datetime(end_time_str)
    #         break

    # 以上代码不匹配['2024-10-18 19:24:12', '2024-10-18 19:24:17'] 
    # 遂改策略：
    # 先把可能出现的“列表整体”或“尖括号范围整体”摘出来；
    # 统一用一条正则 \d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2} 抓出所有合法时间串；
    # 第一个当 start_time，最后一个个当 end_time，只有一个就返回 (time, None)。

    if not isinstance(question, str):
        question = str(question)

    # 匹配 YYYY-MM-DD hh:mm:ss 或 YYYY/MM/DD hh:mm:ss，允许日期/时间之间没有空格（\s*）
    datetime_pattern = re.compile(
        r'\d{4}[-/]\d{2}[-/]\d{2}\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?'
    )

    matches = datetime_pattern.findall(question)

    # 没找到任何时间
    if not matches:
        return None, None

    # 决定采用哪两个匹配：若正好2个，取前两个；若>2个，取第一个和最后一个（更常见于范围或日志）
    if len(matches) == 1:
        try:
            start_time = pd.to_datetime(matches[0].strip(" '\"<>[]()"))
            return start_time, None
        except Exception:
            return None, None

    if len(matches) == 2:
        first_str, second_str = matches[0], matches[1]
    else:
        first_str, second_str = matches[0], matches[-1]

    # 去除可能的引号或尖括号，然后转换
    first_str = first_str.strip(" '\"<>[]()")
    second_str = second_str.strip(" '\"<>[]()")

    try:
        start_time = pd.to_datetime(first_str)
        end_time = pd.to_datetime(second_str)
        return start_time, end_time
    except Exception:
        # 如果 pandas 解析失败，返回 None
        return None, None
    
    # # 5. 解析成 pd.Timestamp
    # start_time = pd.to_datetime(times[0]) if times else None
    # end_time   = pd.to_datetime(times[1]) if len(times) >= 2 else None

    # return start_time, end_time






def export_health_management_report_manual(User_question, data):
    """
    将data中的历史对话总结成健康管理报告，输出markdown

    : param User_question: 用户提问
    : param data: 包含历史对话的字典数据
    """
    messages = messages = data.get('messages', [])
    # 将所有'role': 'user'和'role': 'assistant'的问答content拼接后，每轮对话通过三个换行连接起来，打包成markdown
    user_contents = []
    for message in messages:
        if message.get('role') == 'user':
            content = message.get('content', '')
            if content:
                # 只需要## My question is:和## Reference Materials:之间的内容
                if "## My question is:" in content:
                    content = content.split("## My question is:")[1].strip()
                if "## Reference Materials:" in content:
                    content = content.split("## Reference Materials:")[0].strip()
                user_contents.append(content)

    assistant_contents = []
    for message in messages:
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            if content:
                assistant_contents.append(content.strip())

    # 两个列表个数统一一下，一般是user比assistant多一个，把user最后一个删了
    if len(user_contents) != len(assistant_contents):
        user_contents = user_contents[:-1]
        if len(user_contents) != len(assistant_contents):
            raise ValueError("用户提问和助手回答数量不匹配，请检查数据。")
        
    user_assistant_pairs = []
    for user_content, assistant_content in zip(user_contents, assistant_contents):
        user_assistant_pairs.append(f"## Question:\n{user_content}\n## Answer:\n{assistant_content}")

    # 将内容连接成一个字符串
    content = "\n\n\n".join(user_assistant_pairs)

    # 将用户提问和助手回答打包成markdown
    markdown_report = f"""
# Anomaly Detection Report


{content}
"""
    return markdown_report





def save_report_markdown(response, export_report_by_manual_or_yun):
    """
    保存报告为Markdown文件
    :param response: 报告内容
    :param export_report_by_manual_or_yun: 是否为手动编制的报告
    """
    # 先获取report的保存路径
    import yaml
    params_path = "/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml"
    with open(params_path, 'r', encoding='utf-8') as f:
        Yaml_params = yaml.safe_load(f)
    Common_config = Yaml_params["Common_configs"]
    temporal_block = Common_config['temporal_block']['value']
    spatial_block = Common_config['spatial_block']['value']
    Sample_config = Yaml_params[temporal_block]
    Sample_config.update(Yaml_params[spatial_block])
    "# Sample_config的优先级更高，若是共同的参数，则用Sample_config的参数覆盖Common_config的参数"
    for key, value in Sample_config.items():
        if key in Common_config:
            Common_config[key]["value"] = value["value"]
    Common_config = {k: v["value"] for k, v in Common_config.items()}  # 将字典转换为普通字典
    "# 更新exp_name和data_path和save_path"
    exp_name = (Common_config['Version'] + '_' + Common_config['Method'] + '_' + Common_config['data_name']
                + '_' + Common_config['Decompose'] + '_' + Common_config['TASK'])
    from AD_repository.main import set_args
    save_path = (Common_config['result_root_path'] + '/' + Common_config['Method']
                 + '/' + Common_config['data_name']
                 + '/' + exp_name)
    report_save_path = save_path + '/report'

    # 如果目录不存在，则创建目录
    import os
    if not os.path.exists(report_save_path):
        os.makedirs(report_save_path)
    file_name = "health_management_report.md"
    file_path = os.path.join(report_save_path, file_name)

    # {"id":"chatcmpl-e30f5ae7-3063-93c4-90fe-beb5f900bd57","choices":[{"delta":{"content":"叫通义千","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1735113344,"model":"qwen-plus","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}

    if export_report_by_manual_or_yun == "manual":
        markdown_content = response
    else:
        markdown_content = ""
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    markdown_content += delta.content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return markdown_content






def add_history_memory(messages, memory_conv_turns, history_messages):
    """
    添加历史对话到messages中，保留最近的memory_conv_turns轮对话

    :param messages: 当前对话消息列表
    :param memory_conv_turns: 保留的历史对话轮数
    :param history_messages: 历史对话数据

    :return: 更新后的messages列表
    """
    # 如果历史对话数据为空，则直接返回当前消息
    if not history_messages:
        return messages
    # 计算需要保留的历史对话数量
    num_history = min(len(history_messages), 2 * memory_conv_turns)
    # 获取最近的历史对话
    recent_history = history_messages[-num_history:]
    # 将历史对话添加到当前消息列表中
    messages = messages[:-1] + recent_history + [messages[-1]]
    
    return messages













