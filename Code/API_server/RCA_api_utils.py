import xml.etree.ElementTree as ET
import json




def get_RAG_materials_def_xml(Keywords):
    """
    编写XML格式的RAG搜索函数，用来让Cherry Studio查询RAG材料。
    :param Keywords: 关键词
    :return: 
    """
    # 以下是Cherry Studio提供的RAG搜索函数调用方法：

    # Tool Use Formatting

    # Tool use is formatted using XML-style tags.
    # The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags.

    # Here's the structure:

    # <tool_use>
    # {tool_name}
    # {json_arguments}
    # </tool_use>

    # The tool name should be the exact name of the tool you are using,
    # and the arguments should be a JSON object containing the parameters required by that tool.

    # For example:

    # <tool_use>
    # python_interpreter
    # {"code": "5 + 3 + 1294.678"}
    # </tool_use>

    # The user will respond with the result of the tool use, which should be formatted as follows:

    # <tool_use_result>
    # {tool_name}
    # {result}
    # </tool_use_result>

    # The result should be a string, which can represent a file or any other output type.
    # You can use this result as input for the next action.

    # You only have access to these tools:\n<tools>\n\n<tool>\n  <name>builtin_knowledge_search</name>\n  <description>Knowledge base search tool for retrieving information from user\'s private knowledge base. This searches your local collection of documents, web content, notes, and other materials you have stored.\n\nThis tool has been configured with search parameters based on the conversation context:\n- Prepared queries: "检测到航天器电源系统发生了【负载2开路】故障，请查询我的知识库，执行根因分析及维护决策任务。An fault of "Load 2 open circuit" has been detected in the spacecraft power system. Please consult my knowledge base and carry out the task of root cause analysis and maintenance decision-making.\n"\n- Query rewrite: "检测到航天器电源系统发生了【负载2开路】故障，请查询我的知识库，执行根因分析及维护决策任务。An fault of "Load 2 open circuit" has been detected in the spacecraft power system. Please consult my knowledge base and carry out the task of root cause analysis and maintenance decision-making.\n"\n\nYou can use this tool as-is, or provide additionalContext to refine the search focus within the knowledge base.</description>\n  <arguments>\n    {"~standard":{"vendor":"zod","version":1},"def":{"type":"object","shape":{"additionalContext":{"~standard":{"vendor":"zod","version":1},"def":{"type":"optional","innerType":{"~standard":{"vendor":"zod","version":1},"def":{"type":"string"},"type":"string","format":null,"minLength":null,"maxLength":null}},"type":"optional"}}},"type":"object"}\n  </arguments>\n</tool>\n\n</tools>\n\n## Tool Use Rules\nHere are the rules you should always follow to solve your task:\n1. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.\n2. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.\n3. If no tool call is needed, just answer the question directly.\n4. Never re-do a tool call that you previously did with the exact same parameters.\n5. For tool use, MAKE SURE use XML tag format as shown in the examples above. Do not use any other format.\n\n# User Instructions\n\n\nNow Begin! If you solve the task correctly, you will receive a reward of $1,000,000.'

    # 构造 XML
    tool_name = "builtin_knowledge_search"
    arguments = {"additionalContext": Keywords}

    # 创建 XML 结构
    root = ET.Element("tool")
    name = ET.SubElement(root, "name")
    name.text = tool_name

    args = ET.SubElement(root, "arguments")
    # args.text = str(arguments)  # 注意：这里必须是字符串形式的字典
    args.text = json.dumps(arguments, ensure_ascii=False)  # 使用双引号且保留中文

    # 输出 XML 字符串
    # xml_str = ET.tostring(root, encoding="unicode")
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    """
    <tool>
        <name>builtin_knowledge_search</name>
        <arguments>{"additionalContext": "需要搜索的文本"}</arguments>
    </tool>
    """
    
    return xml_str






def get_RAG_materials_from_XML_result(xml_result):
    """
    从XML格式的RAG搜索结果中提取RAG材料内容。
    :param xml_result: XML格式的RAG搜索结果
    :return: 提取的RAG材料内容
    """
    # try:
    #     root = ET.fromstring(xml_result)
    #     # 假设结果在<result>标签内
    #     result = root.find('result')
    #     if result is not None:
    #         return result.text
    #     else:
    #         return "No result found in the provided XML."
    # except ET.ParseError:
    #     return "Invalid XML format."
    import re
    m = re.search(r"<result>(.*)</result>", xml_result, re.S)
    RAG_materials = m.group(1).strip() if m else ""
    # RAG_materials = prompt.split("<result>")[-1].split("</result>")[0].strip()
    return RAG_materials






def get_all_messages_for_RCA(User_prompt, User_question, RAG_materials, RCA_prompt_style):
    """
    获取所有消息内容，适用于RCA任务。
    :param User_prompt: 用户提示
    :param User_question: 用户问题
    :param RAG_materials: RAG材料
    :param RCA_prompt_style: RCA提示样式
    :return: 包含所有消息的列表
    """
    # 先把User_prompt, User_question, RAG_materials拼接起来，之间换行两次\n\n
    User_prompt_and_question = f"User_prompt:{User_prompt}\n\nUser_question:{User_question}\n\nRAG_materials:{RAG_materials}"

    # 如果User_question里面提供了完成RCA任务所需的回答模板和任务描述，那就不需要将RCA_prompt_style作为system contenct了
    if ('## 任务' in User_question and '## 回答模板' in User_question) or ('##任务' in User_question and '##回答模板' in User_question) or ('## Task' in User_question and '## Answer Template' in User_question) or ('##Task' in User_question and '##Answer Template' in User_question):
        System_prompt = "You are an expert in the operation and maintenance of spacecraft power systems. Please execute the root cause analysis and maintenance decision-making task"
    else:
        System_prompt = RCA_prompt_style

    messages = [
        {"role": "system", "content": System_prompt},
        {"role": "user", "content": User_prompt_and_question}
    ]
    
    return messages









