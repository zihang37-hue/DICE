import re


def extract_thought(content):
    """从模型输出中提取纯 Thought 文本（去掉 Action 及之后部分），用于写入 history"""
    if not content:
        return ""
    # 功能点：以 "Action:" 为界，只保留前半段
    head = content.split("Action:")[0]
    if "Thought:" in head:
        head = head.split("Thought:", 1)[1]
    return head.strip()


def clean_demo_text(demo_text):
    """清洗检索到的 demo：去掉 Thought 行里夹带的 Action、合并重复的 Action 行"""
    lines = []
    last_action = None
    for line in demo_text.splitlines():
        # 功能点：若 Thought 行内包含 "Action:"，只保留 Thought 部分，避免格式错乱
        if line.startswith("Thought:") and "Action:" in line:
            line = line.split("Action:")[0].rstrip()
        if line.startswith("Action:"):
            if last_action == line:
                continue
            last_action = line
        lines.append(line)
    return "\n".join(lines)


def normalize_react_demo(demo_text):
    """规范 demo 格式：去掉步骤编号、重复 Observation 前缀、末尾提示句、多余空行，避免干扰主模型"""
    if not demo_text:
        return demo_text
    text = demo_text
    # 功能点：Thought 1: / Action 2: 等统一为 Thought: / Action:
    text = re.sub(r'(?m)^(Thought|Action|Observation)\s+\d+\s*:\s*', r'\1: ', text)
    text = re.sub(r'(?m)^Observation:\s*Observation:\s*', 'Observation: ', text)
    text = re.sub(r'(?m)^\(If this answers the question.*\)$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def normalize_answer(s):
    """标准化答案字符串：小写、去首尾空白、去冠词、去标点、合并空白，用于 Exact Match 对比"""
    if not s:
        return ""
    s = s.lower().strip()
    # 功能点：去掉句首的 a / an / the
    for article in ["a ", "an ", "the "]:
        if s.startswith(article):
            s = s[len(article):]
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def exact_match(pred, gold):
    """判断预测答案与标准答案是否在标准化后完全一致"""
    return normalize_answer(pred) == normalize_answer(gold)
