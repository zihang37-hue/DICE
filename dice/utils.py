import re


def extract_thought(content):
    """从模型输出中提取纯 Thought（去掉 Action 部分）"""
    if not content:
        return ""
    head = content.split("Action:")[0]
    if "Thought:" in head:
        head = head.split("Thought:", 1)[1]
    return head.strip()


def clean_demo_text(demo_text):
    """清洗检索到的demo，去除重复Action和Thought中夹带的Action"""
    lines = []
    last_action = None
    for line in demo_text.splitlines():
        if line.startswith("Thought:") and "Action:" in line:
            line = line.split("Action:")[0].rstrip()
        if line.startswith("Action:"):
            if last_action == line:
                continue
            last_action = line
        lines.append(line)
    return "\n".join(lines)


def normalize_react_demo(demo_text):
    """规范 demo 格式，避免编号/多余提示干扰主模型"""
    if not demo_text:
        return demo_text
    text = demo_text
    text = re.sub(r'(?m)^(Thought|Action|Observation)\s+\d+\s*:\s*', r'\1: ', text)
    text = re.sub(r'(?m)^Observation:\s*Observation:\s*', 'Observation: ', text)
    text = re.sub(r'(?m)^\(If this answers the question.*\)$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def normalize_answer(s):
    """标准化答案，用于 Exact Match 对比"""
    if not s:
        return ""
    s = s.lower().strip()
    for article in ["a ", "an ", "the "]:
        if s.startswith(article):
            s = s[len(article):]
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def exact_match(pred, gold):
    """判断预测答案是否和标准答案匹配"""
    return normalize_answer(pred) == normalize_answer(gold)
