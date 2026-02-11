import wikipedia


class RobustWikipediaEnv:
    def __init__(self, max_chars=3000):
        # 功能点：指定 Wikipedia 语言为英文，与 HotpotQA 题目一致
        wikipedia.set_lang("en")
        # 功能点：单次返回的摘要最大字符数，由 config 的 observation_max_chars 传入
        self.max_chars = max_chars

    def search(self, query):
        """根据 query 查 Wikipedia，返回格式化的 Observation 字符串"""
        print(f"🌍 正在搜索 Wiki: {query}")
        try:
            # 功能点：用 Wikipedia API 做关键词搜索，得到候选页面标题列表
            results = wikipedia.search(query)
            if not results:
                return f"Observation: No Wikipedia page found for '{query}'. Try a different keyword."

            page_title = results[0]
            try:
                # 功能点：取第一个候选页面，拉取摘要并截断到 max_chars，换行改为空格
                page = wikipedia.page(page_title, auto_suggest=False)
                content = page.summary[: self.max_chars].replace("\n", " ")
                return f"Observation: [Title: {page_title}] {content}..."
            except wikipedia.exceptions.DisambiguationError as e:
                # 功能点：歧义页时返回可选条目（最多 5 个），供模型换更精确的搜索词
                return f"Observation: Ambiguous term '{query}'. Options: {', '.join(e.options[:5])}"
            except wikipedia.exceptions.PageError:
                return f"Observation: Page '{page_title}' does not exist."

        except Exception as e:
            # 功能点：网络或请求异常时返回统一错误信息
            return f"Observation: Wikipedia Connection Error: {str(e)}. (Check your network!)"
