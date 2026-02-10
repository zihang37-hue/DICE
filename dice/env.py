import wikipedia


class RobustWikipediaEnv:
    def __init__(self):
        wikipedia.set_lang("en")

    def search(self, query):
        print(f"🌍 正在搜索 Wiki: {query}")  # 搜索入口
        try:
            results = wikipedia.search(query)
            if not results:
                return f"Observation: No Wikipedia page found for '{query}'. Try a different keyword."  # 无结果

            page_title = results[0]
            try:
                page = wikipedia.page(page_title, auto_suggest=False)
                content = page.summary[:3000].replace("\n", " ")
                return f"Observation: [Title: {page_title}] {content}..."  # 返回摘要
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Observation: Ambiguous term '{query}'. Options: {', '.join(e.options[:5])}"  # 歧义页
            except wikipedia.exceptions.PageError:
                return f"Observation: Page '{page_title}' does not exist."  # 页面不存在

        except Exception as e:
            return f"Observation: Wikipedia Connection Error: {str(e)}. (Check your network!)"  # 网络/请求错误
