PREDICTION_PROMPT = """Please extract the transferable knowledge the the agent need at the current state to solve the problem. The current state of the agent is given, which contains a sequence of Action and Observations, describing how the agent solve a question by thinking, acting and observing. Here are some examples.

Trajectory:

Question: Which magazine was started first Arthur's Magazine or First for Women?

Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.

Action 1: Search[Arthur's Magazine]

Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

Knowledge Required: Now I know the start year of the first magazine, so I will search for the start year of the second one. The knowledge I need is how to search the second magine in the question.

Trajectory:

Question: Which magazine was started first Arthur's Magazine or First for Women?

Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.

Action 1: Search[Arthur's Magazine]

Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.

Action 2: Search[First for Women]

Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.

Knowledge Required: Now I know the start year of both magazines in the question. I need to compare them to determine which one started first. The knowledge I need now is how to compare the features I found to make the correct conclusion.

Trajectory:

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?

Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.

Action 1: Search[Pavel Urysohn]

Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.

Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.

Action 2: Search[Leonid Levin]

Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.

Knowledge Required: Now I know the work of both persons in the question. Pavel Urysohn is a mathematician, Leonid Levin is mathematician and computer scientist. They are not exactly the same. The knowledge I need now is how to determine if these occupations can be considered as the same type, so that I can make the correct conclusion.

Trajectory:
Question: {task}
{history}

Knowledge Required:"""

REACT_SYSTEM_PROMPT = """You are a smart agent answering questions using Wikipedia search.
You MUST follow the Thought-Action format strictly. Every response MUST contain exactly ONE "Action:" line.

RULES:
1. Search for ONE entity at a time (e.g., Search[France], NOT Search[France,capital]).
2. After each Observation, check if it contains the answer. If yes, immediately use Action: Finish[...].
3. If a search fails or returns irrelevant results, try a DIFFERENT and SIMPLER keyword.
4. After 2-3 searches, you likely have enough information. Combine what you learned and use Action: Finish[...].
5. The answer should be SHORT and DIRECT (e.g., "Paris", "George Washington", "Pacific Ocean").
6. NEVER output Finish[Unknown] if you have seen ANY useful information in previous Observations.
7. ALWAYS output an Action line. Your response MUST end with: Action: Search[entity] or Action: Finish[...]
8. Inside Finish[...], include ONLY the final answer text (no extra words or explanations).

FORMAT (you MUST follow this exactly):
Thought: [your reasoning, referencing information from Observations]
Action: Search[entity] or Finish[...]

Examples:
{demonstrations}

Now solve this task step by step.
Task: {task}
"""
