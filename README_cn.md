# Awesome 生成式人工智能 [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

> 一个经过精心整理的现代生成式人工智能项目和服务清单。

[English](README.md)/简体中文

生成式 AI 是一种通过在大量数据上训练的机器学习算法来生成原创内容（如图像、声音和文本）的技术。与其他形式的人工智能不同，它能够创造出独特且前所未见的输出，如照片级真实感图像、数字艺术、音乐和写作。这些输出通常具有独特的风格，甚至难以与人类创作的作品区分。生成式人工智能在艺术、娱乐、市场营销、学术研究和计算机科学等领域都有广泛应用。

欢迎为该列表贡献内容。在提交建议之前，请先查看 [贡献指南](CONTRIBUTING.md)，以确保您的条目符合标准。您可以通过 [pull requests](https://github.com/hifiveszu/awesome-generative-ai/pulls) 添加链接，或创建 [issue](https://github.com/hifiveszu/awesome-generative-ai/issues) 开启讨论。更多项目可在 [发现列表](DISCOVERIES.md) 中找到，我们会展示各种新兴的生成式人工智能项目。

## 目录

- [推荐阅读](#recommended-reading)
- [文本](#text)
- [编程](#coding)
- [智能体](#agents)
- [图像](#image)
- [视频](#video)
- [音频](#audio)
- [其他](#other)
- [学习资源](#learning-resources)
- [更多列表](#more-lists)

## 推荐阅读

- [大型语言模型将如何改变科学、社会和人工智能](https://hai.stanford.edu/news/how-large-language-models-will-transform-science-society-and-ai) - 一篇总结 GPT-3 模型能力和局限性及其潜在社会影响的文章。作者：Alex Tamkin 和 Deep Ganguli，2021 年 2 月 5 日。
- [生成式 AI：一个创造性的全新世界](https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/) - 对生成式 AI 行业的全面研究，提供历史视角和行业生态系统的深入分析。作者：Sonya Huang、Pat Grady 和 GPT-3，2022 年 9 月 19 日。
- [生成式 AI 的出道派对，硅谷的新狂热](https://www.nytimes.com/2022/10/21/technology/generative-ai.html) - 介绍生成式 AI 崛起的文章，尤其是 Stable Diffusion 图像生成器的成功及相关争议。纽约时报，2022 年 10 月 21 日。
- [AI 的新创造力引发硅谷淘金热](https://www.wired.com/story/ais-new-creative-streak-sparks-a-silicon-valley-gold-rush/) - 讨论生成式 AI 初创公司日益增长的热潮和投资，不同行业探索其潜在应用。Wired，2022 年 10 月 27 日。
- [ChatGPT 宣告一场思想革命](https://www.wsj.com/articles/artificial-intelligence-generative-ai-chatgpt-kissinger-84512912) - Henry Kissinger、Eric Schmidt 和 Daniel Huttenlocher 的评论文章。华尔街日报，2023 年 2 月 24 日。

### 里程碑

- [OpenAI API](https://openai.com/blog/openai-api/) - 基于 GPT-3 的文本生成通用 AI 模型 API 公告。OpenAI 博客，2020 年 6 月 11 日。
- [GitHub Copilot](https://github.blog/2021-06-29-introducing-github-copilot-ai-pair-programmer/) - 发布 Copilot，一款帮助你编写更好代码的 AI 编程助手。GitHub 博客，2021 年 6 月 29 日。
- [DALL·E 2](https://openai.com/blog/dall-e-2/) - 发布 DALL·E 2，高级图像生成系统，具备更高分辨率、扩展的图像生成能力和多种安全措施。OpenAI 博客，2022 年 4 月 6 日。
- [Stable Diffusion 公共发布](https://stability.ai/blog/stable-diffusion-public-release) - 发布 Stable Diffusion，一种基于 AI 的图像生成模型，在广泛的网络数据上训练，使用 Creative ML OpenRAIL-M 许可证。Stable Diffusion 博客，2022 年 8 月 22 日。
- [ChatGPT](https://openai.com/blog/chatgpt/) - 发布 ChatGPT，一种能回答追问、承认错误、挑战错误前提并拒绝不当请求的对话模型。OpenAI 博客，2022 年 11 月 30 日。
- [必应搜索](https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-your-copilot-for-the-web/) - 微软宣布新版必应搜索引擎，由下一代 OpenAI 模型驱动。微软博客，2023 年 2 月 7 日。
- [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) - Meta 发布的 650 亿参数的大型语言模型。Meta，2023 年 2 月 23 日。#opensource
- [GPT-4](https://openai.com/research/gpt-4) - 发布 GPT-4，一种大型多模态模型。OpenAI 博客，2023 年 3 月 14 日。
- [DALL·E 3](https://openai.com/index/dall-e-3/) - 发布 DALL·E 3 图像生成器。OpenAI 博客，2023 年 9 月 20 日。
- [Sora](https://openai.com/research/video-generation-models-as-world-simulators) - 发布 Sora，一个大型视频生成模型。OpenAI，2024 年 2 月 15 日。

## 文本

### 模型

- [OpenAI API](https://openai.com/api/) - OpenAI 的 API 提供最新的 GPT-5 模型的访问，支持多种自然语言任务，以及将自然语言翻译为代码的 Codex。
- [DeepSeek](https://www.deepseek.com/) - DeepSeek 提供多款开源大型语言模型（如 R1、V3、Coder V2、VL、Math 等），支持自然语言理解、编程、视觉语言和数学推理等多种任务。
- [Gopher](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval) - DeepMind 发布的 Gopher，是一个 2800 亿参数的语言模型。
- [OPT](https://huggingface.co/facebook/opt-350m) - Facebook 的开放预训练 Transformer（OPT）系列，仅解码器的预训练模型。[公告](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)。[OPT-175B 文本生成](https://opt.alpa.ai/) 由 Alpa 托管。
- [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom) - Hugging Face 的 BLOOM，是一个类似 GPT-3 的模型，训练数据包含 46 种语言和 13 种编程语言。#opensource
- [Llama](https://www.llama.com/) - Meta 的开源大型语言模型。#opensource
- [Claude](https://claude.ai/) - 来自 Anthropic 的 AI 助手 Claude。
- [Vicuna-13B](https://lmsys.org/blog/2023-03-30-vicuna/) - 一个基于 LLaMA 微调的开源聊天机器人，训练数据来自 ShareGPT 用户共享对话。#opensource
- [Mistral](https://mistral.ai/en/models) - Mistral AI 发布的前沿开源大语言模型。#opensource
- [Grok](https://grok.x.ai/) - xAI 发布的 LLM，带有 [开源代码](https://github.com/xai-org/grok-1) 和开放权重。#opensource
- [Qwen](https://qwenlm.github.io/) - 阿里云自主研发的一系列大语言模型。[#opensource](https://github.com/QwenLM/Qwen)

### 聊天机器人

- [ChatGPT](https://chatgpt.com/) - 由 OpenAI 开发的 ChatGPT，是一个能够进行对话式交互的大型语言模型。
- [Copilot](https://copilot.microsoft.com/) - 微软推出的日常 AI 助手。
- [Gemini](https://gemini.google.com/) - 由 Google Deepmind 开发的多模态大型语言模型家族。
- [Meta AI](https://www.meta.ai/) - Meta AI 助手，可完成任务、生成 AI 图片并获取答案，基于 Llama 大模型构建。
- [DeepSeek](https://www.deepseek.com/) - 面向企业、消费级和科研应用的前沿大模型。#开源
- [Character.AI](https://character.ai/) - Character.AI 让你创建角色并与之聊天。
- [Pi](https://pi.ai) - 一个个性化的数字 AI 助理平台。
- [Qwen](https://chat.qwenlm.ai/) - Qwen 聊天机器人，支持图像生成、文档处理、网页搜索集成、视频理解等功能。
- [Le Chat](https://chat.mistral.ai/) - 与 Mistral AI 的前沿语言模型聊天。

### 自定义 AI UI 界面

- [LibreChat](https://librechat.ai/) - 免费开源的 AI 助手聊天界面。 [#开源](https://github.com/danny-avila/LibreChat)。
- [Chatbot UI](https://www.chatbotui.com/) - 开源的 ChatGPT 用户界面。 [#开源](https://github.com/mckaywrigley/chatbot-ui)。

### 搜索引擎

- [Perplexity AI](https://www.perplexity.ai/) - AI 驱动的搜索工具。
- [Metaphor](https://metaphor.systems/) - 基于语言模型的搜索。
- [Phind](https://phind.com/) - 基于 AI 的搜索引擎。
- [You.com](https://you.com/) - 基于 AI 的搜索引擎，为用户提供个性化的搜索体验，同时保证数据 100% 隐私。
- [Komo](https://komo.ai/) - AI 驱动的搜索引擎。

### 本地搜索引擎

- [privateGPT](https://github.com/imartinez/privateGPT) - 在没有网络连接的情况下，利用大模型向文档提问。
- [quivr](https://github.com/StanGirard/quivr) - 把所有文件集中存放，并通过大模型和向量嵌入与之对话，成为你的第二大脑。

### 写作助手

- [Jasper](https://www.jasper.ai/) - 利用人工智能更快地创建内容。
- [Compose AI](https://www.compose.ai/) - 免费的 Chrome 插件，借助 AI 自动补全可将写作时间缩短 40%。
- [Rytr](https://rytr.me/) - AI 写作助手，帮助你创作高质量内容。
- [wordtune](https://www.wordtune.com/) - 个性化写作助手。
- [HyperWrite](https://hyperwriteai.com/) - 帮助你从想法到最终稿件更快完成写作。
- [Moonbeam](https://www.gomoonbeam.com/) - 在极短时间内写出更好的博客。
- [copy.ai](https://www.copy.ai/) - 用 AI 写出更好的营销文案和内容。
- [ChatSonic](https://writesonic.com/chat) - 支持文本与图像生成的 AI 助手。
- [Anyword](https://anyword.com/) - AI 写作助手，为任何人生成高效文案。
- [Hypotenuse AI](https://www.hypotenuse.ai/) - 将关键词转化为原创的文章、产品描述和社交媒体文案。
- [Lavender](https://www.lavender.ai/) - AI 邮件助手，帮助你更快获得回复。
- [Lex](https://lex.page/) - 内置 AI 的文字处理器，让你更快完成写作。
- [Jenni](https://jenni.ai/) - 高效写作助手，节省构思与写作时间。
- [LAIKA](https://www.writewithlaika.com/) - 基于你的写作风格训练 AI，成为个性化创意搭档。
- [QuillBot](https://quillbot.com) - AI 驱动的改写工具。
- [Postwise](https://postwise.ai/) - 用 AI 写推文、安排发布并增长粉丝。
- [Copysmith](https://copysmith.ai/) - 面向企业和电商的 AI 内容创作解决方案。

### ChatGPT 扩展

- [WebChatGPT](https://chrome.google.com/webstore/detail/webchatgpt-chatgpt-with-i/lpfemeioodjbpieminkklglpmhlngfcn) - 用网络搜索结果增强 ChatGPT 提示。
- [GPT for Sheets and Docs](https://workspace.google.com/marketplace/app/gpt_for_sheets_and_docs/677318054654) - Google Sheets 和 Docs 的 ChatGPT 扩展。
- [YouTube Summary with ChatGPT](https://chrome.google.com/webstore/detail/youtube-summary-with-chat/nmmicjeknamkfloonkhhcjmomieiodli) - 使用 ChatGPT 总结 YouTube 视频。
- [ChatGPT Prompt Genius](https://chrome.google.com/webstore/detail/chatgpt-prompt-genius/jjdnakkfjnnbbckhifcfchagnpofjffo) - 发现、分享、导入和使用最佳 ChatGPT 提示，并可本地保存聊天记录。
- [ChatGPT for Search Engines](https://chrome.google.com/webstore/detail/chatgpt-for-search-engine/feeonheemodpkdckaljcjogdncpiiban) - 在 Google、Bing 和 DuckDuckGo 搜索结果旁显示 ChatGPT 响应。
- [ShareGPT](https://sharegpt.com/) - 分享你的 ChatGPT 对话，并探索他人分享的内容。
- [Merlin](https://merlin.foyer.work/) - 在所有网站上使用的 ChatGPT Plus 扩展。
- [ChatGPT Writer](https://chatgptwriter.ai/) - 使用 ChatGPT 生成整封邮件和讯息。
- [ChatGPT for Jupyter](https://github.com/TiesdeKok/chat-gpt-jupyter-extension) - 为 Jupyter Notebook 和 Jupyter Lab 添加 ChatGPT 功能。
- [editGPT](https://www.editgpt.app/) - 在 ChatGPT 中轻松校对、编辑和跟踪内容修改。
- [Forefront](https://www.forefront.ai/) - 更好的 ChatGPT 使用体验。

### 效率工具

- [ChatPDF](https://www.chatpdf.com/) - 与任何 PDF 文件对话。
- [Mem](https://mem.ai/) - 全球首个 AI 驱动的个性化工作区，提升创造力、自动化日常任务并保持高效。
- [Taskade](https://www.taskade.com/) - 使用 Taskade AI 创建任务、笔记、结构化列表和思维导图。
- [Notion AI](https://www.notion.so/product/ai) - 更好更高效地撰写笔记和文档。
- [Nekton AI](https://nekton.ai) - 用 AI 自动化工作流程，用自然语言逐步描述即可。
- [Rewind](https://www.rewind.ai/) - 个性化 AI，基于你看过、说过或听过的一切构建。
- [NotebookLM](https://notebooklm.google/) - Google Gemini 驱动的在线研究与笔记工具，可与文档互动。

### 会议助手

- [Otter.ai](https://otter.ai/) - 会议助手，可录音、撰写笔记、自动捕捉幻灯片并生成摘要。
- [Cogram](https://www.cogram.com/) - 在虚拟会议中自动记录笔记，并识别待办事项。
- [Sybill](https://www.sybill.ai/) - 结合转录和情绪分析生成销售通话摘要，包括下一步行动、痛点和关注点。
- [Loopin AI](https://www.loopinhq.com/) - 协作型会议工作空间，可用 AI 记录、转录、总结会议，并自动在日历上整理会议笔记。
- [Fireflies.ai](https://fireflies.ai/) - 帮助团队转录、总结、搜索和分析语音对话。
- [Read AI](https://www.read.ai/) - AI 助手，提升会议、邮件和信息效率，通过摘要、内容发现和推荐提高工作效率。
- [Fireflies.ai](https://fireflies.ai/) - 转录、总结、搜索并分析团队所有对话。

### 学术研究

- [Elicit](https://elicit.org/) - 使用语言模型帮助自动化研究工作流，例如文献综述的部分流程。
- [genei](https://www.genei.io/) - 秒级摘要学术文章，节省 80% 的研究时间。
- [Explainpaper](https://www.explainpaper.com/) - 更高效地阅读学术论文，上传论文、标记难理解的文本并获得解释。
- [Galactica](https://galactica.org/) - 面向科学的大型语言模型，可总结学术文献、解决数学问题、生成 Wiki 条目、编写科学代码、注释分子与蛋白质等。 [模型 API](https://github.com/paperswithcode/galai)
- [Consensus](https://consensus.app/search/) - 使用 AI 在科学研究中寻找答案的搜索引擎。
- [Synthical](https://synthical.com/) - AI 驱动的协作研究环境。
- [scite](https://scite.ai/) - 用于发现和评估科学文章的平台。
- [SciSpace](https://typeset.io/) - 理解科学文献的 AI 研究助手。
- [STORM](https://storm.genie.stanford.edu/) - 基于 LLM 的知识整理系统，研究主题并生成带引用的完整报告。 [#开源](https://github.com/stanford-oval/storm/)

### 排行榜

- [Chatbot Arena](https://lmarena.ai/) - 由 UC Berkeley SkyLab 和 LMArena 研究人员主持的众包 AI 基准开放平台。
- [Artificial Analysis](https://artificialanalysis.ai/) - 提供客观基准和信息，帮助选择 AI 模型和托管服务。
- [imgsys](https://imgsys.org/rankings) - fal.ai 的生成式图像模型竞技场。
- [OpenRouter LLM Rankings][https://openrouter.ai/rankings] - 基于应用使用情况对语言模型进行排名和分析。

### 其他文本生成器

- [EmailTriager](https://www.emailtriager.com/) - 使用 AI 在后台自动撰写邮件回复。
- [AI Poem Generator](https://www.aipoemgenerator.org) - 根据文本提示生成任意主题的押韵美诗。

## 编程

### 编程助手

- [GitHub Copilot](https://github.com/features/copilot) - GitHub Copilot 使用 OpenAI Codex 在编辑器中实时提供代码和完整函数建议。
- [OpenAI Codex](https://platform.openai.com/docs/guides/code/) - OpenAI 的 AI 系统，可将自然语言转换为代码。
- [Ghostwriter](https://blog.replit.com/ai) - Replit 提供的 AI 编程助手。
- [Amazon Q](https://aws.amazon.com/q/) - AWS 生成式 AI 助手，帮助回答问题、编写代码和自动化任务。
- [tabnine](https://www.tabnine.com/) - 提供整行及完整函数的代码补全，加快编码速度。
- [Stenography](https://stenography.dev/) - 自动生成代码文档。
- [Mintlify](https://mintlify.com/) - AI 驱动的文档撰写工具。
- [Debuild](https://debuild.app/) - AI 驱动的低代码 Web 应用开发工具。
- [AI2sql](https://www.ai2sql.io/) - 无需了解 SQL，轻松生成高效、无错误的 SQL 查询。
- [CodiumAI](https://www.codium.ai/) - 在 IDE 内提供非平凡测试建议，确保提交代码时更有信心。
- [PR-Agent](https://github.com/Codium-ai/pr-agent) - AI 驱动的 PR 分析、反馈和建议工具。
- [MutableAI](https://mutable.ai/) - AI 加速的软件开发工具。
- [TurboPilot](https://github.com/ravenscroftj/turbopilot) - 自托管 Copilot 克隆，使用 llama.cpp 库在 4GB 内存运行 Salesforce Codegen 60 亿参数模型。
- [GPT-Code UI](https://github.com/ricklamers/gpt-code-ui) - OpenAI ChatGPT 代码解释器开源实现。 #开源
- [MetaGPT](https://github.com/geekan/MetaGPT) - 多智能体框架：根据一行需求生成 PRD、设计、任务和仓库。
- [Open Interpreter](https://github.com/KillianLucas/open-interpreter) - 在终端中本地运行 OpenAI 代码解释器。
- [Continue](https://www.continue.dev/) - 开源 AI 编程助手，可连接任意模型和上下文，实现 IDE 内自定义自动补全和聊天。 [#开源](https://github.com/continuedev/continue)
- [RooCode][https://github.com/RooCodeInc/Roo-Code] - 集成到 VS Code 的 AI 自主编码Agent。 [#开源](https://github.com/RooCodeInc/Roo-Code)

### 开发者工具

- [co:here](https://cohere.ai/) - 提供高级大型语言模型和 NLP 工具。
- [Haystack](https://haystack.deepset.ai/) - 构建 NLP 应用（如Agent、语义搜索、问答）的框架。
- [LangChain](https://langchain.com/) - 构建语言模型驱动应用的框架。
- [gpt4all](https://github.com/nomic-ai/gpt4all) - 基于大量干净的助理数据（代码、故事、对话）训练的聊天机器人。
- [LLM App](https://github.com/pathwaycom/llm-app) - 开源 Python 库，用于构建实时 LLM 数据管道。
- [LMQL](https://lmql.ai/) - 大型语言模型查询语言。
- [LlamaIndex](https://www.llamaindex.ai/) - 用于在外部数据上构建 LLM 应用的数据框架。
- [Phoenix](https://phoenix.arize.com/) - Arize 提供的开源 ML 可观察性工具，可在 Notebook 环境中监控和微调 LLM、CV 和表格模型。
- [Cursor](https://www.cursor.so/) - 面向协作编程的未来 IDE。
- [SymbolicAI](https://github.com/Xpitfire/symbolicai) - 构建以 LLM 为核心应用的神经符号框架。
- [Vanna.ai](https://vanna.ai/) - 开源 Python RAG 框架，用于 SQL 生成及相关功能。 [#开源](https://github.com/vanna-ai/vanna)
- [Portkey](https://portkey.ai/) - LLMOps 全栈平台，提供 LLM 监控、缓存和管理。
- [agenta](https://github.com/agenta-ai/agenta) - 开源端到端 LLMOps 平台，用于提示工程、评估和部署。 #开源
- [Together AI](https://www.together.ai/) - 快速训练、微调并推理 AI 模型，低成本、可生产规模运行。
- [Gitingest](https://gitingest.com/) - 将任意 Git 仓库生成代码摘要，以便输入 LLM。 [#开源](https://github.com/cyclotruc/gitingest)
- [Repomix](https://repomix.com/) - 将代码库打包为 AI 友好格式。 [#开源](https://github.com/yamadashy/repomix)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Meta LLaMA 模型（及其他模型）纯 C/C++ 推理。 #开源
- [bitnet.cpp](https://github.com/microsoft/BitNet) - Microsoft 官方 1-bit LLM 推理框架。 [#开源](https://github.com/microsoft/BitNet)
- [OpenRouter](https://openrouter.ai/) - LLM 统一接口。 [#开源](https://github.com/OpenRouterTeam)
- [Ludwig](https://github.com/ludwig-ai/ludwig) - 低代码框架，用于构建自定义 AI 模型（LLM 及深度神经网络）。 [#开源](https://github.com/ludwig-ai/ludwig)

### Playground

- [OpenAI Playground](https://platform.openai.com/playground) - 探索资源、教程、API 文档和动态示例。
- [Google AI Studio](https://aistudio.google.com/) - 网页原型工具，可使用 Gemini 和实验模型。
- [GitHub Models](https://github.com/marketplace/models) - 查找并试验 AI 模型，开发生成式 AI 应用。

### 本地 LLM 部署

- [Ollama](https://github.com/ollama/ollama) - 在本地快速启动大型语言模型。
- [Open WebUI](https://github.com/open-webui/open-webui) - 可扩展、功能丰富、用户友好的自托管 AI 平台，可完全离线运行。 #开源
- [Jan](https://jan.ai/) - 在本地离线运行 Mistral 或 Llama2 等 LLM，或连接远程 AI API。 [#开源](https://github.com/janhq/jan)
- [Msty](https://msty.app/) - 简单强大的本地和在线 AI 模型界面。
- [PyGPT](https://pygpt.net/) - 桌面个人 AI 助手，支持聊天、视觉、Agent、图像生成、工具和命令、语音控制等。 #开源
- [LLM](https://llm.datasette.io/) - CLI 工具和 Python 库，用于本地或远程大型语言模型交互。 [#开源](https://github.com/simonw/llm)
- [LM Studio](https://lmstudio.ai) - 在电脑上下载并运行本地 LLM。

## AI 智能体

### 自主 AI 智能体

- [Auto-GPT](https://github.com/Torantulino/Auto-GPT) - 一个实验性开源项目，尝试让 GPT-4 完全自主运行。
- [babyagi](https://github.com/yoheinakajima/babyagi) - AI 驱动的任务管理系统。
- [AgentGPT](https://github.com/reworkd/AgentGPT) - 在浏览器中组装、配置并部署自主 AI Agent。
- [GPT Engineer](https://github.com/AntonOsika/gpt-engineer) - 指定构建需求，AI 提出澄清问题，然后完成构建。
- [GPT Prompt Engineer](https://github.com/mshumer/gpt-prompt-engineer) - 自动化提示工程，生成、测试并排名提示，以找到最佳提示。
- [MetaGPT](https://github.com/geekan/MetaGPT) - 多智能体框架：输入一行需求，返回 PRD、设计、任务和代码仓库。
- [AutoGen](https://github.com/microsoft/autogen) - 框架，可使用多个智能体相互对话来解决任务，开发 LLM 应用。
- [GPT Pilot](https://github.com/Pythagora-io/gpt-pilot) - 开发工具，可从零开始生成可扩展应用，开发者监督实现过程。
- [Devin](https://devin.ai/) - Cognition Labs 开发的自主 AI 软件工程师。
- [OpenDevin](https://github.com/OpenDevin/OpenDevin) - 用于处理软件工程复杂性的自主Agent。 #开源
- [Davika](https://github.com/stitionai/devika) - 具备Agent能力的 AI 软件工程师。 #开源

### 自定义 AI 智能体

- [GPTBots.ai](https://www.gptbots.ai) - 无代码企业智能体平台，用于构建 AI Agent。支持 GPT-5、DeepSeek 等领先模型，提供知识管理、自定义工具、workflow AI 工作流自动化和企业 RBAC 权限控制，并可无缝连接 钉钉、企业微信、WhatsApp、Telegram 等平台。
- [Poe](https://poe.com/) - 提供多种 AI 机器人访问。
- [GPT Builder](https://chat.openai.com/gpts/editor) - 用于创建基于 GPT 的助手。
- [GPTStore](https://gptstore.ai/) - 查找有用的 GPT，分享自己的 GPT。

## 图像

### 模型

- [DALL·E 2](https://openai.com/dall-e-2/) - OpenAI 的 DALL·E 2 可根据自然语言描述生成逼真的图像和艺术作品。
- [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) - Stability AI 的 Stable Diffusion，是先进的文本到图像生成模型。 #开源
- [Midjourney](https://www.midjourney.com/) - 独立研究实验室，探索新的思维媒介并扩展人类想象力。
- [Imagen](https://imagen.research.google/) - Google 的文本到图像扩散模型，具备极高的真实感和深度语言理解能力。
- [Make-A-Scene](https://ai.facebook.com/blog/greater-creative-control-for-ai-image-generation/) - Meta 的多模态生成 AI，让用户通过文字描述和自由草图掌控创作。
- [DragGAN](https://github.com/XingangPan/DragGAN) - 基于生成图像流形的交互式点控操作工具。

### 服务

- [Craiyon](https://www.craiyon.com/) - Craiyon（原 DALL-E mini）可根据任意文本提示绘制图像。
- [DreamStudio](https://beta.dreamstudio.ai/) - Stable Diffusion 图像生成的易用界面。
- [Artbreeder](https://www.artbreeder.com/) - 创意工具，便于协作和探索创意。
- [GauGAN2](http://gaugan.org/gaugan2/) - 使用文字和绘图生成逼真艺术作品，集成分割映射、修补和文本生成。
- [Magic Eraser](https://www.magiceraser.io/) - 秒级移除图像中不需要的元素。
- [Imagine by Magic Studio](https://magicstudio.com/imagine) - 通过描述想法即可创作图像。
- [Alpaca](https://www.getalpaca.io/) - Stable Diffusion Photoshop 插件。
- [Patience.ai](https://www.patience.ai/) - 基于 Stable Diffusion 的图像生成应用。
- [GenShare](https://www.genshare.io/) - 秒级生成艺术作品，自主拥有并分享，多媒体生成工作室。
- [Playground](https://playground.com/) - 免费在线 AI 图像创作工具，可制作艺术、社交媒体内容、演示文稿、海报、视频、Logo 等。
- [Pixelz AI Art Generator](https://pixelz.ai/) - 使用文本生成惊艳艺术作品，提供 Stable Diffusion、CLIP 引导扩散和 PXL·E 算法。
- [modyfi](https://www.modyfi.io/) - 浏览器中的 AI 图像编辑器，支持实时协作。
- [Ponzu](https://www.ponzu.ai/) - 免费 AI Logo 生成器，秒级设计创意 Logo。
- [PhotoRoom](https://www.photoroom.com/) - 手机即可制作产品和人像图片，移除或更换背景。
- [Avatar AI](https://avatarai.me/) - 创建 AI 生成头像。
- [ClipDrop](https://clipdrop.co/) - 无需摄影棚即可生成专业视觉作品，由 [stability.ai](https://stability.ai/) 提供技术支持。
- [Lensa](https://prisma-ai.com/lensa) - 全能图像编辑 App，支持 Stable Diffusion 个性化头像生成。
- [RunDiffusion](https://rundiffusion.com/) - 云端 AI 艺术创作工作空间。
- [Ideogram](https://ideogram.ai/) - 文本到图像平台，让创意表达更便捷。
- [Bing Image Creator](https://www.bing.com/images/create) - 基于 DALL·E 3 的文本生成图像工具，具备安全功能。
- [KREA](https://www.krea.ai/) - 根据风格、概念或产品生成高质量视觉作品。
- [Nightcafe](https://creator.nightcafe.studio/) - 多方法 AI 艺术生成 App。
- [Leonardo AI](https://leonardo.ai/) - 高质量、快速生成专业视觉素材。
- [Recraft](https://www.recraft.ai/) - AI 工具，轻松生成并迭代原创图像、矢量图、插画、图标和 3D 图形。
- [Reve Image](https://reve.art/) - 优化提示遵循度、美学和排版的模型。

### 平面设计

- [Brandmark](https://brandmark.io/) - AI Logo 设计工具。
- [Gamma](https://gamma.app/) - 快速创建漂亮演示和网页，无需排版设计。
- [Microsoft Designer](https://designer.microsoft.com/) - 一键生成惊艳设计。

### 图像库

- [Lexica](https://lexica.art/) - Stable Diffusion 搜索引擎。
- [OpenArt](https://openart.ai/) - 搜索 1000 万+ 提示，生成 AI 艺术作品（Stable Diffusion、DALL·E 2）。
- [PromptHero](https://prompthero.com/) - 搜索 Stable Diffusion、ChatGPT、Midjourney 等模型提示。
- [PromptBase](https://promptbase.com/) - 查找顶级提示工程师的提示，也可出售自己的提示。

### 模型库

- [Civitai](https://civitai.com/) - 社区驱动的 AI 模型共享平台。
- [Stable Diffusion Models](https://rentry.org/sdmodels) - Stable Diffusion 检查点列表。

### Stable Diffusion 资源

- [Stable Horde](https://stablehorde.net/) - 分布式 Stable Diffusion 群集。
- [DiffusionDB](https://diffusiondb.com/) - Stable Diffusion 应用、工具、插件列表。 [Airtable 版本](https://airtable.com/shr0HlBwbw3nZ8Ht3/tblxOCylXV8ynh7ti)
- [PublicPrompts](https://publicprompts.art/) - Stable Diffusion 免费提示合集。
- [Stableboost](https://stableboost.ai/) - Stable Diffusion WebUI，可快速生成大量图像。
- [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) - Hugging Face 扩散模型在线课程 Python 教材。

## 视频

- [Runway](https://runwayml.com/) - AI 工具套件，实时协作，精确编辑，下一代内容创作。
- [Synthesia](https://www.synthesia.io/) - 几分钟内将文本生成视频。
- [Rephrase AI](https://www.rephrase.ai/) - 超个性化大规模视频生成，提高参与度和商业效率。
- [Hour One](https://hourone.ai/) - 自动将文本生成包含虚拟主播的视频。
- [Colossyan](https://www.colossyan.com/) - L&D 视频创作工具，使用 AI 头像生成多语言教育视频。
- [Fliki](https://fliki.ai/) - AI 配音生成文本到视频和语音内容。
- [Pictory](https://pictory.ai/) - 通过文本创建和编辑专业视频。
- [Pika](https://pika.art/) - 创意到视频的平台。
- [Sora](https://openai.com/sora) - 根据文本指令生成逼真和富有想象力的场景。
- [Luma Dream Machine](https://lumalabs.ai/dream-machine) - 快速从文本和图像生成高质量视频。
- [Infinity AI](https://infinity.ai/) - 视频基础模型，自定义角色并赋予生命。
- [KLING AI](https://klingai.com/) - 创意图像和视频生成工具。
- [Hailuo AI](https://hailuoai.video/) - AI 文本到视频生成器。

### 头像

- [D-ID](https://www.d-id.com/) - 一键创建和互动的会说话头像。
- [HeyGen](https://app.heygen.com/) - 几分钟内将脚本生成可定制 AI 头像视频。
- [RenderNet](https://rendernet.ai/) - 图像和视频生成工具，可控制角色设计、构图和风格。

### 动画

- [Wonder Dynamics](https://wonderdynamics.com/) - 将 CG 角色轻松动画化、灯光处理并合成到实景中。

## 音频

### 文本转语音

- [Eleven Labs](https://beta.elevenlabs.io/) - AI 语音生成器。
- [Resemble AI](https://www.resemble.ai/) - AI 语音生成与克隆。
- [WellSaid](https://wellsaidlabs.com/) - 实时文本转语音。
- [Play.ht](https://play.ht/) - AI 语音生成器，在线生成逼真语音。
- [podcast.ai](https://podcast.ai/) - 全部由 AI 生成的播客，使用 Play.ht。
- [VALL-E X](https://vallex-demo.github.io/) - 跨语言神经编解码模型，用于跨语言语音合成。
- [TorToiSe](https://github.com/neonbjb/tortoise-tts) - 多声音文本转语音系统，注重音质。 #开源
- [Bark](https://github.com/suno-ai/bark) - 基于 Transformer 的文本到音频模型。 #开源

### 语音转文本

- [Whisper](https://openai.com/index/whisper/) - 大规模弱监督下的强大语音识别。 [#开源](https://github.com/openai/whisper)
- [Wispr Flow](https://wisprflow.ai/) - 流畅语音输入工具，加快任何应用文本输入。
- [Vibe Transcribe](https://thewh1teagle.github.io/vibe/) - 一体化音视频转录解决方案。 [#开源](https://github.com/thewh1teagle/vibe)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - OpenAI Whisper 模型的 C/C++ 移植。 #开源

### 音乐

- [Harmonai](https://www.harmonai.org/) - 社区驱动组织，发布开源生成音频工具，让音乐制作更易上手、更有趣。
- [Mubert](https://mubert.com/) - 为内容创作者、品牌和开发者提供免版权音乐生态系统。
- [MusicLM](https://google-research.github.io/seanet/musiclm/examples/) - Google Research 开发的模型，可根据文本描述生成高保真音乐。
- [AudioCraft](https://audiocraft.metademolab.com/) - Meta 提供的一站式生成音频工具库，包括 MusicGen（音乐）和 AudioGen（声音）。 #开源
- [Stable Audio](https://stability.ai/stable-audio) - Stability AI 的首款音乐和音效生成产品。
- [AIVA](https://www.aiva.ai/) - AI 音乐生成助手，支持 250+ 风格选择。
- [Suno AI](https://www.suno.ai/) - 任何人都可以创作优秀音乐，无需乐器，只需想象力。
- [Udio](https://www.udio.com/) - 发现、创作并分享音乐。

## 其他

- [Diagram](https://diagram.com/) - 产品设计的新奇方法。
- [PromptBase](https://promptbase.com/) - DALL·E、GPT-3、Midjourney、Stable Diffusion 提示购买和出售市场。
- [This Image Does Not Exist](https://thisimagedoesnotexist.com/) - 测试你能否分辨图像是人工还是计算机生成。
- [Have I Been Trained?](https://haveibeentrained.com/) - 检查你的图像是否被用于训练流行 AI 艺术模型。
- [AI Dungeon](https://aidungeon.io/) - 文本冒险游戏，由你主导，AI 赋予其生命。
- [Clickable](https://www.clickable.so/) - 秒级生成广告，品牌一致且高转化，适用于各营销渠道。
- [Scale Spellbook](https://scale.com/spellbook) - 构建、比较并部署大型语言模型应用。
- [Scenario](https://www.scenario.com/) - AI 生成的游戏资产。
- [Teleprompter](https://github.com/danielgross/teleprompter) - 会议用本地 AI，听你说话并提供有感染力的引用建议。
- [FinChat](https://finchat.io/) - AI 自动生成关于上市公司和投资者的问题答案。
- [Morpher AI](https://morpher.com/ai) - 提供任意市场的实时洞察与分析。
- [Whimsical AI](https://whimsical.com/ai) - GPT 驱动的思维导图、流程图和可视化工具，加速创意开发与流程组织。

## 学习资源

- [Learn Prompting](https://learnprompting.org/) - 免费开源的 AI 交流课程。
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) - 提示工程指南和资源。
- [ChatGPT prompt engineering for developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) - Isa Fulford（OpenAI）和 Andrew Ng（DeepLearning.AI）提供的简短课程。
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) - OpenAI API 使用示例和指南。
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) - 获取大型语言模型更好结果的策略和技巧。
- [PromptPerfect](https://promptperfect.jina.ai/) - 提示工程工具。
- [Anthropic courses](https://github.com/anthropics/courses) - Anthropic 教育课程。
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) - Sebastian Raschka 教你从零构建可用 LLM。

## 更多列表

- [Tools and Resources for AI Art](https://pharmapsychotic.com/tools.html) - Google Colab 生成 AI 工具合集，由 [@pharmapsychotic](https://twitter.com/pharmapsychotic) 整理。
- [The Generative AI Application Landscape](https://twitter.com/sonyatweetybird/status/1584580362339962880) - 生成式 AI 生态图，由 Sequioa Capital 的 [Sonya Huang](https://twitter.com/sonyatweetybird) 制作。
- [Startups - @builtwithgenai](https://airtable.com/shr6nfE9FOHp17IjG/tblL3ekHZfkm3p6YT) - Airtable 创业公司列表，由 [@builtwithgenai](https://twitter.com/builtwithgenai) 制作。
- [The Generative AI Index](https://airtable.com/shrH4REIgddv8SzUo/tbl5dsXdD1P859QLO) - Scale Venture Partners 制作的 Airtable 生成式 AI 列表。
- [Generative AI for Games](https://twitter.com/gwertz/status/1593268767269670912) - a16z 制作的游戏生成式 AI 公司市场图。
- [Generative Deep Art](https://github.com/filipecalegario/awesome-generative-deep-art) - 生成深度学习艺术工具、作品、模型精选，由 [@filipecalegario](https://github.com/filipecalegario/) 提供。
- [GPT-3 Demo](https://gpt3demo.com/) - GPT-3 示例、演示、应用和 NLP 用例展示。
- [GPT-4 Demo](https://gpt4demo.com/) - GPT-4 应用和用例。
- [The Generative AI Landscape](https://github.com/ai-collection/ai-collection) - 生成式 AI 应用精选合集。
- [Molecular design](https://github.com/AspirinCode/papers-for-molecular-design-using-DL) - 使用生成式 AI 和深度学习进行分子设计的资源列表。
- [Open LLMs](https://github.com/eugeneyan/open-llms) - 可商用的开源 LLM 列表。

### ChatGPT 相关列表

- [Awesome ChatGPT](https://github.com/humanloop/awesome-chatgpt) - ChatGPT 和 GPT-3 的工具、演示、文档精选，由 [@jordn](https://github.com/jordn) 整理。
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) - ChatGPT 模型提示示例合集。
- [FlowGPT](https://flowgpt.com/) - 利用最佳提示优化工作流。
- [ChatGPT Prompts for Data Science](https://github.com/travistangvh/ChatGPT-Data-Science-Prompts) - 数据科学提示集合。
- [Awesome ChatGPT](https://github.com/sindresorhus/awesome-chatgpt) - 另一个 ChatGPT 精选列表。






