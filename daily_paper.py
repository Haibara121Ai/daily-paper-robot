import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import requests

JST = ZoneInfo("Asia/Tokyo")

BASE_URL = "https://export.arxiv.org/api/query"

WORK_DIR = Path(__file__).resolve().parent
STATE_FILE = WORK_DIR / "seen_papers.json"
OUTPUT_DIR = WORK_DIR / "papers"
OUTPUT_DIR.mkdir(exist_ok=True)


KEYWORDS = {
    # 控制
    "control": 5,
    "optimal control": 8,
    "mpc": 8,
    "adaptive control": 7,
    "robust control": 7,
    "nonlinear control": 7,
    "trajectory tracking": 8,
    "motion planning": 7,
    "path planning": 7,

    # 具身智能 / 机器人
    "embodied intelligence": 10,
    "embodied ai": 10,
    "robot": 10,
    "robotics": 6,
    "humanoid": 8,
    "legged robot": 9,
    "mobile robot": 8,
    "manipulation": 6,
    "vision-language-action": 10,
    "vla": 8,
    "diffusion policy": 9,

    # 自动驾驶
    "autonomous driving": 10,
    "self-driving": 9,
    "autonomous vehicle": 9,
    "vehicle control": 9,
    "trajectory prediction": 7,
    "decision making": 5,

    # 无人机
    "uav": 9,
    "drone": 8,
    "quadrotor": 10,
    "aerial robot": 9,
    "flight control": 9,

    # 大语言模型
    "large language model": 8,
    "llm": 7,
    "multimodal": 5,
    "foundation model": 6,
    "vision-language model": 7,
}


def load_seen_ids() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return set()


def save_seen_ids(seen_ids: set[str]) -> None:
    STATE_FILE.write_text(
        json.dumps(sorted(seen_ids), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_query() -> str:
    categories = [
        "cat:eess.SY",  # Systems and Control
        "cat:cs.SY",
        "cat:cs.RO",   # Robotics
        "cat:cs.AI",   # Artificial Intelligence
        "cat:cs.LG",   # Machine Learning
        "cat:cs.CV",   # Computer Vision
        "cat:cs.CL",   # Computation and Language
    ]

    keyword_terms = [
        'all:"control"',
        'all:"robot"',
        'all:"robotics"',
        'all:"embodied"',
        'all:"autonomous driving"',
        'all:"uav"',
        'all:"drone"',
        'all:"quadrotor"',
        'all:"large language model"',
        'all:"LLM"',
        'all:"vision-language-action"',
        'all:"diffusion policy"',
    ]

    category_query = "(" + " OR ".join(categories) + ")"
    keyword_query = "(" + " OR ".join(keyword_terms) + ")"

    return category_query



def fetch_arxiv_papers(max_results: int = 50) -> list[dict]:
    query = build_query()

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    response = requests.get(url, timeout=20)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except requests.RequestException as exc:
        raise RuntimeError(f"请求 arXiv API 失败：{exc}") from exc
    except ET.ParseError as exc:
        raise RuntimeError(f"arXiv 返回内容不是有效 XML：{exc}") from exc


    root = ET.fromstring(response.text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []

    for entry in root.findall("atom:entry", ns):
        paper_id_url = entry.findtext("atom:id", default="", namespaces=ns)
        paper_id = paper_id_url.rstrip("/").split("/")[-1]

        title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)

        authors = [
            clean_text(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]

        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall("atom:category", ns)
        ]

        pdf_url = ""
        abs_url = paper_id_url

        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "categories": categories,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
            }
        )

    return papers


def score_paper(paper: dict) -> int:
    text = f"{paper['title']} {paper['summary']}".lower()
    score = 0

    for keyword, weight in KEYWORDS.items():
        if keyword.lower() in text:
            score += weight

    # 控制、机器人相关类别额外加权
    category_bonus = {
        "eess.SY": 5,
        "cs.RO": 8,
        "cs.AI": 10,
        "cs.LG": 5,
        "cs.CV": 4,
        "cs.CL": 4,
    }

    for cat in paper["categories"]:
        score += category_bonus.get(cat, 0)

    return score


def simple_chinese_intro(paper: dict) -> str:
    title = paper["title"]
    summary = paper["summary"]

    intro = f"""这篇论文主要研究与控制、机器人或智能系统相关的问题。根据标题和摘要判断，它的核心内容是：围绕“{title}”所描述的任务，提出一种算法、模型或系统方法，并通过实验或理论分析验证其效果。

你阅读时可以重点关注三点：

1. 它解决的具体问题是什么？
2. 它的方法相比传统控制、规划、学习或大模型方法有什么改进？
3. 它的实验场景是否接近真实机器人、自动驾驶、无人机或具身智能系统？

英文摘要如下，建议先读摘要，再看方法部分和实验部分：

{summary}
"""
    return intro


def select_paper(papers: list[dict], seen_ids: set[str]) -> dict | None:
    candidates = [p for p in papers if p["id"] not in seen_ids]
    if not candidates:
        return None

    candidates.sort(key=score_paper, reverse=True)
    return candidates[0]


def write_markdown(paper: dict) -> Path:
    today = datetime.now(JST).strftime("%Y-%m-%d")
    safe_id = paper["id"].replace("/", "_")
    output_file = OUTPUT_DIR / f"{today}_{safe_id}.md"

    authors = ", ".join(paper["authors"][:6])
    if len(paper["authors"]) > 6:
        authors += " et al."

    categories = ", ".join(paper["categories"])

    content = f"""# Daily Paper - {today}

## 论文标题

{paper["title"]}

## 作者

{authors}

## 发布时间

{paper["published"]}

## arXiv 分类

{categories}

## 链接

- Abstract: {paper["abs_url"]}
- PDF: {paper["pdf_url"]}

## 推荐理由

这篇论文与控制、机器人、具身智能、自动驾驶、无人机或大语言模型相关，且在最近提交的论文中关键词匹配度较高。

## 简单介绍

{simple_chinese_intro(paper)}

## 阅读建议

第一遍阅读建议按这个顺序：

1. 先读 Abstract，弄清楚论文解决什么问题。
2. 再看 Introduction，理解研究背景和作者声称的贡献。
3. 跳到 Method 或 Approach，抓住核心算法结构。
4. 看 Experiments，判断它是在仿真、真实机器人、数据集，还是理论环境中验证。
5. 最后看 Limitation 或 Conclusion，判断这个方法的边界。

## 你可以记录的问题

- 这篇论文的问题定义是什么？
- 输入和输出分别是什么？
- 是否涉及控制器、规划器、动力学模型、强化学习或大模型？
- 方法能不能迁移到移动机器人、AGV、无人机或自动驾驶场景？
"""

    output_file.write_text(content, encoding="utf-8")
    return output_file


def main():
    print(f"[{datetime.now(JST)}] daily_paper.py started")
    
    seen_ids = load_seen_ids()
    papers = fetch_arxiv_papers(max_results=50)

    paper = select_paper(papers, seen_ids)

    if paper is None:
        print("今天没有找到新的候选论文。")
        return

    output_file = write_markdown(paper)
    seen_ids.add(paper["id"])
    save_seen_ids(seen_ids)

    print("今日推荐论文：")
    print(paper["title"])
    print(paper["abs_url"])
    print(f"已保存到：{output_file}")


if __name__ == "__main__":
    main()