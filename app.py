"""Demo: 稠密向量相似搜索 —— text_demo 集合"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# ── 配置（必须在 import pymilvus 之前设置，pymilvus 在 import 时读取此环境变量）──
MILVUS_URI      = "https://in01-8e7a04dd78ea0f3.gcp-us-west1.vectordb.zillizcloud.com:443"
MILVUS_USER     = "db_4751ecda463a927"
MILVUS_PASSWORD = "ctd5bap-MCQ3bat5gae"
OPENAI_API_KEY  = "sk-vm8QkbaQ6LghH8iF25C49bEaE1C941De86FeC28aDcDd072c"
OPENAI_BASE_URL = "https://aihubmix.com/v1"

os.environ["MILVUS_URI"] = MILVUS_URI

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pymilvus import MilvusClient
from pydantic import BaseModel

COLLECTION  = "text_demo"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS  = 3072

OUTPUT_FIELDS = [
    "userId", "account", "nickname", "platform",
    "followerCount", "averagePlayCount", "region",
    "aiSummary", "signature", "age", "gender", "race", "language",
]

# ── 全局客户端（启动时初始化）────────────────────────────────────────────────

openai_client: OpenAI | None = None
milvus_client: MilvusClient | None = None

app = FastAPI(title="KOL 向量搜索 Demo")


@app.on_event("startup")
def _startup() -> None:
    global openai_client, milvus_client
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    milvus_client = MilvusClient(
        uri=MILVUS_URI,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
    )


# ── 路由 ──────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    precision_mode: bool = False
    precision_level: float = 0.8   # 0.0 ~ 1.0，对应滑块百分比
    # 字段过滤
    genders: Optional[List[str]] = None
    races: Optional[List[str]] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None


class SearchResult(BaseModel):
    score: float
    userId: str
    account: str
    nickname: str
    platform: str
    followerCount: int
    averagePlayCount: float
    region: str
    aiSummary: str
    signature: str
    age: Any
    gender: str
    race: str
    language: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_before_filter: int          # 过滤前候选数


FETCH_LIMIT = 200  # 每次从 Milvus 最多取回的候选数


def _build_filter(req: "SearchRequest") -> Optional[str]:
    """将请求中的过滤条件转成 Milvus 标量过滤表达式。"""
    clauses: List[str] = []

    def _in_clause(field: str, values: List[str]) -> str:
        quoted = ", ".join(f'"{v}"' for v in values)
        return f'{field} in [{quoted}]'

    if req.genders:
        clauses.append(_in_clause("gender", req.genders))
    if req.races:
        clauses.append(_in_clause("race", req.races))
    # age 为字符串字段，在 Python 侧过滤，此处不加

    return " and ".join(clauses) if clauses else None


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    level = max(0.0, min(1.0, req.precision_level))

    # 1. 嵌入查询文本
    text = req.query.strip().replace("\n", " ")
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIMS,
    )
    vector = resp.data[0].embedding

    # 2. Milvus 搜索（始终取最大候选量）
    if req.precision_mode:
        search_params = {"metric_type": "IP", "params": {"ef": 512}}
    else:
        search_params = {"metric_type": "IP", "params": {"ef": 64}}

    filter_expr = _build_filter(req)

    search_kwargs: Dict[str, Any] = dict(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="dense_vector",
        limit=FETCH_LIMIT,
        search_params=search_params,
        output_fields=OUTPUT_FIELDS,
    )
    if filter_expr:
        search_kwargs["filter"] = filter_expr

    raw: List[List[Dict]] = milvus_client.search(**search_kwargs)

    hits = raw[0]
    total_before_filter = len(hits)

    # 3b. age 为字符串字段，Python 侧过滤
    if req.min_age is not None or req.max_age is not None:
        def _age_ok(hit: Dict) -> bool:
            raw = hit.get("entity", {}).get("age")
            try:
                age = int(raw)
            except (TypeError, ValueError):
                return False
            if req.min_age is not None and age < req.min_age:
                return False
            if req.max_age is not None and age > req.max_age:
                return False
            return True
        hits = [h for h in hits if _age_ok(h)]

    # 4. 整理结果
    results: List[SearchResult] = []
    for hit in hits:
        e = hit.get("entity", {})
        results.append(SearchResult(
            score=round(float(hit.get("distance", 0)), 4),
            userId=e.get("userId") or "",
            account=e.get("account") or "",
            nickname=e.get("nickname") or "",
            platform=e.get("platform") or "",
            followerCount=int(e.get("followerCount") or 0),
            averagePlayCount=float(e.get("averagePlayCount") or 0),
            region=e.get("region") or "",
            aiSummary=e.get("aiSummary") or "",
            signature=e.get("signature") or "",
            age=e.get("age"),
            gender=e.get("gender") or "",
            race=e.get("race") or "",
            language=e.get("language") or "",
        ))

    return SearchResponse(results=results, total_before_filter=total_before_filter)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


# ── 前端 HTML ─────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KOL 向量搜索</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; color: #1a1a2e; min-height: 100vh; }

  .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px 60px; text-align: center; color: #fff; }
  .header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
  .header p  { margin-top: 8px; opacity: 0.85; font-size: 0.95rem; }

  .search-box { max-width: 700px; margin: -28px auto 0; padding: 0 16px; position: relative; }
  .search-row { display: flex; gap: 10px; }
  .search-row input {
    flex: 1; padding: 16px 20px; border: none; border-radius: 12px;
    font-size: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,.15);
    outline: none; transition: box-shadow .2s;
  }
  .search-row input:focus { box-shadow: 0 4px 24px rgba(102,126,234,.4); }
  .search-row button {
    padding: 16px 28px; background: #667eea; color: #fff; border: none;
    border-radius: 12px; font-size: 1rem; font-weight: 600; cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,.15); transition: background .2s, transform .1s;
    white-space: nowrap;
  }
  .search-row button:hover  { background: #5567d5; }
  .search-row button:active { transform: scale(.97); }
  .search-row button:disabled { background: #aaa; cursor: not-allowed; }

  .options { display: flex; align-items: center; gap: 12px; margin-top: 12px;
             padding: 0 4px; font-size: 0.85rem; color: #555; }
  .options label { display: flex; align-items: center; gap: 6px; }
  .options select { padding: 4px 8px; border-radius: 6px; border: 1px solid #ddd;
                    font-size: 0.85rem; }

  /* 精准模式面板 */
  .precision-panel { max-width: 700px; margin: 14px auto 0; padding: 0 16px; }
  .precision-card { background: #fff; border-radius: 14px; padding: 16px 20px;
                    box-shadow: 0 2px 12px rgba(0,0,0,.07); }
  .precision-header { display: flex; align-items: center; justify-content: space-between; }
  .precision-title { font-size: 0.95rem; font-weight: 600; color: #1a1a2e; display: flex; align-items: center; gap: 10px; }
  .precision-intensity { font-size: 0.9rem; font-weight: 400; color: #667eea; }

  /* Toggle 开关 */
  .toggle { position: relative; width: 44px; height: 24px; flex-shrink: 0; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle-track { position: absolute; inset: 0; background: #ddd; border-radius: 24px;
                  cursor: pointer; transition: background .25s; }
  .toggle input:checked + .toggle-track { background: #4caf50; }
  .toggle-thumb { position: absolute; top: 3px; left: 3px; width: 18px; height: 18px;
                  background: #fff; border-radius: 50%; box-shadow: 0 1px 3px rgba(0,0,0,.25);
                  transition: transform .25s; pointer-events: none; }
  .toggle input:checked ~ .toggle-thumb { transform: translateX(20px); }

  /* 滑块区域 */
  .precision-slider-row { margin-top: 14px; display: none; }
  .precision-slider-row.visible { display: block; }
  .slider-wrap { position: relative; height: 24px; display: flex; align-items: center; }
  input[type=range].pslider {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 4px; border-radius: 4px; outline: none; cursor: pointer;
    background: linear-gradient(to right, #4caf50 var(--pct, 80%), #e0e0e0 var(--pct, 80%));
  }
  input[type=range].pslider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 18px; height: 18px; border-radius: 50%;
    background: #4caf50; box-shadow: 0 1px 4px rgba(0,0,0,.25); cursor: pointer;
  }

  /* 说明文字 */
  .precision-desc { margin-top: 14px; font-size: 0.82rem; color: #666; line-height: 1.6;
                    display: none; padding-top: 12px; border-top: 1px solid #f0f0f0; }
  .precision-desc.visible { display: block; }

  .suggestions { max-width: 700px; margin: 16px auto 0; padding: 0 16px; }
  .suggestions-title { font-size: 0.75rem; color: #aaa; margin-bottom: 8px;
                        text-transform: uppercase; letter-spacing: .5px; }
  .chips { display: flex; flex-wrap: wrap; gap: 7px; }
  .chip { padding: 5px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 500;
          cursor: pointer; border: 1.5px solid transparent; transition: all .15s; user-select: none; }
  .chip:hover { filter: brightness(.92); transform: translateY(-1px); }
  .chip-lifestyle  { background: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
  .chip-beauty     { background: #fce4ec; color: #c62828; border-color: #f8bbd0; }
  .chip-social     { background: #fff3e0; color: #e65100; border-color: #ffe0b2; }
  .chip-family     { background: #f3e5f5; color: #6a1b9a; border-color: #e1bee7; }
  .chip-fitness    { background: #e3f2fd; color: #0d47a1; border-color: #bbdefb; }
  .chip-travel     { background: #e0f7fa; color: #006064; border-color: #b2ebf2; }
  .chip-comedy     { background: #fff8e1; color: #f57f17; border-color: #ffecb3; }
  .chip-gaming     { background: #ede7f6; color: #4527a0; border-color: #d1c4e9; }
  .chip-food       { background: #fbe9e7; color: #bf360c; border-color: #ffccbc; }
  .chip-biz        { background: #e8eaf6; color: #283593; border-color: #c5cae9; }
  .chip-health     { background: #e0f2f1; color: #004d40; border-color: #b2dfdb; }
  .chip-crime      { background: #efebe9; color: #3e2723; border-color: #d7ccc8; }

  .status { text-align: center; margin: 32px 0 16px; font-size: 0.9rem; color: #666; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #ddd;
             border-top-color: #667eea; border-radius: 50%; animation: spin .6s linear infinite;
             vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 16px; max-width: 1200px; margin: 0 auto 48px; padding: 0 16px; }

  .card { background: #fff; border-radius: 14px; padding: 20px;
          box-shadow: 0 2px 12px rgba(0,0,0,.07); transition: transform .15s, box-shadow .15s; }
  .card:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(0,0,0,.12); }

  .card-top { display: flex; justify-content: space-between; align-items: flex-start; }
  .card-name { font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
               max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card-account { font-size: 0.8rem; color: #888; margin-top: 2px; }
  .card-account a { color: #667eea; text-decoration: none; }
  .card-account a:hover { text-decoration: underline; }

  .score-badge { background: #f0f4ff; color: #667eea; font-weight: 700;
                 font-size: 0.85rem; padding: 4px 10px; border-radius: 20px;
                 white-space: nowrap; flex-shrink: 0; }

  .tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
  .tag { font-size: 0.75rem; padding: 3px 9px; border-radius: 20px; font-weight: 500; }
  .tag-platform { background: #e8f5e9; color: #2e7d32; }
  .tag-region   { background: #fff3e0; color: #e65100; }
  .tag-gender   { background: #f3e5f5; color: #6a1b9a; }
  .tag-race     { background: #e1f5fe; color: #0277bd; }
  .tag-lang     { background: #fce4ec; color: #c62828; }

  .stats { display: flex; gap: 20px; margin: 12px 0; font-size: 0.82rem; color: #555; }
  .stat-item { display: flex; flex-direction: column; gap: 2px; }
  .stat-label { font-size: 0.7rem; color: #aaa; text-transform: uppercase; letter-spacing: .5px; }
  .stat-value { font-weight: 600; color: #333; }

  .summary { font-size: 0.85rem; color: #444; line-height: 1.55;
             display: -webkit-box; -webkit-line-clamp: 3;
             -webkit-box-orient: vertical; overflow: hidden; }
  .signature { font-size: 0.78rem; color: #999; margin-top: 8px; font-style: italic;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .empty { text-align: center; color: #999; padding: 60px 20px; font-size: 0.95rem; }

  /* 过滤面板 */
  .filter-panel { max-width: 700px; margin: 14px auto 0; padding: 0 16px; }
  .filter-card { background: #fff; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,0,0,.07); overflow: hidden; }
  .filter-header { display: flex; align-items: center; justify-content: space-between;
                   padding: 14px 20px; cursor: pointer; user-select: none; }
  .filter-header-left { display: flex; align-items: center; gap: 8px; font-size: 0.95rem; font-weight: 600; color: #1a1a2e; }
  .filter-badge { background: #667eea; color: #fff; font-size: 0.72rem; font-weight: 700;
                  padding: 2px 8px; border-radius: 20px; display: none; }
  .filter-badge.visible { display: inline-block; }
  .filter-chevron { font-size: 0.8rem; color: #aaa; transition: transform .25s; }
  .filter-chevron.open { transform: rotate(180deg); }
  .filter-body { border-top: 1px solid #f0f0f0; padding: 16px 20px; display: none; }
  .filter-body.open { display: block; }

  .filter-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 14px; }
  .filter-group { flex: 1; min-width: 140px; }
  .filter-group label { display: block; font-size: 0.75rem; color: #aaa;
                        text-transform: uppercase; letter-spacing: .4px; margin-bottom: 6px; }
  .filter-group select, .filter-group input[type=number] {
    width: 100%; padding: 7px 10px; border: 1px solid #e0e0e0; border-radius: 8px;
    font-size: 0.85rem; color: #333; background: #fafafa; outline: none;
    transition: border-color .2s;
  }
  .filter-group select:focus, .filter-group input[type=number]:focus { border-color: #667eea; }
  .filter-group select[multiple] { height: 90px; }

  .filter-actions { display: flex; justify-content: flex-end; }
  .filter-clear { background: none; border: 1px solid #ddd; border-radius: 8px;
                  padding: 6px 16px; font-size: 0.82rem; color: #888; cursor: pointer; transition: all .15s; }
  .filter-clear:hover { border-color: #999; color: #555; }
</style>
</head>
<body>

<div class="header">
  <h1>KOL 向量搜索</h1>
  <p>基于 text-embedding-3-large 稠密向量相似搜索 · text_demo 集合</p>
</div>

<div class="search-box">
  <div class="search-row">
    <input id="q" type="text" placeholder="描述你想找的 KOL，例如：美妆博主，专注护肤教程，面向年轻女性…"
           onkeydown="if(event.key==='Enter') doSearch()" />
    <button id="btn" onclick="doSearch()">搜索</button>
  </div>
</div>

<div class="precision-panel">
  <div class="precision-card">
    <div class="precision-header">
      <div class="precision-title">
        精准模式：<span class="precision-intensity" id="intensityLabel">强度 80%</span>
      </div>
      <label class="toggle">
        <input type="checkbox" id="precisionToggle" onchange="onToggleChange()">
        <div class="toggle-track"></div>
        <div class="toggle-thumb"></div>
      </label>
    </div>
    <div class="precision-slider-row" id="sliderRow">
      <div class="slider-wrap">
        <input type="range" class="pslider" id="precisionSlider"
               min="0" max="100" value="80"
               oninput="onSliderInput(this.value)">
      </div>
    </div>
    <div class="precision-desc" id="precisionDesc">
      精准模式使用更加严格的替代搜索算法，以带来相关性更强但数量更少的搜索结果；强度越高，分数阈值越严格，结果数量有时会低于预期。
    </div>
  </div>
</div>

<div class="filter-panel">
  <div class="filter-card">
    <div class="filter-header" onclick="toggleFilter()">
      <div class="filter-header-left">
        筛选条件
        <span class="filter-badge" id="filterBadge">0</span>
      </div>
      <span class="filter-chevron" id="filterChevron">▼</span>
    </div>
    <div class="filter-body" id="filterBody">
      <div class="filter-row">
        <div class="filter-group">
          <label>性别</label>
          <select id="fGender" multiple onchange="updateFilterBadge()">
            <option value="Woman">女</option>
            <option value="Man">男</option>
          </select>
        </div>
        <div class="filter-group">
          <label>种族</label>
          <select id="fRace" multiple onchange="updateFilterBadge()">
            <option value="white">White</option>
            <option value="black">Black</option>
            <option value="latino">Latino</option>
            <option value="asian">Asian</option>
            <option value="middle eastern">Middle Eastern</option>
            <option value="indian">Indian</option>
          </select>
        </div>
        <div class="filter-group">
          <label>最小年龄</label>
          <input type="number" id="fMinAge" min="0" max="100"
                 placeholder="不限" oninput="updateFilterBadge()">
        </div>
        <div class="filter-group">
          <label>最大年龄</label>
          <input type="number" id="fMaxAge" min="0" max="100"
                 placeholder="不限" oninput="updateFilterBadge()">
        </div>
      </div>
      <div class="filter-actions">
        <button class="filter-clear" onclick="clearFilters()">清除筛选</button>
      </div>
    </div>
  </div>
</div>

<div class="suggestions">
  <div class="suggestions-title">热门方向</div>
  <div class="chips">
    <span class="chip chip-lifestyle"  onclick="fillSearch('lifestyle vlogging and personal daily life content')">日常生活</span>
    <span class="chip chip-beauty"     onclick="fillSearch('beauty skincare and makeup tutorials targeting young women')">美妆护肤</span>
    <span class="chip chip-beauty"     onclick="fillSearch('fashion styling and personal aesthetic content')">时尚穿搭</span>
    <span class="chip chip-social"     onclick="fillSearch('pop culture celebrity news and entertainment commentary')">娱乐八卦</span>
    <span class="chip chip-social"     onclick="fillSearch('social commentary and current events discussion')">社会评论</span>
    <span class="chip chip-family"     onclick="fillSearch('motherhood parenting and family life vlogging')">母婴育儿</span>
    <span class="chip chip-fitness"    onclick="fillSearch('fitness gym workout and bodybuilding content')">健身运动</span>
    <span class="chip chip-fitness"    onclick="fillSearch('basketball football and sports entertainment')">体育竞技</span>
    <span class="chip chip-travel"     onclick="fillSearch('travel experiences and destination exploration')">旅游探索</span>
    <span class="chip chip-comedy"     onclick="fillSearch('comedy skits observational humor and stand-up')">搞笑喜剧</span>
    <span class="chip chip-gaming"     onclick="fillSearch('gaming anime and pop culture for young audiences')">游戏动漫</span>
    <span class="chip chip-food"       onclick="fillSearch('home cooking food reviews and recipe tutorials')">美食烹饪</span>
    <span class="chip chip-biz"        onclick="fillSearch('international trade wholesale sourcing and B2B business')">跨境电商</span>
    <span class="chip chip-health"     onclick="fillSearch('healthcare nursing and medical education content')">医疗健康</span>
    <span class="chip chip-crime"      onclick="fillSearch('true crime mystery and criminal investigation storytelling')">真实犯罪</span>
    <span class="chip chip-social"     onclick="fillSearch('political commentary and social issues discussion')">时政评论</span>
    <span class="chip chip-lifestyle"  onclick="fillSearch('personal development self improvement and motivation')">个人成长</span>
    <span class="chip chip-lifestyle"  onclick="fillSearch('pregnancy fertility and reproductive health journey')">孕产记录</span>
  </div>
</div>

<div id="status" class="status" style="display:none"></div>
<div id="grid" class="grid"></div>

<script>
function fmt(n) {
  if (n == null) return '-';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return String(n);
}

function tag(text, cls) {
  if (!text) return '';
  return `<span class="tag ${cls}">${text}</span>`;
}

function renderCard(r, rank) {
  const score = (r.score * 100).toFixed(1);
  return `
  <div class="card">
    <div class="card-top">
      <div>
        <div class="card-name" title="${r.nickname}">${r.nickname || r.account || '—'}</div>
        <div class="card-account">
          ${r.platform === 'TIKTOK' && r.account
            ? `<a href="https://www.tiktok.com/@${r.account}" target="_blank" rel="noopener">@${r.account}</a>`
            : `@${r.account || r.userId}`}
        </div>
      </div>
      <div class="score-badge">相似度 ${score}%</div>
    </div>
    <div class="tags">
      ${tag(r.platform,  'tag-platform')}
      ${tag(r.region,    'tag-region')}
      ${r.gender ? tag(r.gender, 'tag-gender') : ''}
      ${r.race   ? tag(r.race,   'tag-race')   : ''}
      ${r.language ? tag(r.language, 'tag-lang') : ''}
    </div>
    <div class="stats">
      <div class="stat-item">
        <span class="stat-label">粉丝</span>
        <span class="stat-value">${fmt(r.followerCount)}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">均播</span>
        <span class="stat-value">${fmt(Math.round(r.averagePlayCount))}</span>
      </div>
      ${r.age != null ? `<div class="stat-item"><span class="stat-label">年龄</span><span class="stat-value">${r.age}</span></div>` : ''}
    </div>
    ${r.aiSummary ? `<div class="summary">${r.aiSummary}</div>` : ''}
    ${r.signature ? `<div class="signature">"${r.signature}"</div>` : ''}
  </div>`;
}

function toggleFilter() {
  document.getElementById('filterBody').classList.toggle('open');
  document.getElementById('filterChevron').classList.toggle('open');
}

function getMultiSelect(id) {
  const sel = document.getElementById(id);
  const vals = Array.from(sel.selectedOptions).map(o => o.value);
  return vals.length ? vals : null;
}

function updateFilterBadge() {
  let count = 0;
  if (getMultiSelect('fGender')) count++;
  if (getMultiSelect('fRace')) count++;
  if (document.getElementById('fMinAge').value) count++;
  if (document.getElementById('fMaxAge').value) count++;
  const badge = document.getElementById('filterBadge');
  badge.textContent = count;
  badge.classList.toggle('visible', count > 0);
}

function clearFilters() {
  ['fGender','fRace'].forEach(id => {
    Array.from(document.getElementById(id).options).forEach(o => o.selected = false);
  });
  ['fMinAge','fMaxAge'].forEach(id => {
    document.getElementById(id).value = '';
  });
  updateFilterBadge();
}

function getFilters() {
  const f = {};
  const genders = getMultiSelect('fGender');
  if (genders) f.genders = genders;
  const races = getMultiSelect('fRace');
  if (races) f.races = races;
  const minAge = parseInt(document.getElementById('fMinAge').value);
  if (!isNaN(minAge)) f.min_age = minAge;
  const maxAge = parseInt(document.getElementById('fMaxAge').value);
  if (!isNaN(maxAge)) f.max_age = maxAge;
  return f;
}

function onToggleChange() {
  const on = document.getElementById('precisionToggle').checked;
  document.getElementById('sliderRow').classList.toggle('visible', on);
  document.getElementById('precisionDesc').classList.toggle('visible', on);
}

function onSliderInput(val) {
  document.getElementById('intensityLabel').textContent = `强度 ${val}%`;
  document.getElementById('precisionSlider').style.setProperty('--pct', val + '%');
}

function fillSearch(text) {
  document.getElementById('q').value = text;
  doSearch();
}

async function doSearch() {
  const q = document.getElementById('q').value.trim();
  if (!q) { document.getElementById('q').focus(); return; }
  const precisionMode = document.getElementById('precisionToggle').checked;
  const precisionLevel = parseInt(document.getElementById('precisionSlider').value) / 100;
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  const grid = document.getElementById('grid');

  btn.disabled = true;
  btn.textContent = '搜索中…';
  grid.innerHTML = '';
  status.style.display = 'block';
  status.innerHTML = '<span class="spinner"></span>正在嵌入查询并搜索…';

  const t0 = Date.now();
  try {
    const res = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, precision_mode: precisionMode, precision_level: precisionLevel, ...getFilters() }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
    const { results, total_before_filter } = data;
    let statusText = `找到 <strong>${results.length}</strong> 条结果 · 耗时 ${elapsed}s`;
    if (precisionMode) {
      statusText += ` · 精准模式（从 ${total_before_filter} 条候选中筛选）`;
    }
    status.innerHTML = statusText;
    if (results.length === 0) {
      grid.innerHTML = '<div class="empty">暂无匹配结果，可降低精准模式强度后重试</div>';
    } else {
      grid.innerHTML = results.map((r, i) => renderCard(r, i + 1)).join('');
    }
  } catch (e) {
    status.innerHTML = `<span style="color:#e53935">错误：${e.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '搜索';
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
