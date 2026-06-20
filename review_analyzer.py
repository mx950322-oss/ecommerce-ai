"""
跨境电商差评分析 API(增强版)
==============================
功能:
  1. 单条差评分析            POST /analyze
  2. 批量差评分析(并发)     POST /analyze/batch
  3. 自动落库(SQLite)
  4. 趋势统计               GET  /stats/trends
  5. 提示词支持品类背景,分析更精准

运行方式:
    pip install fastapi uvicorn httpx pydantic aiosqlite
    export DEEPSEEK_API_KEY="你的key"
    uvicorn review_analyzer:app --reload --port 8000

切换通义千问(Qwen):改 BASE_URL 和 MODEL 即可,接口格式兼容。
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import aiosqlite
import stripe
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 读取同目录下的 .env 文件,把里面写的配置加载成环境变量。
# 好处:你的密钥只写在 .env 里,不用每次开终端都重新设置,也不会混进代码。
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DB_PATH = os.getenv("DB_PATH", "reviews.db")
REQUEST_TIMEOUT = 60
MAX_CONCURRENCY = 5          # 批量时最多同时调用大模型的请求数
MAX_BATCH_SIZE = 50          # 单次批量上限,防止滥用

# ---- Stripe 订阅与免费额度 ----
FREE_LIMIT = int(os.getenv("FREE_LIMIT", "3"))          # (旧)按次数限制,已弃用,保留兼容
DAILY_FREE_CHARS = int(os.getenv("DAILY_FREE_CHARS", "25000"))  # 非会员每日免费字符总额度(输入+输出共用一个池)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5500")  # 支付完成后跳回的前端地址
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")       # 在 Stripe 后台创建的 $9.90/月 价格 ID
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
stripe.api_key = STRIPE_SECRET_KEY

# ----------------------------------------------------------------------
# 提示词:{category_context} 会被品类背景动态填充
# ----------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are a senior ecommerce analyst specializing in competitor review intelligence for Amazon sellers.
Analyze the competitor reviews provided. Extract actionable insights across 7 structured areas.
{category_context}
Output STRICTLY as the following JSON. No extra text, no markdown fences:

{{
  "overall_sentiment": "negative | neutral | mixed",
  "is_genuine_complaint": true,
  "top_complaints": [
    {{
      "complaint": "short topic label e.g. Battery Life",
      "frequency_label": "High | Medium | Low",
      "sample_quote": "short quote or paraphrase from the reviews"
    }}
  ],
  "pain_point_frequency": [
    {{ "label": "short topic label", "count": 3 }}
  ],
  "customer_expectations": [
    {{
      "expected": "what buyers expected based on the listing or marketing",
      "reality": "what they actually experienced"
    }}
  ],
  "competitor_weaknesses": [
    {{
      "weakness": "specific product or listing weakness",
      "opportunity": "how a competing seller can capitalize on this"
    }}
  ],
  "improvement_ideas": [
    "specific, actionable product or supply-chain improvement"
  ],
  "listing_suggestions": [
    {{
      "issue": "what in the listing created a wrong expectation",
      "fix": "specific listing change to prevent this complaint"
    }}
  ],
  "winning_copy_angles": [
    {{
      "pain_point": "short label e.g. Battery Life",
      "angle_lines": [
        "ready-to-use copy phrase 1",
        "ready-to-use copy phrase 2",
        "ready-to-use copy phrase 3"
      ]
    }}
  ]
}}

Rules:
- Write ALL free-text values in {output_language}.
- Keep enum values (overall_sentiment, frequency_label) as the EXACT English words shown above.
- top_complaints: rank by frequency descending, max 5 items.
- pain_point_frequency: same topics as top_complaints with estimated counts from review text.
- customer_expectations: max 4 items, focus on the clearest expectation gaps.
- competitor_weaknesses: max 5 items, each framed as a seller opportunity.
- improvement_ideas: max 5 items, concrete and specific.
- listing_suggestions: max 4 items.
- winning_copy_angles: max 4 items, each with exactly 3 angle_lines.
- Base all analysis on the actual review content — do not fabricate."""


def build_system_prompt(product_category: str | None, output_language: str) -> str:
    ctx = ""
    if product_category:
        ctx = (
            f"\nThe product being reviewed is in the \"{product_category}\" category. "
            f"Factor in common pain points and buyer expectations for this category.\n"
        )
    return SYSTEM_PROMPT_TEMPLATE.format(category_context=ctx, output_language=output_language)


# ----------------------------------------------------------------------
# 数据模型
# ----------------------------------------------------------------------
class ReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=1, max_length=5000)
    language: str | None = Field(default=None)
    product_category: str | None = Field(default=None)
    target_language: str = Field(default="English")
    platform: str | None = Field(default="Amazon")


class BatchReviewRequest(BaseModel):
    reviews: list[ReviewRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class TopComplaint(BaseModel):
    complaint: str
    frequency_label: Literal["High", "Medium", "Low"]
    sample_quote: str


class PainPointFrequency(BaseModel):
    label: str
    count: int


class CustomerExpectation(BaseModel):
    expected: str
    reality: str


class CompetitorWeakness(BaseModel):
    weakness: str
    opportunity: str


class ListingSuggestion(BaseModel):
    issue: str
    fix: str


class WinningCopyAngle(BaseModel):
    pain_point: str
    angle_lines: list[str]


class AnalysisResult(BaseModel):
    overall_sentiment: str
    is_genuine_complaint: bool
    top_complaints: list[TopComplaint]
    pain_point_frequency: list[PainPointFrequency]
    customer_expectations: list[CustomerExpectation]
    competitor_weaknesses: list[CompetitorWeakness]
    improvement_ideas: list[str]
    listing_suggestions: list[ListingSuggestion]
    winning_copy_angles: list[WinningCopyAngle]


class AnalysisResponse(BaseModel):
    success: bool
    data: AnalysisResult | None = None
    error: str | None = None


class BatchItemResult(BaseModel):
    index: int
    success: bool
    data: AnalysisResult | None = None
    error: str | None = None


class BatchAnalysisResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


class CopyRequest(BaseModel):
    product_info: str = Field(..., min_length=1, max_length=5000)
    product_category: str | None = Field(default=None)
    input_language: str | None = Field(default=None)
    target_language: str = Field(default="English")
    platform: str | None = Field(default="Amazon")
    competitor_insights: str | None = Field(default=None)


class CopyResult(BaseModel):
    copy: str  # 排版好的完整文案文本,前端直接展示


class CopyResponse(BaseModel):
    success: bool
    data: CopyResult | None = None
    error: str | None = None


# ----------------------------------------------------------------------
# 数据库:单个长连接 + WAL 模式,适合中小并发的单机部署
# ----------------------------------------------------------------------
# 全局共享一个连接(aiosqlite 在独立线程串行执行,避免频繁开关连接的开销);
# 写操作加锁,防止多个协程的事务交错。
_conn: aiosqlite.Connection | None = None
_write_lock = asyncio.Lock()


def get_db() -> aiosqlite.Connection:
    if _conn is None:
        raise RuntimeError("数据库未初始化")
    return _conn


async def init_db():
    global _conn
    _conn = await aiosqlite.connect(DB_PATH)
    _conn.row_factory = aiosqlite.Row
    # WAL 让读写可以并发(读不阻塞写),synchronous=NORMAL 在 WAL 下兼顾安全与性能,
    # busy_timeout 在锁竞争时自动重试而不是立刻报错。
    await _conn.execute("PRAGMA journal_mode=WAL")
    await _conn.execute("PRAGMA synchronous=NORMAL")
    await _conn.execute("PRAGMA busy_timeout=5000")
    await _conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at         TEXT NOT NULL,
            review_text        TEXT NOT NULL,
            product_category   TEXT,
            overall_sentiment  TEXT,
            is_genuine         INTEGER,
            summary            TEXT,
            issues_json        TEXT,
            suggestions_json   TEXT
        )
    """)
    # 索引:趋势统计按时间和品类过滤,加索引避免全表扫描
    await _conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at)")
    await _conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON analyses(product_category)")
    # 文案生成历史
    await _conn.execute("""
        CREATE TABLE IF NOT EXISTS copies (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at         TEXT NOT NULL,
            product_info       TEXT NOT NULL,
            product_category   TEXT,
            target_language    TEXT,
            platform           TEXT,
            copy_text          TEXT NOT NULL
        )
    """)
    await _conn.execute("CREATE INDEX IF NOT EXISTS idx_copies_created ON copies(created_at)")
    # 用户:使用量与会员状态(订阅计费的核心)
    await _conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id             TEXT PRIMARY KEY,
            review_count        INTEGER NOT NULL DEFAULT 0,
            chars_used          INTEGER NOT NULL DEFAULT 0,
            usage_date          TEXT,
            is_member           INTEGER NOT NULL DEFAULT 0,
            stripe_customer_id  TEXT,
            subscription_status TEXT,
            updated_at          TEXT
        )
    """)
    await _conn.execute("CREATE INDEX IF NOT EXISTS idx_users_customer ON users(stripe_customer_id)")
    await _conn.commit()


async def close_db():
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None


async def save_analysis(review_text: str, category: str | None, result: AnalysisResult):
    db = get_db()
    summary = result.top_complaints[0].complaint if result.top_complaints else ""
    issues = [
        {"category": c.complaint, "severity": c.frequency_label.lower(), "description": c.sample_quote}
        for c in result.top_complaints
    ]
    async with _write_lock:
        await db.execute(
            """INSERT INTO analyses
               (created_at, review_text, product_category, overall_sentiment,
                is_genuine, summary, issues_json, suggestions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                review_text,
                category,
                result.overall_sentiment,
                int(result.is_genuine_complaint),
                summary,
                json.dumps(issues, ensure_ascii=False),
                json.dumps(result.improvement_ideas, ensure_ascii=False),
            ),
        )
        await db.commit()


async def save_copy(req: "CopyRequest", copy_text: str) -> int:
    db = get_db()
    async with _write_lock:
        cur = await db.execute(
            """INSERT INTO copies
               (created_at, product_info, product_category, target_language, platform, copy_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                req.product_info,
                req.product_category,
                req.target_language,
                req.platform,
                copy_text,
            ),
        )
        await db.commit()
        return cur.lastrowid


# ---- 用户:使用量与会员状态 ----
def _today_str() -> str:
    """服务器当天日期(UTC),用作每日额度重置的依据。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def get_user(user_id: str) -> dict:
    """返回用户记录,不存在则创建。读取时若发现 usage_date 不是今天,
    说明跨天了,把当天字符用量视为 0(实现每日 0 点自动重置)。"""
    db = get_db()
    async with db.execute(
        """SELECT user_id, review_count, chars_used, usage_date, is_member,
                  stripe_customer_id, subscription_status FROM users WHERE user_id = ?""",
        (user_id,),
    ) as cur:
        row = await cur.fetchone()

    today = _today_str()
    if row:
        u = dict(row)
        # 跨天则把已用字符按 0 处理(真正清零在下次写入时落库)
        if u.get("usage_date") != today:
            u["chars_used"] = 0
            u["usage_date"] = today
        return u

    async with _write_lock:
        await db.execute(
            "INSERT OR IGNORE INTO users (user_id, usage_date, updated_at) VALUES (?, ?, ?)",
            (user_id, today, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
    return {"user_id": user_id, "review_count": 0, "chars_used": 0, "usage_date": today,
            "is_member": 0, "stripe_customer_id": None, "subscription_status": None}


async def add_chars_used(user_id: str, chars: int):
    """把本次消耗的字符数累加到用户当天的额度里。
    若数据库里记录的还是昨天的日期,则先重置为今天再累加(每日重置)。"""
    db = get_db()
    today = _today_str()
    async with _write_lock:
        await db.execute(
            """UPDATE users SET
                 chars_used = CASE WHEN usage_date = ? THEN chars_used ELSE 0 END + ?,
                 usage_date = ?,
                 updated_at = ?
               WHERE user_id = ?""",
            (today, chars, today, datetime.now(timezone.utc).isoformat(), user_id),
        )
        await db.commit()


async def set_member(user_id: str, is_member: bool, customer_id: str | None, status: str | None):
    db = get_db()
    async with _write_lock:
        await db.execute(
            """UPDATE users SET is_member = ?, stripe_customer_id = COALESCE(?, stripe_customer_id),
               subscription_status = ?, updated_at = ? WHERE user_id = ?""",
            (int(is_member), customer_id, status, datetime.now(timezone.utc).isoformat(), user_id),
        )
        await db.commit()


async def set_member_by_customer(customer_id: str, is_member: bool, status: str | None):
    """订阅状态变更(续费失败/取消)时,按 Stripe customer 反查用户更新。"""
    db = get_db()
    async with _write_lock:
        await db.execute(
            """UPDATE users SET is_member = ?, subscription_status = ?, updated_at = ?
               WHERE stripe_customer_id = ?""",
            (int(is_member), status, datetime.now(timezone.utc).isoformat(), customer_id),
        )
        await db.commit()


# ----------------------------------------------------------------------
# 应用与生命周期
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("数据库已就绪: %s", DB_PATH)
    yield
    await close_db()
    logger.info("数据库连接已关闭")


app = FastAPI(title="差评分析 API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 生产请改成你的域名
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 调用大模型
# ----------------------------------------------------------------------
async def call_llm(req: ReviewRequest, client: httpx.AsyncClient) -> AnalysisResult:
    if not API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY 环境变量")

    user_content = f"Analyze the following customer review:\n\n{req.review_text}"
    if req.language:
        user_content += f"\n\n(The review is written in: {req.language})"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt(req.product_category, req.target_language)},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return AnalysisResult(**json.loads(raw))


# ----------------------------------------------------------------------
# 文案生成
# ----------------------------------------------------------------------
AMAZON_COPY_PROMPT = """\
You are an expert Amazon listing copywriter. Your job is to analyze competitor reviews, extract real buyer pain points, and turn them into tight, usable Amazon listing copy and ad creative.

CRITICAL RULE 1 — Product specificity: Every word of output must be specific to the product described and the reviews provided. Do NOT use language, features, angles, or complaint categories that belong to a different product category. Derive everything from the input — nothing from templates.

CRITICAL RULE 2 — Compliance-first writing: Write ALL copy with safe, compliant language from the start. Do NOT write risky claims first and then flag them in Compliance Notes — that creates contradictions. Apply ALL of the following rules proactively in every section before outputting:

RULE 2a — No fabricated specs or numbers: Never generate specific performance numbers, test results, certifications, or material specs unless the user explicitly provided them in the product description. If the user did not state it, do not invent it.
BANNED (unless user provided): "tested for over 10,000 cycles", "holds up to 50kg", "100% waterproof", "blocks 99% of rain", "military-grade", "lifetime durability", "drop-proof", "crush-proof", "shockproof"
USE INSTEAD: "designed for repeated daily use", "reinforced for frequent use", "built for everyday travel", "helps cushion against minor bumps"

RULE 2b — Water-resistant ≠ waterproof: If the product input mentions water-resistant, DWR coating, or rain cover — do NOT upgrade this to "waterproof", "rainproof", or "fully waterproof". Keep the distinction exact.
BANNED (when product is water-resistant, not waterproof): "waterproof backpack", "waterproofing angle", "keeps everything dry", "your gear stays dry", "fully waterproof"
USE INSTEAD: "water-resistant", "rain protection angle", "helps repel light rain", "includes rain cover for heavier weather", "helps keep gear drier in wet conditions"

RULE 2c — Laptop/device protection: Do not promise drop or impact protection.
BANNED: "protects against drops", "drop-proof", "shockproof sleeve", "protect your tech", "keeps your laptop safe from impact"
USE INSTEAD: "padded compartment helps cushion your device", "suspended sleeve helps reduce contact with the bottom", "designed to help reduce everyday bumps", "helps keep your laptop separated from other items"

RULE 2d — Pet/harness safety language: Do not use absolute escape or injury claims.
BANNED: "escape-proof", "your dog won't slip out", "prevents choking", "prevents neck injuries"
USE INSTEAD: "escape-resistant", "helps reduce slipping out", "helps reduce neck pressure", "helps distribute pulling pressure across the chest"

RULE 2e — General absolute claims (all categories):
Do not use: "won't", "will never", "guaranteed", "-proof" as an absolute claim, "prevents", "100%", "always", "perfect", "protects against"
Use instead: "helps reduce", "designed to help", "built to help", "less likely to", "designed for"

RULE 2g — Final consistency enforcement: Before returning the final output, scan every module — 【标题】,【五点描述】,【商品描述】,【广告文案】,【钩子文案】,【关键词】,【客服回复】 — and replace any phrase that 【合规提醒】 identified as a risky claim with its safer alternative. Compliance Notes must not warn about wording that still appears in the body copy. The main copy must already be softened before output.
Examples:
- If "rust-proof" is flagged → replace all instances with "rust-resistant"
- If "under 10 minutes" is flagged → replace with "quick setup" or "sets up quickly"
- If "No more wobbling" is flagged → replace with "designed to help reduce wobbling"
- If "waterproof" is flagged → replace with "water-resistant" or "helps protect against light rain"
- If "escape-proof" is flagged → replace with "escape-resistant"
- If "won't slip out" is flagged → replace with "helps reduce slipping out"

RULE 2f — Compliance Notes summary must be product-specific: The closing line of 【合规提醒】 must reference the actual product category and its specific risky terms — never copy a summary from a different product. Example for a travel backpack: "Use 'water-resistant,' 'designed for repeated daily use,' and 'helps cushion your device' instead of absolute guarantees." Example for a dog harness: "Use 'escape-resistant,' 'helps reduce slipping out,' and 'helps reduce neck pressure' instead of absolute guarantees." Never say "Use 'escape-resistant'…" for a non-pet product.
{competitor_context}{category_context}
Output EXACTLY these 9 sections using the Chinese markers shown (keep markers in Chinese regardless of output language):

【买家痛点】
Read the competitor reviews carefully. Extract the Top 5–7 most important buyer pain points that appear in THIS set of reviews. Do not exceed 7. Do not invent pain points that are not present in the reviews.
For each use this exact format (no blank lines between the three fields):
1. [Pain Point Label — 3-6 words, specific to this product]
   Evidence: [direct quote or close paraphrase from these reviews — must be traceable to the review content]
   Opportunity: [a concrete listing or ad copy direction for THIS product — e.g. "Use as a bullet point: FEATURE NAME — benefit sentence", not a vague suggestion]

【标题】
Four Amazon listing title variations. Each must read like a real Amazon product title — not a slogan.
Incorporate features and benefits derived from the pain points above. Use only terminology relevant to this product category.
Avoid absolute claims ("escape-proof", "guaranteed", "won't ever", "prevent injuries"). Use softened language where needed ("escape-resistant", "helps reduce", "designed to").
Label each:
SEO-Focused: keyword-dense, includes core search terms for this product, factual, 150–200 chars
Benefit-Driven: leads with the buyer outcome that solves the top pain point, 150–200 chars
Short: under 60 characters, clean, mobile-friendly
Premium: brand-forward, aspirational, avoids feature dumping

【五点描述】
Five Amazon bullet points. Each bullet directly addresses one specific pain point from above.
Format: FEATURE IN CAPS — Benefit sentence (1–2 sentences max).
No absolute claims ("won't", "guaranteed", "prevents", "proof"). Use "helps reduce", "designed to", "built to help", "less likely to" instead.
Natural Amazon listing tone. Only product-relevant terminology.

【商品描述】
Two product description versions. Use only language relevant to this product — no features or use cases from other categories.

Short Product Description (80–120 words):
Open with the buyer frustration from the reviews. Show how this product addresses it. Clear paragraphs. Copyable as-is. No exaggerated claims.

Long Product Description (150–200 words):
Expand on the key pain points and how the product addresses each. Conversational tone. No absolute safety claims. Ends with a quiet call to action.

【广告文案】
Generate 4–5 ad copy sets. Choose the angles that are most relevant to THIS product and the pain points extracted above. Name each angle based on what it addresses (e.g. "Escape / safety angle", "Comfort / no-chafe angle", "Easy adjustment angle" for a dog harness — or "Battery life angle", "Comfort angle", "Call clarity angle" for earbuds). Do NOT use angles that belong to a different product category.

For each angle, output on separate labeled lines:
[Angle name]:
Hook: [1 sentence — starts from a specific buyer frustration]
Body: [2–3 sentences]
CTA: [short phrase]

Each angle must reflect a different pain point. Do not repeat the same idea across angles.

【钩子文案】
Eight hook lines for TikTok, Facebook, and short-form ads. Each hook must relate to a specific pain point from THIS product's reviews. Mix formats: statements, questions, scenario openers. No "Introducing". No emoji. No more than 3 questions out of the 8.

【关键词】
Primary Keywords: 8–12 keywords for this product category, comma-separated
Long-Tail Keywords: 8–12 phrases, comma-separated
Buyer-Intent Keywords: 8–12 phrases, comma-separated
Only include realistic, search-relevant terms for this product. No terms from unrelated categories.

【客服回复】
Identify the 3–5 most common negative review scenarios for THIS product based on the pain points above. Generate one reply template per scenario. Use the actual complaint type as the template label — do not use labels from a different product category.

For each use this exact format:
Template [N] - [Complaint type specific to this product]:
Scenario: [one sentence describing what the customer complained about]
Reply: [polite, natural, professional English, 3–5 sentences — suggest checking setup, sizing, usage, or contacting support where appropriate]

Rules: Do not admit serious defects. Do not over-apologize. Do not make promises you cannot keep. Only use language appropriate for this product.

【合规提醒】
Review all copy above. Because you already applied compliant language throughout (per CRITICAL RULE 2), most sections should be clean. Your job here is to catch anything that slipped through.

Check specifically for:
- Any remaining absolute safety or performance claims ("won't", "guaranteed", "-proof" used absolutely, "prevents", "100%")
- Medical or health claims ("prevents choking", "prevents neck injuries", "prevents strain")
- Superlatives that can't be substantiated ("best", "most durable", "most secure")
- SEO Keywords that contain banned phrases (e.g. "escape-proof harness" in the keyword list)

For each remaining issue output:
Risky claim: "[exact phrase still present in the copy above]"
Why risky: [one sentence]
Safer alternative: "[ready-to-use replacement]"

If you flagged any issues above, end with:
ℹ Some claims should be softened before publishing. Use "helps reduce," "designed to support," and "escape-resistant" instead of absolute guarantees.

If after genuine review no issues remain:
ℹ No major compliance issues detected after applying safer wording throughout.

SELF-CHECK before outputting — fix any issue before proceeding:
(1) Does any section contain terminology from a different product category (e.g. "escape-resistant" in a backpack listing, "laptop compartment" in a dog harness listing)? → Remove it.
(2) Does the copy contain any fabricated numbers or specs the user did not provide (e.g. "tested for 10,000 cycles", "holds 50kg")? → Replace with general durability language.
(3) If the product is water-resistant (not waterproof), does any section say "waterproof", "keeps everything dry", or use a "Waterproofing angle"? → Change to "water-resistant" / "rain protection angle".
(4) Does any section contain "protects against drops", "drop-proof", "protect your tech", or "shockproof"? → Replace with padded/cushioning language.
(5) Does the copy use "won't", "guaranteed", absolute "-proof", "prevents", or "100%"? → Rewrite using RULE 2e replacements.
(6) Do the Customer Reply labels match this product's actual pain points (not another product's)?
(7) Do the Ad Copy angles match this product category?
(8) Does the Compliance Notes closing summary use product-specific wording (not a copied template from another category)?
(9) CONSISTENCY CHECK — For every phrase flagged as "Risky claim" in 【合规提醒】: does that exact phrase (or a close variant) still appear anywhere in 【标题】,【五点描述】,【商品描述】,【广告文案】,【钩子文案】, or 【关键词】? If yes, go back and replace it with the Safer alternative before outputting. The final body copy must not contain any expression already identified as risky.

Rules: Write ALL copy in {language}. Keep all 9 Chinese markers exactly as shown. Output NOTHING outside these 9 sections.\
"""

SHOPIFY_COPY_PROMPT = """\
You are an expert Shopify/DTC ecommerce copywriter. Write compelling product copy for a Shopify store.
{competitor_context}{category_context}
Output EXACTLY these 5 sections using the Chinese markers shown:

【头条】
Three headline variations on separate lines, each with its label:
Short & Punchy: under 8 words, bold claim or hook
Benefit-Led: leads with what the buyer gains
Story: opens with a relatable buyer situation

【产品页文案】
Full product page copy, 200–250 words. Conversational, story-driven, benefit-first. End with a clear reason to buy now.

【卖点板块】
Four benefit sections. Each must follow this format exactly:
✦ [Benefit Headline]
[2-3 sentences of supporting detail.]

【FAQ】
Five buyer FAQ pairs. Format exactly:
Q: [question]
A: [answer in 1-2 sentences]

【邮件文案】
Subject line: (under 50 chars)
Preview text: (under 90 chars)
Body: (3-4 sentences, conversational, ends with CTA)

Rules: Write ALL copy in {language}. Keep markers in Chinese. Output NOTHING outside these 5 sections.\
"""

ADS_COPY_PROMPT = """\
You are an expert performance marketing copywriter for Facebook, TikTok, and Google ads.
Write high-converting ad copy based on the product information provided.
{competitor_context}{category_context}
Output EXACTLY these 5 sections using the Chinese markers shown:

【钩子文案】
Ten hook lines for ad openers and TikTok videos. One sentence each. No "Introducing". Lead with pain, curiosity, or a bold claim. These are the first thing the buyer sees — make them stop scrolling.

【主文案】
Facebook Version: 3-4 sentences. Problem → solution → benefit → CTA.
TikTok Version: 2-3 sentences. Hook continuation → proof or benefit → CTA.

【标题】
Six short ad headlines for Facebook/Google. Under 30 characters each. Benefit-focused, punchy, test-ready. One per line.

【CTA】
Six call-to-action button texts. Mix urgency, benefit, and curiosity. One per line.

【短视频脚本】
30-second video ad script in this format:
Hook (0-3s): [opening line or action]
Problem (3-8s): [relatable pain point]
Solution (8-18s): [product demo or benefit]
Proof (18-24s): [social proof or result]
CTA (24-30s): [closing call to action]

Rules: Write ALL copy in {language}. Keep markers in Chinese. Output NOTHING outside these 5 sections.\
"""

_COPY_PROMPTS = {
    'Amazon':  AMAZON_COPY_PROMPT,
    'Shopify': SHOPIFY_COPY_PROMPT,
    'Ads':     ADS_COPY_PROMPT,
}


async def generate_copy(req: CopyRequest, client: httpx.AsyncClient) -> str:
    if not API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY 环境变量")

    platform = req.platform or "Amazon"
    prompt_template = _COPY_PROMPTS.get(platform, AMAZON_COPY_PROMPT)

    competitor_context = ""
    if req.competitor_insights:
        competitor_context = (
            "IMPORTANT — Use these competitor insights to sharpen every section of the copy:\n"
            f"{req.competitor_insights}\n\n"
        )

    category_context = f"Product category: {req.product_category}\n" if req.product_category else ""

    system_prompt = prompt_template.format(
        competitor_context=competitor_context,
        category_context=category_context,
        language=req.target_language,
    )

    user_content = f"Product information:\n{req.product_info}"
    if req.input_language:
        user_content += f"\n\n(Info provided in {req.input_language} — write all copy in {req.target_language}.)"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.85,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------
# 接口:单条
# ----------------------------------------------------------------------
def _count_result_chars(result: AnalysisResult) -> int:
    n = sum(len(c.complaint) + len(c.sample_quote) for c in result.top_complaints)
    n += sum(len(w.weakness) + len(w.opportunity) for w in result.competitor_weaknesses)
    n += sum(len(s) for s in result.improvement_ideas)
    n += sum(len(a.pain_point) + sum(len(l) for l in a.angle_lines) for a in result.winning_copy_angles)
    return n


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_review(
    req: ReviewRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="缺少用户标识 X-User-Id")

    user = await get_user(x_user_id)
    is_member = bool(user["is_member"])
    input_chars = len(req.review_text)

    # 门禁(粗筛):非会员,且"今日已用 + 本次输入"已超出每日字符额度 -> 402。
    # 调用模型前只能拿到输入长度,所以这里用输入做预判;真实的"输入+输出"在成功后再累加。
    if not is_member and user["chars_used"] + input_chars > DAILY_FREE_CHARS:
        raise HTTPException(
            status_code=402,
            detail={"code": "DAILY_LIMIT_REACHED", "limit": DAILY_FREE_CHARS,
                    "used": user["chars_used"],
                    "message": "Daily limit reached. Please upgrade to Pro or try again tomorrow."},
        )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            result = await call_llm(req, client)
        await save_analysis(req.review_text, req.product_category, result)
        if not is_member:
            # 累加真实消耗:本次输入 + 模型实际输出
            await add_chars_used(x_user_id, input_chars + _count_result_chars(result))
        return AnalysisResponse(success=True, data=result)
    except HTTPException:
        raise
    except json.JSONDecodeError:
        logger.exception("模型返回无法解析")
        raise HTTPException(status_code=502, detail="模型返回格式异常,请重试")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"大模型接口错误: {e.response.status_code}")
    except httpx.RequestError:
        raise HTTPException(status_code=504, detail="连接大模型超时或失败,请稍后重试")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("未知错误")
        raise HTTPException(status_code=500, detail="服务器内部错误")


# ----------------------------------------------------------------------
# 接口:生成爆款文案
# ----------------------------------------------------------------------
@app.post("/generate", response_model=CopyResponse)
async def generate(
    req: CopyRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="缺少用户标识 X-User-Id")

    user = await get_user(x_user_id)
    is_member = bool(user["is_member"])
    input_chars = len(req.product_info)

    # 与差评分析共用同一个每日字符池
    if not is_member and user["chars_used"] + input_chars > DAILY_FREE_CHARS:
        raise HTTPException(
            status_code=402,
            detail={"code": "DAILY_LIMIT_REACHED", "limit": DAILY_FREE_CHARS,
                    "used": user["chars_used"],
                    "message": "Daily limit reached. Please upgrade to Pro or try again tomorrow."},
        )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            copy_text = await generate_copy(req, client)
        if not copy_text:
            raise HTTPException(status_code=502, detail="模型未返回文案内容,请重试")
        await save_copy(req, copy_text)
        if not is_member:
            await add_chars_used(x_user_id, input_chars + len(copy_text))
        return CopyResponse(success=True, data=CopyResult(copy=copy_text))
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"大模型接口错误: {e.response.status_code}")
    except httpx.RequestError:
        raise HTTPException(status_code=504, detail="连接大模型超时或失败,请稍后重试")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("文案生成未知错误")
        raise HTTPException(status_code=500, detail="服务器内部错误")


# ----------------------------------------------------------------------
# 接口:文案生成历史(回看)
# ----------------------------------------------------------------------
@app.get("/copies")
async def list_copies(limit: int = 20, offset: int = 0):
    """按时间倒序返回最近生成的文案,支持分页。"""
    limit = max(1, min(limit, 100))
    db = get_db()
    async with db.execute(
        """SELECT id, created_at, product_info, product_category,
                  target_language, platform, copy_text
           FROM copies ORDER BY id DESC LIMIT ? OFFSET ?""",
        (limit, max(0, offset)),
    ) as cur:
        rows = await cur.fetchall()
    async with db.execute("SELECT COUNT(*) c FROM copies") as cur:
        total = (await cur.fetchone())["c"]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [dict(r) for r in rows],
    }


# ----------------------------------------------------------------------
# 接口:批量(并发,单条失败不影响整体)
# ----------------------------------------------------------------------
@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(req: BatchReviewRequest):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(idx: int, item: ReviewRequest, client: httpx.AsyncClient) -> BatchItemResult:
        async with semaphore:
            try:
                result = await call_llm(item, client)
                await save_analysis(item.review_text, item.product_category, result)
                return BatchItemResult(index=idx, success=True, data=result)
            except Exception as e:
                logger.warning("批量第 %d 条失败: %s", idx, e)
                return BatchItemResult(index=idx, success=False, error=str(e))

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        tasks = [worker(i, item, client) for i, item in enumerate(req.reviews)]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda r: r.index)
    succeeded = sum(1 for r in results if r.success)
    return BatchAnalysisResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


# ----------------------------------------------------------------------
# 接口:趋势统计
# ----------------------------------------------------------------------
@app.get("/stats/trends")
async def stats_trends(days: int = 30, product_category: str | None = None):
    """返回:问题分类排行、严重程度分布、情绪分布、每日分析量。"""
    db = get_db()

    where = "WHERE created_at >= datetime('now', ?)"
    params: list = [f"-{int(days)} days"]
    if product_category:
        where += " AND product_category = ?"
        params.append(product_category)

    # 情绪分布
    async with db.execute(
        f"SELECT overall_sentiment, COUNT(*) c FROM analyses {where} GROUP BY overall_sentiment",
        params,
    ) as cur:
        sentiment_rows = await cur.fetchall()

    # 每日分析量
    async with db.execute(
        f"SELECT date(created_at) d, COUNT(*) c FROM analyses {where} GROUP BY d ORDER BY d",
        params,
    ) as cur:
        daily_rows = await cur.fetchall()

    # 取出 issues_json 在 Python 侧聚合(SQLite 无原生 JSON 展开)
    async with db.execute(f"SELECT issues_json FROM analyses {where}", params) as cur:
        issue_rows = await cur.fetchall()

    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for row in issue_rows:
        for issue in json.loads(row["issues_json"] or "[]"):
            cat = issue.get("category", "其他")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            sev = issue.get("severity")
            if sev in severity_counts:
                severity_counts[sev] += 1

    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "range_days": days,
        "product_category": product_category,
        "sentiment_distribution": {r["overall_sentiment"]: r["c"] for r in sentiment_rows},
        "severity_distribution": severity_counts,
        "top_issue_categories": [{"category": k, "count": v} for k, v in top_categories],
        "daily_volume": [{"date": r["d"], "count": r["c"]} for r in daily_rows],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


# ----------------------------------------------------------------------
# 接口:用量查询(前端用来显示剩余免费次数 / 会员状态)
# ----------------------------------------------------------------------
@app.get("/usage")
async def usage(x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="缺少用户标识 X-User-Id")
    user = await get_user(x_user_id)
    is_member = bool(user["is_member"])
    used = user["chars_used"]
    return {
        "is_member": is_member,
        "chars_used": used,
        "limit": DAILY_FREE_CHARS,
        "remaining": None if is_member else max(0, DAILY_FREE_CHARS - used),
        "subscription_status": user["subscription_status"],
    }


# ----------------------------------------------------------------------
# 接口:创建 Stripe 订阅收银台(每月 $9.90)
# ----------------------------------------------------------------------
@app.post("/create-checkout-session")
async def create_checkout_session(x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="缺少用户标识 X-User-Id")
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="后台未配置 Stripe 密钥或价格 ID")

    user = await get_user(x_user_id)
    try:
        # stripe-python 是同步库,放线程池里跑,避免阻塞事件循环
        session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            client_reference_id=x_user_id,                 # 关键:把用户 ID 带进会话
            metadata={"user_id": x_user_id},
            customer=user.get("stripe_customer_id") or None,
            success_url=f"{FRONTEND_URL}?paid=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}?paid=cancel",
            allow_promotion_codes=True,
        )
        return {"url": session.url}
    except Exception as e:
        logger.exception("创建 Checkout 会话失败")
        raise HTTPException(status_code=502, detail=f"创建支付会话失败: {e}")


# ----------------------------------------------------------------------
# 接口:创建 Stripe Billing Portal(会员自助管理订阅:取消/换卡/看账单)
# ----------------------------------------------------------------------
@app.post("/create-portal-session")
async def create_portal_session(x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="缺少用户标识 X-User-Id")
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="后台未配置 Stripe 密钥")

    user = await get_user(x_user_id)
    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=404, detail="未找到该用户的订阅信息")

    try:
        session = await asyncio.to_thread(
            stripe.billing_portal.Session.create,
            customer=customer_id,
            return_url=FRONTEND_URL,
        )
        return {"url": session.url}
    except Exception as e:
        logger.exception("创建 Billing Portal 会话失败")
        raise HTTPException(status_code=502, detail=f"创建管理页失败: {e}")


# ----------------------------------------------------------------------
# 接口:Stripe Webhook(支付/订阅状态变更的唯一可信来源)
# ----------------------------------------------------------------------
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="后台未配置 STRIPE_WEBHOOK_SECRET")
    try:
        # 必须用签名校验,确认请求确实来自 Stripe,否则任何人都能伪造开通会员
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.warning("Webhook 签名校验失败: %s", e)
        raise HTTPException(status_code=400, detail="invalid signature")

    etype = event["type"]
    obj = event["data"]["object"]

    def field(o, key, default=None):
        """安全地从 Stripe 返回的对象里取一个字段。
        不同版本的 stripe-python 库,这个对象有时表现得像字典(.get() 可用),
        有时是新版的属性对象(.get() 反而会报错,要用方括号或属性访问)。
        这里统一用方括号取值,兼容两种情况,避免再因为库版本差异而崩溃。"""
        try:
            value = o[key]
            return value if value is not None else default
        except (KeyError, TypeError):
            return getattr(o, key, default)

    if etype == "checkout.session.completed":
        # 付款完成 -> 开通会员
        metadata = field(obj, "metadata") or {}
        user_id = field(obj, "client_reference_id") or field(metadata, "user_id")
        customer = field(obj, "customer")
        if user_id:
            await set_member(user_id, True, customer, "active")
            logger.info("用户 %s 订阅开通", user_id)

    elif etype == "customer.subscription.updated":
        # 续费成功 / 进入宽限期 / 真正取消等 -> 按状态决定是否仍为会员。
        # 注意:订阅刚创建的瞬间,Stripe 可能先发一次状态为 "incomplete"(等待首次扣款确认)
        # 的过渡态通知,这不代表订阅失败,不应该把刚由 checkout.session.completed 开通的
        # 会员资格撤销掉。所以只在状态明确是"真正活跃"或"明确不活跃"时才更新,
        # 像 incomplete 这种中间态直接忽略,交给后续事件自然收敛。
        status = field(obj, "status")
        DEFINITIVE_ACTIVE = {"active", "trialing"}
        DEFINITIVE_INACTIVE = {"canceled", "unpaid", "past_due", "incomplete_expired"}
        if status in DEFINITIVE_ACTIVE:
            await set_member_by_customer(field(obj, "customer"), True, status)
        elif status in DEFINITIVE_INACTIVE:
            await set_member_by_customer(field(obj, "customer"), False, status)
        # 其余状态(如 incomplete)忽略,不动会员资格

    elif etype == "customer.subscription.deleted":
        # 订阅取消/到期 -> 收回会员资格
        await set_member_by_customer(field(obj, "customer"), False, field(obj, "status"))

    return {"received": True}
