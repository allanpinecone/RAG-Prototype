"""
Generate 200K synthetic fund documents, chunk them, embed via Pinecone
hosted inference (llama-text-embed-v2 @ 2048d), and write to Parquet
files formatted for Pinecone bulk import.

Usage:
    python backend/generate_dummy_data.py
"""

import os
import random
import hashlib
import time
import re
import string
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

NUM_DOCUMENTS = 200_000
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 96
PARQUET_BATCH_SIZE = 100_000
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SEED = 42

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ---------------------------------------------------------------------------
# Data pools for randomization
# ---------------------------------------------------------------------------

MANAGERS = [
    "Vanguard", "BlackRock", "Fidelity International", "Schroders",
    "Jupiter", "Invesco", "Aberdeen Standard", "Legal & General",
    "M&G Investments", "Baillie Gifford", "Rathbone", "Liontrust",
    "Janus Henderson", "HSBC Asset Management", "Coutts",
    "Aviva Investors", "Royal London", "Columbia Threadneedle",
    "Ninety One", "Artemis", "AXA Investment Managers", "BNY Mellon",
    "Dimensional", "State Street Global Advisors", "JP Morgan AM",
    "Goldman Sachs AM", "UBS Asset Management", "Pictet",
    "Amundi", "T. Rowe Price",
]

STRATEGIES = [
    "LifeStrategy", "ActiveLife", "Growth", "Income", "Balanced",
    "Cautious", "Dynamic", "Adventurous", "Defensive", "Moderate",
    "Conservative", "Progressive", "Opportunity", "Strategic",
    "Tactical", "Enhanced", "Core", "Select", "Premium", "Foundation",
    "Sustainable", "Climate Aware", "ESG Leaders", "Multi-Factor",
    "Dividend", "Value", "Momentum", "Quality", "Low Volatility",
    "High Yield",
]

GEOGRAPHIES = [
    "Global", "US", "UK", "European", "Emerging Markets",
    "Asia Pacific", "Japan", "North American", "International",
    "World", "Pacific", "EMEA", "Pan European", "Greater China",
    "Developed Markets",
]

ASSET_CLASSES = [
    "Equity", "Bond", "Multi-Asset", "Fixed Income", "Index",
    "Real Estate", "Gilt", "Corporate Bond", "Government Bond",
    "Aggregate Bond",
]

EQUITY_PCTS = [20, 30, 40, 50, 60, 70, 80, 90, 100]

SHARE_CLASSES = [
    "Class A Acc GBP", "Class A Inc GBP", "Class B Acc GBP",
    "Class C Acc GBP", "Class I Acc GBP", "Class I Inc GBP",
    "Class R Acc GBP", "Class R Inc GBP", "Class 3 Acc GBP",
    "Class S Acc GBP", "Institutional Acc GBP", "Retail Acc GBP",
    "Class A Acc EUR", "Class A Acc USD",
]

COUNTRIES = ["IE", "LU", "GB"]

DEPOSITARIES = [
    "State Street Trustees Limited",
    "BNY Mellon Trust & Depositary (UK) Limited",
    "HSBC Securities Services",
    "Northern Trust Global Services",
    "J.P. Morgan Europe Limited",
    "Citibank Europe plc",
]

REGULATORS = {
    "IE": "Central Bank of Ireland",
    "LU": "Commission de Surveillance du Secteur Financier (CSSF)",
    "GB": "Financial Conduct Authority (FCA)",
}

SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Industrials", "Energy", "Materials", "Utilities",
    "Communication Services", "Consumer Staples", "Real Estate",
]

HOLDINGS = [
    "Apple Inc", "Microsoft Corp", "Amazon.com Inc", "NVIDIA Corp",
    "Alphabet Inc Class A", "Meta Platforms Inc", "Tesla Inc",
    "Berkshire Hathaway Inc", "UnitedHealth Group Inc",
    "Johnson & Johnson", "JPMorgan Chase & Co", "Visa Inc",
    "Procter & Gamble Co", "Mastercard Inc", "Eli Lilly & Co",
    "Home Depot Inc", "Samsung Electronics Co", "Taiwan Semiconductor",
    "ASML Holding NV", "Novo Nordisk A/S", "Nestle SA",
    "LVMH Moet Hennessy", "Shell plc", "AstraZeneca plc",
    "Unilever plc", "HSBC Holdings plc", "BP plc",
    "GlaxoSmithKline plc", "Rio Tinto plc", "Diageo plc",
    "British American Tobacco", "Reckitt Benckiser",
    "London Stock Exchange Group", "National Grid plc",
    "Barclays plc", "Lloyds Banking Group", "Rolls-Royce Holdings",
    "BAE Systems plc", "Glencore plc", "Anglo American plc",
]

EXCLUSION_THEMES = [
    "controversial weapons", "tobacco production",
    "thermal coal extraction", "arctic oil and gas drilling",
    "UN Global Compact violators", "civilian firearms manufacture",
    "nuclear weapons proliferation", "cluster munitions",
    "non-certified palm oil", "deforestation-linked commodities",
]

BENCHMARKS = [
    "FTSE All-Share Index", "FTSE 100 Index", "FTSE 250 Index",
    "S&P 500 Index", "MSCI World Index", "MSCI ACWI Index",
    "MSCI Emerging Markets Index",
    "Bloomberg Global Aggregate Bond Index",
    "FTSE Actuaries UK Conventional Gilts All Stocks Index",
    "Bloomberg Sterling Aggregate Bond Index",
    "MSCI Europe ex-UK Index", "MSCI Pacific Index",
    "Bloomberg US Aggregate Bond Index",
    "ICE BofA Sterling Corporate Bond Index",
]

RISK_FACTORS = [
    "Market risk: the value of investments may fall as well as rise and investors may not get back the amount originally invested.",
    "Currency risk: changes in exchange rates may reduce or increase the returns an investor may expect to receive independent of the performance of such assets.",
    "Interest rate risk: changes in interest rates will affect the value of fixed income securities. Generally, as interest rates rise the price of a fixed income security falls.",
    "Credit risk: the issuer of a security held by the fund may default on its obligation to pay interest or repay capital, or both, which would negatively affect the fund's value.",
    "Liquidity risk: in difficult market conditions, the fund may not be able to sell a security for full value or at all. This could affect performance and could cause the fund to defer or suspend redemptions.",
    "Counterparty risk: the insolvency of any institution providing services such as safekeeping of assets or acting as counterparty to derivatives may expose the fund to financial loss.",
    "Operational risk: failures or delays in operational processes may negatively affect the fund. There is a risk that any company responsible for administering the fund may fail.",
    "Concentration risk: a portfolio that is concentrated in a small number of holdings may be more volatile than a diversified portfolio.",
    "Emerging markets risk: emerging markets may be more volatile and carry higher risks than developed markets, including political instability and currency fluctuation.",
    "Derivative risk: derivatives are highly sensitive to changes in the value of the underlying asset and can magnify both losses and gains, resulting in greater fluctuations in the fund's value.",
    "Valuation risk: assets held by the fund may be difficult to value accurately, particularly during periods of market stress or when markets are illiquid.",
    "Regulatory risk: changes in government regulations and tax laws could adversely affect the value of investments or the returns available to investors.",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def generate_fund_pool(n: int, rng: random.Random) -> list[dict]:
    funds = []
    seen = set()
    attempts = 0
    while len(funds) < n and attempts < n * 20:
        attempts += 1
        manager = rng.choice(MANAGERS)
        strategy = rng.choice(STRATEGIES)
        geo = rng.choice(GEOGRAPHIES)
        asset = rng.choice(ASSET_CLASSES)
        eq_pct = rng.choice(EQUITY_PCTS)
        name = f"{manager} {strategy} {eq_pct}% {geo} {asset} Fund"
        if name in seen:
            continue
        seen.add(name)
        country = rng.choice(COUNTRIES)
        funds.append({
            "name": name,
            "manager": manager,
            "strategy": strategy,
            "geography": geo,
            "asset_class": asset,
            "equity_pct": eq_pct,
            "share_class": rng.choice(SHARE_CLASSES),
            "country": country,
            "isin": _make_isin(country, rng),
            "sedol": _make_sedol(rng),
            "srri": rng.randint(1, 7),
            "ocf": round(rng.uniform(0.06, 1.50), 2),
            "nav": round(rng.uniform(50, 500), 2),
            "launch_year": rng.randint(1990, 2024),
            "benchmark": rng.choice(BENCHMARKS),
            "depositary": rng.choice(DEPOSITARIES),
        })
    return funds


def _make_isin(country: str, rng: random.Random) -> str:
    chars = string.ascii_uppercase + string.digits
    return country + "".join(rng.choices(chars, k=9)) + str(rng.randint(0, 9))


def _make_sedol(rng: random.Random) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(rng.choices(chars, k=7))


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _make_record_id(source_file: str, chunk_idx: int) -> str:
    path_hash = hashlib.md5(source_file.encode()).hexdigest()[:10]
    return f"{path_hash}_chunk{chunk_idx:04d}"


def _random_date(rng: random.Random, min_year=2022, max_year=2025) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(min_year, max_year)}"


def _random_allocation(rng: random.Random, labels: list[str], n: int = 5) -> list[tuple[str, float]]:
    picks = rng.sample(labels, min(n, len(labels)))
    weights = [rng.random() for _ in picks]
    total = sum(weights)
    return [(label, round(w / total * 100, 1)) for label, w in zip(picks, weights)]


def _random_holdings(rng: random.Random, n: int = 10) -> list[tuple[str, float]]:
    picks = rng.sample(HOLDINGS, min(n, len(HOLDINGS)))
    weights = sorted([rng.random() for _ in picks], reverse=True)
    total = sum(weights)
    return [(name, round(w / total * 100, 1)) for name, w in zip(picks, weights)]


# ---------------------------------------------------------------------------
# Text cleaning & chunking (matching ingest.py)
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E£€¥%°±²³µ·¼½¾]", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


# ---------------------------------------------------------------------------
# Document generators
# ---------------------------------------------------------------------------


def generate_kiid(fund: dict, rng: random.Random) -> str:
    objective = rng.choice([
        f"provide long-term capital growth by investing primarily in "
        f"{fund['geography'].lower()} {fund['asset_class'].lower()} securities",
        f"track the performance of the {fund['benchmark']}, before fees and "
        f"expenses, by investing in the constituent securities of the index",
        f"achieve a total return in excess of the {fund['benchmark']} over "
        f"rolling five-year periods through capital growth and income",
        f"maximise total return by investing at least 80% of its assets in "
        f"{fund['geography'].lower()} {fund['asset_class'].lower()} securities",
        f"deliver consistent risk-adjusted returns by maintaining a "
        f"diversified portfolio of {fund['geography'].lower()} securities "
        f"across market capitalisations",
    ])

    risk_paragraph = rng.choice([
        f"Historical data may not be a reliable indication of the future risk "
        f"profile of the fund. The risk category shown is not guaranteed and "
        f"may change over time. The lowest category does not mean a risk-free "
        f"investment. This fund is rated {fund['srri']} due to the nature of "
        f"its investments which include the risks listed below. "
        f"{rng.choice(RISK_FACTORS)} {rng.choice(RISK_FACTORS)}",

        f"The risk indicator assumes you keep the product for the recommended "
        f"holding period of {rng.randint(3, 7)} years. The actual risk can "
        f"vary significantly if you cash in at an early stage and you may get "
        f"back less than you invest. The summary risk indicator is a guide to "
        f"the level of risk of this product compared to other products. This "
        f"fund has been assigned a risk level of {fund['srri']} out of 7. "
        f"{rng.choice(RISK_FACTORS)}",

        f"The risk and reward indicator is calculated using historical and "
        f"simulated data. It may not be a reliable indication of the future "
        f"risk profile of the fund. The fund's category of {fund['srri']} "
        f"reflects the volatility of the fund's reference portfolio, which "
        f"comprises {fund['equity_pct']}% equities and "
        f"{100 - fund['equity_pct']}% bonds. {rng.choice(RISK_FACTORS)}",
    ])

    entry_charge = round(rng.choice([0, 0, 0, rng.uniform(0.5, 5.0)]), 2)
    exit_charge = round(rng.choice([0, 0, 0, rng.uniform(0.25, 2.0)]), 2)
    perf_fee = rng.choice(["None", "None", "None",
                           f"{rng.uniform(5, 20):.1f}% of outperformance"])
    returns = [round(rng.uniform(-20, 35), 1) for _ in range(5)]
    yr = 2025
    income = rng.choice(["reinvested", "distributed quarterly",
                         "distributed annually"])
    holding = rng.randint(3, 7)

    return f"""KEY INVESTOR INFORMATION

This document provides you with key investor information about this fund. It is not marketing material. The information is required by law to help you understand the nature and the risks of investing in this fund. You are advised to read it so you can make an informed decision about whether to invest.

{fund['name']} ({fund['share_class']})
ISIN: {fund['isin']}  SEDOL: {fund['sedol']}
This fund is managed by {fund['manager']}.

OBJECTIVES AND INVESTMENT POLICY

The fund aims to {objective}. The fund may use financial derivative instruments for efficient portfolio management and hedging purposes. Income from the fund will be {income} in the {fund['share_class']} share class. You may buy and sell shares on any UK business day. The recommended holding period is at least {holding} years. The ongoing charges figure is {fund['ocf']:.2f}% per annum and includes the annual management charge and other operating costs. The fund is classified under Article {rng.choice([6, 8, 9])} of the EU Sustainable Finance Disclosure Regulation.

RISK AND REWARD PROFILE

Lower Risk - Higher Risk
Typically lower rewards - Typically higher rewards
1 | 2 | 3 | 4 | 5 | 6 | 7
This fund is currently rated {fund['srri']}.

{risk_paragraph}

CHARGES FOR THIS FUND

The charges you pay are used to cover the costs of running the fund, including the costs of marketing and distributing it. These charges reduce the potential growth of your investment.
Entry charge: {entry_charge:.2f}%
Exit charge: {exit_charge:.2f}%
Ongoing charges: {fund['ocf']:.2f}%
Performance fee: {perf_fee}
The ongoing charges figure is based on expenses for the year ending December {yr - 1}. This figure may vary from year to year. It excludes portfolio transaction costs.

PAST PERFORMANCE

Past performance is not a reliable indicator of future results. The chart shows annual returns for each of the last five complete calendar years. Performance is calculated net of ongoing charges and is shown in GBP.
{yr - 5}: Fund {returns[0]:+.1f}% | Benchmark {returns[0] + rng.uniform(-3, 3):+.1f}%
{yr - 4}: Fund {returns[1]:+.1f}% | Benchmark {returns[1] + rng.uniform(-3, 3):+.1f}%
{yr - 3}: Fund {returns[2]:+.1f}% | Benchmark {returns[2] + rng.uniform(-3, 3):+.1f}%
{yr - 2}: Fund {returns[3]:+.1f}% | Benchmark {returns[3] + rng.uniform(-3, 3):+.1f}%
{yr - 1}: Fund {returns[4]:+.1f}% | Benchmark {returns[4] + rng.uniform(-3, 3):+.1f}%
The fund was launched in {fund['launch_year']}. Performance is shown in GBP.

PRACTICAL INFORMATION

The depositary is {fund['depositary']}. Further information about the fund, including the prospectus and latest annual report, can be obtained free of charge from {fund['manager']}. The most recent share price is published daily on the manager's website. This fund is authorised in {fund['country']} and regulated by the {REGULATORS[fund['country']]}. {fund['manager']} is authorised and regulated by the Financial Conduct Authority. Tax legislation in the fund's home country may have an impact on your personal tax position. This key investor information is accurate as at {_random_date(rng)}."""


def generate_factsheet(fund: dict, rng: random.Random) -> str:
    holdings = _random_holdings(rng, 10)
    sectors = _random_allocation(rng, SECTORS, 6)
    returns_1m = round(rng.uniform(-8, 8), 2)
    returns_3m = round(rng.uniform(-12, 15), 2)
    returns_ytd = round(rng.uniform(-15, 25), 2)
    returns_1y = round(rng.uniform(-20, 35), 2)
    returns_3y = round(rng.uniform(-10, 60), 2)
    returns_5y = round(rng.uniform(-5, 100), 2)
    fund_size = round(rng.uniform(50, 15000), 1)
    num_holdings = rng.randint(30, 2500)
    date = _random_date(rng)

    approach = rng.choice([
        f"The fund employs a systematic, rules-based investment approach "
        f"designed to capture broad market exposure to {fund['geography'].lower()} "
        f"{fund['asset_class'].lower()} markets. Portfolio construction is driven by "
        f"quantitative models that seek to optimise risk-adjusted returns while "
        f"maintaining diversification across sectors and regions.",

        f"The investment team follows a fundamental, bottom-up stock selection "
        f"process. Analysts conduct in-depth company research, evaluating "
        f"competitive positioning, management quality, and balance sheet "
        f"strength. The portfolio typically holds {num_holdings} securities, with "
        f"active positions taken relative to the {fund['benchmark']}.",

        f"The fund adopts a multi-asset approach, blending {fund['equity_pct']}% "
        f"in equities with {100 - fund['equity_pct']}% in fixed income and "
        f"alternative assets. The strategic asset allocation is set by the "
        f"investment committee and reviewed quarterly, with tactical tilts "
        f"applied when the team identifies compelling opportunities.",

        f"This passively managed fund seeks to replicate the performance of "
        f"the {fund['benchmark']} by holding a representative sample of the "
        f"constituent securities. The fund uses an optimised sampling "
        f"methodology to minimise tracking error while managing transaction costs.",
    ])

    commentary = rng.choice([
        f"During the period, global equity markets experienced heightened "
        f"volatility driven by central bank policy decisions and evolving "
        f"macroeconomic conditions. The {fund['geography']} region "
        f"saw mixed performance across sectors, with technology and "
        f"healthcare leading while energy and materials lagged. The fund's "
        f"overweight position in quality growth stocks contributed positively "
        f"to relative returns.",

        f"Markets rallied broadly over the quarter, supported by resilient "
        f"corporate earnings and moderating inflation expectations. Within "
        f"the portfolio, our allocation to {fund['geography'].lower()} "
        f"equities benefited from improving economic sentiment. Stock "
        f"selection in financials and industrials added value, while our "
        f"underweight in utilities detracted marginally.",

        f"The reporting period was characterised by a rotation from growth "
        f"to value stocks, as rising bond yields prompted a reassessment of "
        f"equity valuations. The fund navigated this environment by "
        f"maintaining its disciplined approach to portfolio construction. "
        f"Exposure to {fund['geography'].lower()} markets provided "
        f"diversification benefits during periods of elevated uncertainty.",
    ])

    holdings_text = "\n".join(
        f"  {i + 1}. {name} — {pct}%" for i, (name, pct) in enumerate(holdings)
    )
    sector_text = "\n".join(
        f"  {label}: {pct}%" for label, pct in sectors
    )

    return f"""FUND FACTSHEET — {fund['name']}
As at {date}

FUND OVERVIEW

Fund name: {fund['name']}
Manager: {fund['manager']}
Benchmark: {fund['benchmark']}
ISIN: {fund['isin']}
Share class: {fund['share_class']}
Fund size: £{fund_size:,.1f}m
Number of holdings: {num_holdings}
Launch date: {fund['launch_year']}
Ongoing charges (OCF): {fund['ocf']:.2f}%
Risk rating (SRRI): {fund['srri']} out of 7
NAV per share: £{fund['nav']:.2f}
Dealing frequency: Daily
Distribution policy: {rng.choice(["Accumulation", "Income - quarterly", "Income - semi-annual"])}
Domicile: {fund['country']}

INVESTMENT APPROACH

{approach}

PERFORMANCE (%)

                1 month   3 months   YTD      1 year   3 years   5 years
Fund            {returns_1m:+.2f}    {returns_3m:+.2f}     {returns_ytd:+.2f}   {returns_1y:+.2f}   {returns_3y:+.2f}    {returns_5y:+.2f}
Benchmark       {returns_1m + rng.uniform(-2, 2):+.2f}    {returns_3m + rng.uniform(-3, 3):+.2f}     {returns_ytd + rng.uniform(-4, 4):+.2f}   {returns_1y + rng.uniform(-5, 5):+.2f}   {returns_3y + rng.uniform(-8, 8):+.2f}    {returns_5y + rng.uniform(-10, 10):+.2f}

Performance is calculated net of fees in GBP. Past performance does not guarantee future results.

TOP 10 HOLDINGS

{holdings_text}

SECTOR ALLOCATION

{sector_text}

MANAGER COMMENTARY

{commentary}

IMPORTANT INFORMATION

This factsheet is issued by {fund['manager']} and is for information purposes only. It does not constitute investment advice or a recommendation to buy or sell any security. The value of investments and income from them can go down as well as up and you may not get back the amount you originally invested. Past performance is not a reliable indicator of future results. {fund['manager']} is authorised and regulated by the Financial Conduct Authority."""


def generate_esg_disclosure(fund: dict, rng: random.Random) -> str:
    exclusions = rng.sample(EXCLUSION_THEMES, rng.randint(4, 7))
    carbon_intensity = round(rng.uniform(20, 350), 1)
    benchmark_carbon = round(carbon_intensity * rng.uniform(1.1, 2.5), 1)
    portfolio_coverage = round(rng.uniform(60, 98), 1)
    green_revenue = round(rng.uniform(5, 45), 1)
    fossil_revenue = round(rng.uniform(0, 15), 1)
    engagement_count = rng.randint(50, 500)
    voting_meetings = rng.randint(100, 2000)
    date = _random_date(rng)

    approach = rng.choice([
        f"The fund integrates environmental, social, and governance (ESG) "
        f"factors throughout the investment process. ESG analysis is embedded "
        f"in the fundamental research conducted by our analysts and portfolio "
        f"managers. We believe that companies managing ESG risks effectively "
        f"are more likely to deliver sustainable long-term returns. The fund "
        f"excludes companies involved in {', '.join(exclusions[:3])}, and "
        f"applies a best-in-class approach to select companies demonstrating "
        f"strong ESG practices relative to their peers.",

        f"This fund is classified as Article 8 under the EU Sustainable "
        f"Finance Disclosure Regulation (SFDR). The fund promotes "
        f"environmental and social characteristics by applying exclusionary "
        f"screening criteria and integrating ESG factors into portfolio "
        f"construction. The investment team utilises proprietary ESG scores "
        f"alongside third-party data from providers including MSCI ESG "
        f"Research, Sustainalytics, and ISS ESG. Companies that derive "
        f"significant revenue from {', '.join(exclusions[:4])} are excluded "
        f"from the investable universe.",

        f"The fund follows the FCA's Sustainability Disclosure Requirements "
        f"(SDR) framework and is categorised as a Sustainability Focus fund. "
        f"At least 70% of the fund's assets are invested in accordance with "
        f"the sustainability objective. The fund applies a robust exclusion "
        f"policy covering {', '.join(exclusions[:5])}. Engagement and proxy "
        f"voting are integral components of our stewardship approach.",
    ])

    return f"""RESPONSIBLE INVESTING INFORMATION DOCUMENT
{fund['name']}
As at {date}

SUSTAINABILITY OVERVIEW

{approach}

EXCLUSION CRITERIA

The fund applies the following exclusion criteria. Companies deriving more than {rng.choice([0, 5, 10])}% of revenue from the following activities are excluded from the portfolio:
{chr(10).join(f'  - {e.capitalize()}' for e in exclusions)}

CARBON AND CLIMATE METRICS

Weighted Average Carbon Intensity (tCO2e/$m revenue):
  Fund: {carbon_intensity:.1f}
  Benchmark ({fund['benchmark']}): {benchmark_carbon:.1f}

Portfolio carbon data coverage: {portfolio_coverage:.1f}%
Green revenue share: {green_revenue:.1f}%
Fossil fuel revenue share: {fossil_revenue:.1f}%

The fund targets a {rng.choice([30, 40, 50])}% reduction in carbon intensity relative to the benchmark. Our net-zero alignment methodology follows the Paris Aligned Investment Initiative framework, with an interim target of a {rng.choice([50, 55, 65])}% reduction in financed emissions by 2030 relative to a 2019 baseline.

STEWARDSHIP AND ENGAGEMENT

During the reporting period, {fund['manager']} conducted {engagement_count} engagement activities with portfolio companies on topics including climate transition planning, board diversity, executive remuneration, and supply chain labour practices. We voted at {voting_meetings} shareholder meetings and voted against management on {round(rng.uniform(5, 25), 1)}% of proposals, primarily on remuneration and board composition matters.

Our engagement priorities for the current year include: climate transition plans and targets, natural capital and biodiversity commitments, human rights due diligence processes, and board effectiveness and diversity.

REGULATORY DISCLOSURES

This document is prepared in accordance with the {rng.choice(["FCA Sustainability Disclosure Requirements (SDR)", "EU Sustainable Finance Disclosure Regulation (SFDR)"])}. The sustainability indicators and metrics reported herein are based on data available as at {date} and may be subject to revision. {fund['manager']} is a signatory to the UN Principles for Responsible Investment (PRI) and the UK Stewardship Code 2020. For further information on our responsible investment policies and approach, please visit our website or contact your client relationship manager."""


def generate_prospectus(fund: dict, rng: random.Random) -> str:
    risks = rng.sample(RISK_FACTORS, rng.randint(6, 10))
    sub_fund_count = rng.randint(5, 30)
    min_investment = rng.choice([1000, 2500, 5000, 10000, 50000, 100000])
    mgmt_fee = round(rng.uniform(0.10, 1.25), 2)
    admin_fee = round(rng.uniform(0.02, 0.15), 2)
    custodian_fee = round(rng.uniform(0.01, 0.05), 3)
    date = _random_date(rng)
    yr = rng.randint(2020, 2025)

    vehicle = rng.choice([
        f"an open-ended investment company with variable capital (ICVC) "
        f"incorporated in England and Wales under registered number "
        f"IC{rng.randint(100000, 999999)}",
        f"a public limited company organised as a soci\u00e9t\u00e9 d'investissement "
        f"\u00e0 capital variable (SICAV) registered in Luxembourg under number "
        f"B{rng.randint(100000, 999999)}",
        f"an Irish Collective Asset-management Vehicle (ICAV) registered "
        f"with the Central Bank of Ireland under registration number "
        f"C{rng.randint(100000, 999999)}",
    ])

    tax_section = rng.choice([
        f"The fund is exempt from UK corporation tax on capital gains and "
        f"UK income tax on interest and overseas dividends. UK resident "
        f"individual shareholders may be subject to income tax on "
        f"distributions and capital gains tax on disposal of shares. The "
        f"tax treatment depends on individual circumstances and may be "
        f"subject to change. Investors should seek independent tax advice.",

        f"The Company is subject to Luxembourg tax laws. Under current "
        f"Luxembourg law, the Company is subject to a taxe d'abonnement at "
        f"an annual rate of 0.05% of net asset value (0.01% for "
        f"institutional share classes). No Luxembourg income or capital "
        f"gains tax is payable by the Company. Shareholders may be subject "
        f"to withholding tax or other taxes in their country of residence.",

        f"As an Irish domiciled UCITS fund, the Company is generally exempt "
        f"from Irish tax on income and gains. Irish resident shareholders "
        f"are subject to tax on distributions and disposals under the gross "
        f"roll-up regime. A deemed disposal event occurs every 8 years. "
        f"Non-Irish resident investors who provide appropriate declarations "
        f"will not be subject to Irish withholding tax.",
    ])

    return f"""PROSPECTUS

{fund['manager']}
{fund['name']}

This document constitutes the prospectus for {fund['name']} (the "Fund") and has been prepared in accordance with the {rng.choice(["FCA Handbook (COLL)", "UCITS Directive (2009/65/EC)", "Central Bank (Supervision and Enforcement) Act 2013"])}. Investors should read this prospectus in its entirety before making an investment decision.

Date of publication: {date}

IMPORTANT NOTICE

This prospectus may only be distributed in jurisdictions where it is legally permitted and constitutes neither an offer nor a solicitation in any jurisdiction where such offer would be unlawful. Potential investors should consult their professional advisers as to the legal, tax, and financial implications of subscribing for, purchasing, holding, or disposing of shares.

GENERAL INFORMATION

The Fund is {vehicle}. The Fund operates as an umbrella fund comprising {sub_fund_count} sub-funds, each with distinct investment objectives and policies. The Authorised Corporate Director (ACD) is {fund['manager']}, which is authorised and regulated by the {REGULATORS[fund['country']]}. The depositary is {fund['depositary']}.

The base currency of the Fund is {rng.choice(["GBP", "EUR", "USD"])}. Shares are offered in multiple share classes including accumulation and income variants denominated in GBP, EUR, and USD.

INVESTMENT OBJECTIVES AND POLICIES

The investment objective of the Fund is to {rng.choice(["provide long-term capital growth", "achieve a total return", "generate income with capital growth potential", "track the performance of a specified benchmark index"])} by investing primarily in {fund['geography'].lower()} {fund['asset_class'].lower()} securities.

The Fund may invest in transferable securities, money market instruments, collective investment schemes (up to {rng.choice([10, 20, 30])}% of NAV), deposits, and financial derivative instruments. The Fund will invest at least {rng.choice([70, 80, 90])}% of its net assets in securities that are consistent with its investment objective. The Fund may hold up to {rng.choice([5, 10, 20])}% in cash and cash equivalents during normal market conditions.

The Fund is {rng.choice(["actively", "passively"])} managed{f" against the {fund['benchmark']}" if rng.random() > 0.3 else ""}. {rng.choice(["The benchmark is used for performance comparison purposes only and does not constrain the portfolio construction.", "The Fund aims to outperform the benchmark over rolling 3-5 year periods.", "The Fund seeks to replicate the benchmark as closely as practicable while minimising tracking error."])}

RISK FACTORS

Investment in the Fund carries risk. The value of your investment may go down as well as up and you may lose some or all of the money you invest. The following risk factors should be carefully considered:

{chr(10).join(f'{i + 1}. {r}' for i, r in enumerate(risks))}

The above list is not exhaustive. Additional risks are described in the relevant supplement for each sub-fund.

MANAGEMENT AND ADMINISTRATION

Authorised Corporate Director / Management Company: {fund['manager']}
Depositary: {fund['depositary']}
Administrator: {rng.choice(["State Street Fund Services", "BNY Mellon Fund Management", "Northern Trust Fund Administration", "HSBC Securities Services"])}
Auditor: {rng.choice(["PricewaterhouseCoopers LLP", "KPMG LLP", "Deloitte LLP", "Ernst & Young LLP"])}
Legal Adviser: {rng.choice(["Linklaters LLP", "Clifford Chance LLP", "Freshfields Bruckhaus Deringer LLP", "Allen & Overy LLP"])}

FEES AND CHARGES

Annual management charge: {mgmt_fee:.2f}%
Administration fee: {admin_fee:.2f}%
Custodian fee: {custodian_fee:.3f}%
Ongoing charges figure (OCF): {fund['ocf']:.2f}%
Maximum initial charge: {rng.choice([0, 3, 5])}%
Redemption charge: {rng.choice([0, 0, 1, 2])}%
Minimum initial investment: £{min_investment:,}
Minimum subsequent investment: £{rng.choice([500, 1000, 2500])}

The ACD may waive or reduce charges at its discretion. Transaction costs are borne by the Fund and are not included in the OCF. A detailed breakdown of costs is available in the annual report.

TAXATION

{tax_section}

SUBSCRIPTIONS AND REDEMPTIONS

Shares may be purchased or redeemed on any dealing day. The dealing day is every UK business day. The dealing cut-off time is {rng.choice(["12:00 noon", "2:00 pm", "4:00 pm"])} UK time. The price at which shares are issued or redeemed is the net asset value per share calculated at the valuation point, which is {rng.choice(["12:00 noon", "4:30 pm", "6:00 pm"])} UK time on each dealing day.

Settlement of subscriptions is due within {rng.choice([3, 4])} business days of the dealing day. Redemption proceeds will normally be paid within {rng.choice([3, 4, 5])} business days.

The ACD may limit or defer redemptions if total redemption requests on any dealing day exceed {rng.choice([5, 10])}% of the Fund's net asset value, in order to protect the interests of remaining shareholders.

GENERAL PROVISIONS

This prospectus is dated {date} and supersedes all previous versions. Copies of the latest annual and semi-annual reports, the instrument of incorporation, and the most recent published price of shares are available from the ACD. The Fund's register of shareholders is maintained by {rng.choice(["DST Financial Services", "International Financial Data Services", "SS&C Financial Services"])}. Complaints should be addressed to the ACD's compliance officer. If you are not satisfied with the response, you may refer the matter to the Financial Ombudsman Service."""


def generate_report(fund: dict, rng: random.Random,
                    report_type: str = "Annual") -> str:
    period = ("1 January" if report_type == "Annual"
              else rng.choice(["1 January", "1 July"]))
    yr = rng.randint(2022, 2025)
    end = ("31 December" if report_type == "Annual"
           else rng.choice(["30 June", "31 December"]))
    fund_size = round(rng.uniform(100, 20000), 1)
    returns_period = round(rng.uniform(-15, 30), 2)
    bench_returns = round(returns_period + rng.uniform(-5, 5), 2)
    num_holdings = rng.randint(40, 3000)

    holdings = _random_holdings(rng, 10)
    holdings_text = "\n".join(
        f"  {name}: £{round(rng.uniform(1, 500), 1)}m ({pct}%)"
        for name, pct in holdings
    )

    market_review = rng.choice([
        f"The period under review was marked by significant macroeconomic "
        f"uncertainty. Central banks maintained a data-dependent approach to "
        f"monetary policy, with the Bank of England holding base rates at "
        f"{rng.uniform(3.5, 5.5):.2f}%. Inflation remained above target for "
        f"much of the period, though the trajectory improved towards the end. "
        f"Equity markets in {fund['geography'].lower()} regions delivered "
        f"positive returns overall, supported by resilient corporate earnings "
        f"and easing financial conditions. Government bond yields traded in a "
        f"wide range, reflecting shifting interest rate expectations.",

        f"Global markets navigated a complex environment during the period, "
        f"characterised by geopolitical tensions, evolving monetary policy, "
        f"and sectoral rotation. {fund['geography']} equities outperformed "
        f"broader benchmarks, driven by strong earnings growth in the "
        f"technology and healthcare sectors. Fixed income markets experienced "
        f"volatility as investors reassessed the path of interest rate "
        f"normalisation across major economies.",

        f"The reporting period saw a continuation of the themes that "
        f"dominated markets in recent quarters. Artificial intelligence and "
        f"its potential economic impact remained a key driver of investor "
        f"sentiment. The {fund['geography']} market benefited from "
        f"relatively robust economic fundamentals and a constructive "
        f"corporate earnings season. Credit spreads tightened, reflecting "
        f"improved risk appetite among fixed income investors.",
    ])

    return f"""{report_type.upper()} REPORT AND FINANCIAL STATEMENTS

{fund['name']}
For the period {period} {yr} to {end} {yr}

AUTHORISED CORPORATE DIRECTOR'S REPORT

{fund['manager']} is pleased to present the {report_type.lower()} report for {fund['name']} for the period ended {end} {yr}. The Fund continues to be managed in accordance with its investment objective and investment policy as set out in the prospectus.

MARKET REVIEW

{market_review}

FUND PERFORMANCE

During the period, the Fund returned {returns_period:+.2f}% compared to the benchmark ({fund['benchmark']}) return of {bench_returns:+.2f}%. The Fund's net asset value at the period end was £{fund_size:,.1f} million, {rng.choice(["an increase", "a decrease", "broadly unchanged"])} from the prior period end. The number of holdings in the portfolio was {num_holdings} at the period end.

Key contributors to performance included {rng.choice(HOLDINGS)} and {rng.choice(HOLDINGS)}, while {rng.choice(HOLDINGS)} and {rng.choice(HOLDINGS)} detracted. The fund's {rng.choice(["overweight", "underweight"])} position in {rng.choice(SECTORS).lower()} {rng.choice(["added", "detracted"])} approximately {rng.uniform(0.1, 1.5):.2f}% relative to the benchmark.

The ongoing charges figure for the period was {fund['ocf']:.2f}%. No performance fee was charged during the period.

PORTFOLIO STATEMENT (EXTRACT)

The largest positions at the period end were:
{holdings_text}

Total value of investments: £{fund_size * 0.97:,.1f}m
Net other assets: £{fund_size * 0.03:,.1f}m
Net assets: £{fund_size:,.1f}m

STATEMENT OF TOTAL RETURN

For the period ended {end} {yr}:
  Income: £{round(rng.uniform(0.5, 50), 2):,.2f}m
  Expenses: £{round(rng.uniform(0.1, 10), 2):,.2f}m
  Net revenue before taxation: £{round(rng.uniform(0.3, 40), 2):,.2f}m
  Taxation: £{round(rng.uniform(0, 5), 2):,.2f}m
  Net revenue after taxation: £{round(rng.uniform(0.2, 35), 2):,.2f}m
  Capital gains / (losses): £{round(rng.uniform(-50, 100), 2):,.2f}m
  Total return: £{round(rng.uniform(-30, 120), 2):,.2f}m

OUTLOOK

{rng.choice([
    f"We remain cautiously optimistic on the outlook for {fund['geography'].lower()} markets. While valuations are not cheap by historical standards, corporate earnings growth and improving macro fundamentals provide support. We continue to focus on high-quality companies with strong balance sheets and sustainable competitive advantages.",
    f"The outlook for {fund['asset_class'].lower()} markets is finely balanced. On one hand, economic resilience and robust labour markets support risk assets. On the other, elevated interest rates and geopolitical risks warrant caution. The fund is positioned to benefit from a normalisation of monetary policy while maintaining appropriate diversification.",
    f"Looking ahead, we expect market volatility to persist as investors navigate the transition to a new interest rate regime. We see selective opportunities in {fund['geography'].lower()} markets, particularly in sectors benefiting from structural growth trends. The fund's diversified positioning should provide resilience across a range of economic scenarios.",
])}

This report was approved by the Board of {fund['manager']} on {_random_date(rng)} and signed on its behalf by the directors."""


def generate_instrument(fund: dict, rng: random.Random) -> str:
    reg_number = f"IC{rng.randint(100000, 999999)}"
    num_sub_funds = rng.randint(3, 25)
    num_share_classes = rng.randint(2, 8)
    date = _random_date(rng)

    return f"""INSTRUMENT OF INCORPORATION

{fund['name']}
An investment company with variable capital
Registered number: {reg_number}
Incorporated on: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{fund['launch_year']}

DEFINITIONS AND INTERPRETATION

In this Instrument, unless the context otherwise requires:
"ACD" means the authorised corporate director of the Company, being {fund['manager']} or such other person as may from time to time be appointed as authorised corporate director.
"Company" means {fund['name']}.
"Depositary" means {fund['depositary']} or such other depositary as may from time to time be appointed.
"FCA" means the Financial Conduct Authority or any successor body.
"FCA Rules" means the rules contained in the Collective Investment Schemes sourcebook (COLL) forming part of the FCA Handbook.
"Net Asset Value" means the value of the scheme property of the Company or a sub-fund less the liabilities of the Company or that sub-fund, as calculated in accordance with the FCA Rules.
"Shareholder" means a registered holder of shares in the Company.
"Sub-fund" means a separate pool of assets and liabilities within the Company, each constituting a separate fund for investment purposes.

CONSTITUTION OF THE COMPANY

The Company is an open-ended investment company with variable capital incorporated under the Open-Ended Investment Companies Regulations 2001. The Company is structured as an umbrella company comprising {num_sub_funds} sub-funds. Each sub-fund has its own investment objective and policy. The assets of each sub-fund belong exclusively to that sub-fund and shall not be used to discharge the liabilities of or claims against any other sub-fund.

The Company is a UCITS scheme for the purposes of the FCA Rules.

SHARE CLASSES

Each sub-fund may issue up to {num_share_classes} classes of shares. The rights attached to each share class may differ in respect of the currency of denomination, charging structure, minimum investment levels, and distribution policy. All shares of the same class within a sub-fund rank pari passu.

The share classes currently available include:
{chr(10).join(f'  - {sc}' for sc in rng.sample(SHARE_CLASSES, min(num_share_classes, len(SHARE_CLASSES))))}

The ACD may create additional share classes with the prior approval of the FCA and the Depositary. Shareholders will be notified of any new share classes in accordance with the FCA Rules.

POWERS AND DUTIES OF THE ACD

The ACD has overall responsibility for the management and administration of the Company in accordance with this Instrument, the prospectus, and the FCA Rules. The ACD's duties include but are not limited to:
  (a) calculating the Net Asset Value of each sub-fund and the price of shares;
  (b) dealing in shares, including the issue, redemption, and cancellation of shares;
  (c) managing the investments of each sub-fund in accordance with the investment objectives and policies;
  (d) maintaining proper books, records, and accounts;
  (e) preparing and distributing annual and interim reports;
  (f) ensuring compliance with applicable laws and regulations.

The ACD may delegate any of its functions to third parties, provided that it retains responsibility for the delegated activities and exercises appropriate oversight.

SHAREHOLDERS' MEETINGS

General meetings of shareholders may be convened by the ACD at any time and must be convened if required by the FCA Rules or requested by shareholders holding at least {rng.choice([10, 25])}% of the shares in issue. Shareholders are entitled to one vote per share held. Resolutions may be passed as ordinary resolutions (requiring a simple majority) or as extraordinary resolutions (requiring a 75% majority). The quorum for a general meeting is {rng.choice([2, 5, 10])} shareholders present in person or by proxy.

AMENDMENT AND TERMINATION

This Instrument may be amended by the ACD with the approval of the FCA and, where required by the FCA Rules, the prior sanction of an extraordinary resolution of shareholders. The Company or any sub-fund may be terminated by the ACD in accordance with the FCA Rules, subject to providing not less than {rng.choice([2, 3, 6])} months' written notice to shareholders.

This Instrument was adopted by written resolution on {date}.
Signed: Director, {fund['manager']}"""


# ---------------------------------------------------------------------------
# Source file and folder generation
# ---------------------------------------------------------------------------

CATEGORY_CONFIG = [
    ("KIID / KID",          0.50, generate_kiid,       None),
    ("Factsheet",           0.20, generate_factsheet,   None),
    ("Information Document", 0.10, generate_esg_disclosure, None),
    ("Prospectus",          0.10, generate_prospectus,  None),
    ("Annual Report",       0.025, generate_report,     "Annual"),
    ("Interim Report",      0.025, generate_report,     "Interim"),
    ("Legal / Instrument",  0.05, generate_instrument,  None),
]


def _make_source_path(fund: dict, category: str, doc_idx: int,
                       rng: random.Random) -> tuple[str, str]:
    slug = _slugify(fund["name"])
    idx_suffix = f"-{doc_idx:06d}"

    if category == "KIID / KID":
        filename = rng.choice([
            f"kiid-gb-en-{fund['isin'].lower()}{idx_suffix}.pdf",
            f"{fund['sedol'].lower()}{idx_suffix}-en.pdf",
        ])
    elif category == "Factsheet":
        filename = f"{slug}{idx_suffix}-factsheet.pdf"
    elif category == "Information Document":
        filename = rng.choice([
            f"{slug}{idx_suffix}-esg-factsheet.pdf",
            f"PPF_{slug}{idx_suffix}_Responsible_Investing_Information_Document.pdf",
        ])
    elif category == "Prospectus":
        filename = f"{slug}{idx_suffix}-prospectus-en.pdf"
    elif category in ("Annual Report", "Interim Report"):
        rtype = "annual" if "Annual" in category else "interim"
        filename = f"oeic-{rtype}-long-report{idx_suffix}.pdf"
    else:
        filename = f"instrument-of-incorporation{idx_suffix}.pdf"

    if rng.random() < 0.3:
        folder = f"{fund['manager']} - {fund['strategy']} {fund['equity_pct']}"
        source_file = f"{folder}/{filename}"
    else:
        folder = "Root"
        source_file = filename

    return source_file, folder


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def generate_all_chunks(funds: list[dict], num_documents: int,
                        rng: random.Random) -> list[dict]:
    cumulative_weights = []
    running = 0.0
    for cat_name, weight, gen_fn, extra in CATEGORY_CONFIG:
        running += weight
        cumulative_weights.append((running, cat_name, gen_fn, extra))

    all_chunks: list[dict] = []

    for doc_idx in tqdm(range(num_documents), desc="Generating documents"):
        fund = rng.choice(funds)
        r = rng.random()
        cat_name, gen_fn, extra = "", None, None
        for threshold, cn, gf, ex in cumulative_weights:
            if r < threshold:
                cat_name, gen_fn, extra = cn, gf, ex
                break

        if extra is not None:
            text = gen_fn(fund, rng, extra)
        else:
            text = gen_fn(fund, rng)

        text = clean_text(text)
        chunks = chunk_text(text)
        source_file, folder = _make_source_path(fund, cat_name, doc_idx, rng)

        for chunk_idx, chunk in enumerate(chunks):
            page_number = (chunk_idx // 2) + 1
            all_chunks.append({
                "id": _make_record_id(source_file, chunk_idx),
                "chunk_text": chunk,
                "fund_name": fund["name"],
                "category": cat_name,
                "source_file": source_file,
                "page_number": page_number,
                "folder": folder,
            })

    return all_chunks


def embed_batch_with_retry(pc: Pinecone, texts: list[str],
                           max_retries: int = 8) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            result = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            return [list(e["values"]) for e in result.data]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 60)
                print(f"\n  Embed error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


PARQUET_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("values", pa.list_(pa.float32())),
    ("chunk_text", pa.string()),
    ("fund_name", pa.string()),
    ("category", pa.string()),
    ("source_file", pa.string()),
    ("page_number", pa.int32()),
    ("folder", pa.string()),
])


def write_parquet(chunks: list[dict], vectors: list[list[float]],
                  output_path: Path):
    table = pa.table(
        {
            "id": [c["id"] for c in chunks],
            "values": [np.array(v, dtype=np.float32) for v in vectors],
            "chunk_text": [c["chunk_text"] for c in chunks],
            "fund_name": [c["fund_name"] for c in chunks],
            "category": [c["category"] for c in chunks],
            "source_file": [c["source_file"] for c in chunks],
            "page_number": [c["page_number"] for c in chunks],
            "folder": [c["folder"] for c in chunks],
        },
        schema=PARQUET_SCHEMA,
    )
    pq.write_table(table, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Dummy Data Generator")
    print("=" * 60)

    rng = random.Random(SEED)

    # Phase 1: fund pool
    print("\n[1/3] Generating fund pool...")
    funds = generate_fund_pool(500, rng)
    print(f"  Created {len(funds)} unique fund identities.")

    # Phase 2: documents -> chunks
    print(f"\n[2/3] Generating {NUM_DOCUMENTS:,} documents and chunking...")
    all_chunks = generate_all_chunks(funds, NUM_DOCUMENTS, rng)
    print(f"  Total chunks: {len(all_chunks):,}")

    # Phase 3: embed + write parquet
    print("\n[3/3] Embedding chunks and writing parquet files...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_files = (len(all_chunks) + PARQUET_BATCH_SIZE - 1) // PARQUET_BATCH_SIZE
    print(f"  Will produce {num_files} parquet file(s), ~{PARQUET_BATCH_SIZE:,} rows each.")

    for file_idx in range(num_files):
        output_path = OUTPUT_DIR / f"fund_data_{file_idx:03d}.parquet"
        if output_path.exists():
            print(f"  Skipping {output_path.name} (already exists)")
            continue

        start = file_idx * PARQUET_BATCH_SIZE
        end = min(start + PARQUET_BATCH_SIZE, len(all_chunks))
        batch_chunks = all_chunks[start:end]

        texts = [c["chunk_text"] for c in batch_chunks]
        vectors: list[list[float]] = []

        embed_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        pbar = tqdm(total=embed_batches,
                    desc=f"  Embedding → {output_path.name}",
                    unit="batch")

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            vecs = embed_batch_with_retry(pc, batch)
            vectors.extend(vecs)
            pbar.update(1)

        pbar.close()

        write_parquet(batch_chunks, vectors, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Wrote {output_path.name}: {len(batch_chunks):,} rows, {size_mb:.1f} MB")

    total_size = sum(
        f.stat().st_size for f in OUTPUT_DIR.glob("fund_data_*.parquet")
    ) / (1024 * 1024)
    print(f"\nDone! {len(all_chunks):,} vectors across {num_files} files ({total_size:.1f} MB total)")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
