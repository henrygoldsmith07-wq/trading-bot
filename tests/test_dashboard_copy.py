"""Guardrails on `public/index.html`.

The dashboard is a measurement instrument, not a funnel. These tests are the
difference between that being a comment in a code review and being enforced:
they fail the build the moment the page starts selling again.

Invariants, each tied to a product decision:

  1. THREE EVIDENCE LABELS ONLY — research / out-of-sample / forward.
     The word "live" does not appear anywhere in the document.
  2. NO AFFORDANCES — no inputs, buttons or forms. Nothing on the page lets a
     visitor deposit money, paste exchange keys, or start trading.
  3. NO BROWSER-SIDE MARKET CALLS — the page reads exactly one endpoint, the
     same-origin /api/summary. No exchange API from the client.
  4. HERO IS FORWARD DAYS + VERDICT — not backtest CAGR.
  5. ONE RECOMMENDED READING — the deflated-Sharpe caveat.
  6. BIG RED EDUCATION BANNER — always in the document, never conditional.
  7. DEGRADE TO QUIET — bad or thin forward evidence must reduce the page's
     volume, never add strategies.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[1] / "public" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    if not HTML.exists():
        pytest.skip("dashboard not present")
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def flat(html) -> str:
    """Whitespace-collapsed source. Use this for PROSE assertions so that a
    re-wrapped paragraph doesn't read as a policy change."""
    return re.sub(r"\s+", " ", html)


@pytest.fixture(scope="module")
def hero(html) -> str:
    """Just the hero block — what a visitor sees before scrolling."""
    return html[html.index('class="hero"'):html.index('id="forward-detail"')]


# ---------------------------------------------------------------------------
# 1. evidence taxonomy
# ---------------------------------------------------------------------------

def test_the_word_live_never_appears(html):
    """A fourth evidence class would imply the system trades real money."""
    hits = [m.group(0) for m in re.finditer(r"\blive\b", html, flags=re.IGNORECASE)]
    assert hits == [], f"forbidden evidence label 'live' appears {len(hits)} time(s)"


def test_exactly_three_evidence_labels(html):
    used = set(re.findall(r"pill pill-([a-z-]+)", html))
    expected = {"forward", "oos", "research"}
    assert used <= expected, f"unexpected evidence label(s): {sorted(used - expected)}"
    assert used == expected, f"missing evidence label(s): {sorted(expected - used)}"


def test_three_labels_are_defined_in_a_legend(flat):
    """The taxonomy is explained on the page, not just implied by colour."""
    assert "produced after the freeze" in flat
    assert "walk-forward folds, never trained on" in flat
    assert "development work, subject to selection effects" in flat


# ---------------------------------------------------------------------------
# 2. no affordances
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["<input", "<button", "<form", "<select", "<textarea"])
def test_no_interactive_controls(html, tag):
    assert tag not in html.lower(), f"{tag} is a deposit/keys/start-trading affordance"


@pytest.mark.parametrize("phrase", [
    "no money is deposited",
    "no exchange keys are accepted",
    "no button on this page that starts trading",
    "not financial advice",
])
def test_the_absence_of_affordances_is_stated(flat, phrase):
    assert phrase in flat.lower()


@pytest.mark.parametrize("phrase", ["start trading now", "open an account", "connect your exchange",
                                    "enter your api key", "deposit funds", "buy now"])
def test_no_calls_to_action(flat, phrase):
    assert phrase not in flat.lower()


# ---------------------------------------------------------------------------
# 3. no browser-side market data
# ---------------------------------------------------------------------------

def test_only_same_origin_fetch(html):
    targets = re.findall(r"""fetch\(\s*["']([^"']+)["']""", html)
    assert targets, "the page must read the forward record from the API"
    for t in targets:
        assert t.startswith("/"), f"browser-side call to {t!r} — external calls leak keys and are blocked in CI"


def test_no_exchange_hosts(html):
    lowered = html.lower()
    for host in ("binance", "coinbase", "kraken", "bybit", "okx"):
        assert host not in lowered, f"{host} must not be contacted from the browser"


# ---------------------------------------------------------------------------
# 4. hero: forward days + verdict
# ---------------------------------------------------------------------------

def test_hero_leads_with_forward_days_and_verdict(html):
    assert 'id="f-days"' in html
    assert 'id="f-verdict"' in html
    assert "Forward paper days since freeze" in html
    assert "Current verdict" in html


def test_hero_does_not_mention_backtest_cagr(hero):
    """CAGR is a research figure. If it reaches the hero, the page has quietly
    gone back to selling the backtest."""
    assert "cagr" not in hero.lower(), "the hero must not carry a research CAGR"
    assert "25.7" not in hero, "the research headline number must not reach the hero"


def test_hero_number_is_the_largest_type_on_the_page(html):
    sizes = [int(s) for s in re.findall(r"font-size:\s*(\d+)px", html)]
    hero = int(re.search(r"\.hero-number\s*\{[^}]*font-size:\s*(\d+)px", html).group(1))
    assert hero == max(sizes), "nothing may out-shout the forward day count"


def test_return_figures_are_demoted_below_the_hero(hero):
    """Forward return / Sharpe / benchmark exist, but only as detail revealed
    behind the volume gate — never in the hero block."""
    for forbidden in ("Forward return", "Forward Sharpe", "Max drawdown"):
        assert forbidden not in hero, f"{forbidden!r} must not sit in the hero"


# ---------------------------------------------------------------------------
# 5. one recommended reading
# ---------------------------------------------------------------------------

def test_exactly_one_recommended_reading(html):
    assert len(re.findall(r"[Rr]ecommended reading", html)) == 1
    links = re.findall(r'href="(https?://[^"]+)"', html)
    assert len(links) == 1, f"one link means one reading; found {len(links)}"


def test_recommended_reading_is_the_dsr_caveat(flat):
    link = re.search(r'href="(https?://[^"]+)"', flat).group(1)
    assert "statistical-validation" in link, "the recommended reading must be the DSR caveat"
    # The flattering headline is the thing the recommendation exists to defuse,
    # so the caveat has to name the numbers, not just gesture at them.
    assert "0.138" in flat, "the selected stream's deflated Sharpe must be stated"
    assert "0.961" in flat, "the un-searched rule's deflated Sharpe must be stated"
    assert "does not clear the conventional 0.95 bar" in flat
    assert "25.7" in flat, "the caveat must name the headline number it defuses"


def test_the_flattering_cagr_is_labelled_as_research(flat):
    assert "CAGR (research)" in flat
    assert "was produced by a search over ~85 candidates" in flat


# ---------------------------------------------------------------------------
# 6. education banner
# ---------------------------------------------------------------------------

def test_big_red_education_banner_is_unconditional(html):
    """The banner is education, not an error message: it is styled red, and it
    renders unconditionally. It must never become a dismissible afterthought."""
    tag = re.search(r'<div class="edu"[^>]*>', html)
    assert tag, "the red education banner is missing"
    assert "hidden" not in tag.group(0), "the banner must not ship hidden"

    css = re.search(r"\.edu\s*\{([^}]*)\}", html)
    assert css, "the banner has no styling"
    body = css.group(1)
    assert "248,81,73" in body, "the banner must be red"
    assert re.search(r"border:\s*2px solid", body), "the banner must have a heavy border"
    assert "display: none" not in body, "the banner must never be display:none"


def test_banner_mentions_all_three_prohibitions(flat):
    banner = flat[flat.index('class="edu"'):flat.index("<!-- ===================== HERO")]
    lowered = banner.lower()
    for phrase in ("no money is deposited", "no exchange keys are accepted", "starts trading"):
        assert phrase in lowered, f"banner must state: {phrase}"


# ---------------------------------------------------------------------------
# 7. degrade to quiet
# ---------------------------------------------------------------------------

def test_volume_gate_exists(html):
    """Three volumes: silent, quiet, full. Thin or broken forward evidence
    moves the page DOWN a volume, never sideways into a new strategy."""
    assert "const SILENT = 0, QUIET = 1, FULL = 2;" in html
    assert "function assess(" in html
    assert "function applyVolume(" in html


def test_thin_evidence_withholds_return_figures(html):
    assert "n < 30" in html
    assert "forward-detail\").hidden = !rich" in html
    assert "forward-curve-panel\").hidden = !rich" in html


def test_degradation_is_explicit_about_not_adding_strategies(flat):
    """The growth reflex is to ship a new strategy when results disappoint.
    This page must say out loud that it does the opposite."""
    assert "showing less, not more" in flat
    assert "no new strategy offered in response" in flat


def test_outage_heavy_evidence_is_downgraded(html):
    assert "outages / n > 0.2" in html


def test_broken_seal_silences_the_page(html):
    assert 'f.code_verified === false' in html
    assert "(f.parameter_changes | 0) !== 0" in html


def test_research_section_is_folded_by_default_when_quiet(html):
    assert 'details class="research-fold"' in html
    assert 'research-fold").open = false' in html
