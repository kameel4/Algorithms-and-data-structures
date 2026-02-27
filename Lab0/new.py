from __future__ import annotations
import re
import requests
from typing import Iterable, List, Dict, Set, Tuple

ISS_BASE = "https://iss.moex.com/iss"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ticker-builder/1.0"})

def parse_tickers(raw: str) -> List[str]:
    """
    Парсит тикеры из текста: разделители пробел/запятая/точка-с-запятой/переносы.
    Пример входа: "SBER, GAZP\nLKOH; rosn"
    """
    tokens = re.split(r"[,\s;]+", raw.strip())
    out = []
    for t in tokens:
        if not t:
            continue
        t = t.strip().upper()
        # фильтр от мусора (оставляем A-Z0-9 и ._- на всякий)
        if re.fullmatch(r"[A-Z0-9._-]{2,15}", t):
            out.append(t)
    # сохранить порядок, убрать дубли
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def get_index_members(index_id: str) -> List[str]:
    """
    Тикеры из индекса MOEX через ISS analytics/<INDEX>.json
    Пример эндпоинта для IMOEX: .../statistics/engines/stock/markets/index/analytics/IMOEX.json :contentReference[oaicite:2]{index=2}
    """
    url = f"{ISS_BASE}/statistics/engines/stock/markets/index/analytics/{index_id}.json"
    params = {"iss.meta": "off", "limit": 9999}
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()

    # Обычно приходит таблица analytics: { columns: [...], data: [...] }
    analytics = js.get("analytics", {})
    cols = analytics.get("columns", [])
    data = analytics.get("data", [])

    # Пытаемся найти колонку с тикером
    # На практике часто это SECID (стандартное поле тикера на MOEX).
    # Если не нашли — возьмём первый столбец как fallback.
    try:
        secid_idx = cols.index("SECID")
    except ValueError:
        secid_idx = 0

    tickers = []
    for row in data:
        if not row or secid_idx >= len(row):
            continue
        t = str(row[secid_idx]).strip().upper()
        if re.fullmatch(r"[A-Z0-9._-]{2,15}", t):
            tickers.append(t)

    # убрать дубли, сохранить порядок
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def build_universe(
    base: Iterable[str],
    target_size: int = 50,
    use_indices: Tuple[str, ...] = ("MOEXBC", "IMOEX", "MCXSM"),
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    Собирает итоговый список target_size, приоритет:
    1) твой base
    2) MOEXBC (blue chips)
    3) IMOEX (шире, “первый эшелон” по ликвидности)
    4) MCXSM (SMID, “второй эшелон”)
    """
    base_list = [t.upper() for t in base]
    universe: List[str] = []
    seen: Set[str] = set()

    def add_many(items: Iterable[str]):
        nonlocal universe
        for t in items:
            if t not in seen:
                seen.add(t)
                universe.append(t)
                if len(universe) >= target_size:
                    return

    add_many(base_list)

    membership: Dict[str, Set[str]] = {}
    for idx in use_indices:
        members = get_index_members(idx)
        membership[idx] = set(members)
        add_many(members)
        if len(universe) >= target_size:
            break

    return universe[:target_size], membership

def label_tiers(tickers: List[str], membership: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
    """
    Простая маркировка:
    - Tier 1: в MOEXBC или IMOEX
    - Tier 2: в MCXSM (если не Tier 1)
    - Other: иначе
    """
    moexbc = membership.get("MOEXBC", set())
    imoex = membership.get("IMOEX", set())
    mcxsm = membership.get("MCXSM", set())

    out = []
    for t in tickers:
        if t in moexbc or t in imoex:
            tier = "Tier1 (MOEXBC/IMOEX)"
        elif t in mcxsm:
            tier = "Tier2 (MCXSM)"
        else:
            tier = "Other/unknown"
        out.append((t, tier))
    return out

def to_quizlet_import(rows: List[Tuple[str, str]]) -> str:
    """
    Формат для Quizlet Import: TERM<TAB>DEFINITION
    """
    return "\n".join(f"{term}\t{definition}" for term, definition in rows)

if __name__ == "__main__":
    # 1) Вставь сюда свой фиксированный список тикеров (как текст)
    raw = """
    "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "PLZL", "TATN",
    "SIBN", "MGNT", "CHMF", "VTBR", "ALRS", "YDEX"
    """
    base = parse_tickers(raw)

    # 2) Собираем 30-50 тикеров (поменяй target_size)
    universe, membership = build_universe(base, target_size=50)

    # 3) Маркируем “эшелоны”
    labeled = label_tiers(universe, membership)

    # 4) Печать
    for t, tier in labeled:
        print(f"t  {tier}")