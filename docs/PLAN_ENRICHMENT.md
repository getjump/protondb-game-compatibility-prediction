# Enrichment — аугментация метаданными игр

Часть preprocessing pipeline. Отдельный шаг, который запускается **до** LLM-экстракции.

## Цель

Обогатить таблицу `game_metadata` данными из внешних источников: движок, graphics API, anti-cheat, developer/publisher. Результат используется:
1. **В LLM промпте** — контекст для более точной экстракции
2. **В ML features** — game-level признаки (см. `PLAN_ML.md`)
3. **В API ответах** — обогащает рекомендации метаданными

## Схема БД

Таблица `game_metadata` — см. `PLAN.md` (единый source of truth для схемы).

## Источники

| Источник | Данные | Auth | Rate limit | Метод |
|---|---|---|---|---|
| **Steam Store API** | developer, publisher, genre, categories, release_date, platforms | Нет | ~200 req/5 min | `GET appdetails?appids=X` |
| **Steam Deck Verified** | deck_status (verified/playable/unsupported), test details | Нет | ~200 req/5 min | `GET saleaction/ajaxgetdeckappcompatibilityreport?nAppID=X` |
| **ProtonDB Summary** | community tier (platinum..borked), score, confidence | Нет | ~1 req/sec | `GET api/v1/reports/summaries/X.json` |
| **PCGamingWiki Cargo API** | engine, graphics API, anti-cheat, DRM | Нет | ~1 req/sec | `GET api.php?action=cargoquery` по `Steam_AppID` |
| **AreWeAntiCheatYet** | anti-cheat тип, Linux-статус | Нет | Один файл | `games.json` с GitHub |

## Приоритизация

Не все 50-60K игр одинаково важны. Стратегия:

1. **Tier 1** (~5K игр): app_ids с 10+ отчётов — обогащаем первыми, покрывают ~90% отчётов
2. **Tier 2** (~15K игр): app_ids с 2-9 отчётов — следующий приоритет
3. **Tier 3** (~30K игр): app_ids с 1 отчётом — по возможности

```sql
-- Получить app_ids по приоритету
SELECT app_id, COUNT(*) as cnt FROM reports
WHERE app_id NOT IN (SELECT app_id FROM game_metadata)
GROUP BY app_id ORDER BY cnt DESC;
```

## Batch-запросы

### Steam Store API

Поддерживает **один appid за запрос** (нет batch endpoint). Но при ~1 req/sec:
- Tier 1 (5K): ~1.5 часа
- Tier 1+2 (20K): ~6 часов

### PCGamingWiki Cargo API

Поддерживает batch через OR в WHERE:

```python
# Batch по 10 app_ids за запрос
batch_ids = app_ids[:10]
where_clause = " OR ".join(f'Infobox_game.Steam_AppID HOLDS "{aid}"' for aid in batch_ids)
params = {
    "action": "cargoquery",
    "tables": "Infobox_game,API,Middleware,Availability",
    "join_on": "Infobox_game._pageName=API._pageName,"
              "Infobox_game._pageName=Middleware._pageName,"
              "Infobox_game._pageName=Availability._pageName",
    "fields": "Infobox_game.Steam_AppID,Infobox_game.Engines,API.Direct3D_versions,"
              "API.Vulkan_versions,API.OpenGL_versions,Middleware.Anticheat,"
              "Availability.Uses_DRM",
    "where": where_clause,
    "limit": "50",
    "format": "json",
}
```

Это ускоряет PCGamingWiki в 10x: 5K игр за ~10 минут вместо ~1.5 часа.

## Архитектура

```
reports table → список уникальных app_ids (приоритизированный)
    │
    ▼
┌───┼───────────────┼────────────────┐
▼   ▼               ▼                ▼
[Steam Store    [PCGamingWiki]  [AreWeAntiCheatYet]  [ProtonDB Summary]
 + Deck Verified]
~200 req/5min   batch по 10     один fetch            ~1 req/sec
    │               │               │                     │
    └───────────────┼───────────────┼─────────────────────┘
                    ▼
              [Merge & Store]
                    │
                    ▼
            game_metadata table
```

Steam, PCGamingWiki и ProtonDB параллелятся (разные домены).

## Логика по источникам

### Steam Store API

```python
def fetch_steam(app_id: int) -> SteamData | None:
    resp = httpx.get(f"https://store.steampowered.com/api/appdetails?appids={app_id}")
    data = resp.json()[str(app_id)]
    if not data["success"]:
        return None
    d = data["data"]
    return SteamData(
        developer=d.get("developers", [None])[0],
        publisher=d.get("publishers", [None])[0],
        genres=[g["description"] for g in d.get("genres", [])],
        categories=[c["description"] for c in d.get("categories", [])],
        release_date=d.get("release_date", {}).get("date"),  # НЕ ISO-8601! Формат "9 Dec, 2020" — нужен парсинг
        has_linux_native=d.get("platforms", {}).get("linux", False),
    )
```

> **Findings**: `release_date.date` возвращается в человекочитаемом формате ("9 Dec, 2020"), НЕ ISO-8601 — нужен `dateutil.parser` или ручной парсинг.

### PCGamingWiki Cargo API

```python
def fetch_pcgw_batch(app_ids: list[int]) -> dict[int, PCGWData]:
    where = " OR ".join(f'Infobox_game.Steam_AppID HOLDS "{aid}"' for aid in app_ids)
    params = { ... , "where": where, "limit": "50" }
    resp = httpx.get("https://www.pcgamingwiki.com/w/api.php", params=params)
    results = {}
    for row in resp.json().get("cargoquery", []):
        title = row["title"]
        # Steam_AppID может быть comma-separated ("1091500,1495710,2060310") — нужен split
        raw_id = title.get("Steam_AppID", "")
        for sid in raw_id.split(","):
            sid = sid.strip()
            if sid.isdigit():
                steam_id = int(sid)
                # Engines приходят с префиксом "Engine:" (e.g. "Engine:REDengine") — strip
                engine = (title.get("Engines") or "").removeprefix("Engine:") or None
                # Uses_DRM: comma-separated ("Denuvo Anti-Tamper,Steam,VMProtect")
                drm_raw = title.get("Uses DRM") or ""  # пробел в имени поля (Cargo API)
                drm_list = [d.strip() for d in drm_raw.split(",") if d.strip() and d.strip() != "DRM-free"]
                results.setdefault(steam_id, PCGWData(
                    engine=engine,
                    graphics_apis=parse_graphics_apis(title),  # Vulkan возвращает "true" (строка), не версию
                    anticheat=title.get("Anticheat") or None,
                    drm=drm_list or None,  # ["Denuvo Anti-Tamper", "Steam"] или None
                ))
    # JOIN может давать duplicate rows (CS2: Source 2 + Source) — берём первую (primary) запись
    # setdefault: первая строка от Cargo API — наиболее релевантная
    return results
```

> **Findings**:
> - `Steam_AppID` может быть comma-separated — нужен split по `,`
> - JOIN с таблицей API/Middleware/Availability даёт дубликаты строк (CS2 вернул 2 строки: Source 2 + Source) — дедупликация обязательна
> - Engines приходят с префиксом `"Engine:"` (e.g. `"Engine:REDengine"`) — нужен strip
> - `Vulkan_versions` возвращает `"true"` (строка), а не номер версии — нужна специальная обработка
> - `Uses_DRM` — comma-separated: `"Denuvo Anti-Tamper,Steam,VMProtect"`. Значение `"DRM-free"` фильтруем. DRM данные в таблице `Availability`, не `Middleware`

### AreWeAntiCheatYet

```python
def load_awacy() -> dict[int, AWACYData]:
    resp = httpx.get(
        "https://raw.githubusercontent.com/AreWeAntiCheatYet/AreWeAntiCheatYet/HEAD/games.json"
    )
    index = {}
    for game in resp.json():
        # storeIds — dict (не list!): {"steam": "730"} или {"epic": {...}}
        steam_id = game.get("storeIds", {}).get("steam")
        if steam_id:
            index[int(steam_id)] = AWACYData(
                anticheats=game.get("anticheats", []),
                status=game.get("status"),
            )
    return index
```

> **Findings**:
> - `storeIds` — dict, не list: `{"steam": "730"}` или `{"epic": {...}}`
> - 1166 записей всего: 681 с Steam ID, 485 без
> - Status distribution: Broken 643, Running 275, Supported 194, Denied 52, Planned 2
> - Файл ~450KB, CDN GitHub — можно кешировать локально

### Steam Deck Verified

```python
def fetch_deck_status(app_id: int) -> DeckData | None:
    resp = httpx.get(
        f"https://store.steampowered.com/saleaction/ajaxgetdeckappcompatibilityreport?nAppID={app_id}"
    )
    data = resp.json()
    if not data.get("success") or not data.get("results"):
        return None
    results = data["results"]
    return DeckData(
        status=results.get("resolved_category", 0),  # 0=unknown, 1=unsupported, 2=playable, 3=verified
        tests=results.get("resolved_items", []),
    )
```

> Можно запрашивать параллельно со Steam Store API (тот же домен, общий rate limit ~200 req/5 min).

### ProtonDB Summary

```python
def fetch_protondb_summary(app_id: int) -> ProtonDBData | None:
    resp = httpx.get(f"https://www.protondb.com/api/v1/reports/summaries/{app_id}.json")
    if resp.status_code == 404:
        return None
    data = resp.json()
    return ProtonDBData(
        tier=data.get("tier"),              # platinum/gold/silver/bronze/borked
        score=data.get("score"),            # 0..1
        confidence=data.get("confidence"),  # strong/good/weak
        trending_tier=data.get("trendingTier"),  # → protondb_trending column
    )
```

> Rate limit ~1 req/sec. Для Tier 1 (5K игр) — ~1.5 часа. Можно параллелить с PCGamingWiki.

## Оценка времени

| Scope | Игр | Steam API + Deck | PCGamingWiki | ProtonDB Summary | Итого |
|---|---|---|---|---|---|
| Tier 1 (10+ отчётов) | ~5K | ~1.5ч | ~10мин (batch) | ~1.5ч | **~2 часа** (параллельно) |
| Tier 1+2 (2+ отчётов) | ~20K | ~6ч | ~30мин | ~6ч | **~7 часов** (параллельно) |
| Все | ~50K | ~14ч | ~1.5ч | ~14ч | **~14 часов** (параллельно) |

Steam и ProtonDB — bottleneck (по ~1 req/sec каждый), но параллелятся. Рекомендация: начать с Tier 1 — покроет 90% отчётов за ~2 часа.

## Структура кода

```
protondb_settings/preprocessing/enrichment/
├── __init__.py
├── main.py            # enrichment logic
├── sources/
│   ├── __init__.py
│   ├── steam.py       # Steam Store API client + Deck Verified (same domain)
│   ├── protondb.py    # ProtonDB Summary API client
│   ├── pcgamingwiki.py # PCGamingWiki Cargo API client (batch)
│   └── anticheat.py   # AreWeAntiCheatYet loader
├── merger.py          # объединение данных из всех источников
└── models.py          # pydantic модели
# Использует ../pipeline.py (PipelineStep), ../store.py (UPSERT)
# Конфигурация — в protondb_settings/config.py
```

## Запуск

```bash
# Tier 1 только (быстро, покрывает 90%). Resume автоматический.
protondb-settings preprocess run --step enrichment --min-reports 10

# Все
protondb-settings preprocess run --step enrichment

# Только конкретный источник (steam | pcgamingwiki | protondb | anticheat)
protondb-settings preprocess run --step enrichment --source steam

# Обновить устаревшие (>30 дней)
protondb-settings preprocess run --step enrichment --refresh-older-than 30d

# Перезапуск с нуля
protondb-settings preprocess run --step enrichment --force
```

## Durability & Resume

Enrichment автоматически обрабатывает только непокрытые app_ids (см. `preprocessing/PLAN.md` → Durability).

- **Implicit checkpoint**: `WHERE app_id NOT IN (SELECT app_id FROM game_metadata)` — пропуск уже обогащённых
- **Batch commits**: каждые 50-100 app_ids
- **UPSERT**: `INSERT OR REPLACE INTO game_metadata` — повторный запуск безопасен
- **`pipeline_runs`**: трекинг прогресса, обнаружение прерванных runs
- **`--force`**: удаляет `game_metadata` и начинает сначала
- **Stale refresh**: `--refresh-older-than 30d` добавляет к обработке app_ids с `enriched_at < datetime('now', '-30 days')`. Логика:
  ```sql
  -- Непокрытые + устаревшие
  SELECT g.app_id FROM games g
  LEFT JOIN game_metadata gm ON g.app_id = gm.app_id
  WHERE gm.app_id IS NULL
     OR gm.enriched_at < datetime('now', '-30 days')
  ORDER BY (SELECT COUNT(*) FROM reports r WHERE r.app_id = g.app_id) DESC
  ```

### Проверка обновлений AreWeAntiCheatYet

```python
# HTTP HEAD с ETag — без скачивания (~50ms)
resp = httpx.head(AWACY_URL, headers={"If-None-Match": meta["awacy_etag"]})
if resp.status_code == 304:
    print("AreWeAntiCheatYet: up to date")
else:
    print("AreWeAntiCheatYet: updated, re-fetch needed")
    # meta.awacy_etag = resp.headers["ETag"]
```

## Обработка ошибок

- Steam API `success: false` → пропускаем, логируем (игра удалена/скрыта)
- PCGamingWiki не нашёл → null-поля (~30% игр не покрыты)
- AreWeAntiCheatYet нет записи → `anticheat_status = null`
- Rate limit / timeout → retry с backoff
- Все ошибки не-фатальные, процесс продолжается
