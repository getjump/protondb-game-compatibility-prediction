# ML Pipeline — фаза 6: анализ, новые фичи, новые источники

> По результатам глубокого анализа каскадной модели (scripts/analyze_cascade_model.py)

## Текущее состояние

**F1 macro: 0.731** | Accuracy: 78.8% | 149 features | 348K reports

| Stage | Task | AUC | LogLoss | Bottleneck |
|-------|------|-----|---------|------------|
| Stage 1 | borked vs works | 0.975 | 0.124 | |
| **Stage 2** | **tinkering vs works_oob** | **0.831** | **0.394** | **3.2x хуже** |

### Распределение ошибок

| Тип ошибки | Count | % от ошибок | Причина |
|---|---|---|---|
| works_oob → tinkering | 5993 | 40.6% | GE-Proton bias, label noise |
| tinkering → works_oob | 5516 | 37.3% | «хорошая игра + хорошее железо» shortcut |
| borked → tinkering | 1584 | 10.7% | текст о попытках починки путает модель |
| tinkering → borked | 1100 | 7.4% | Stage 1 errors |
| borked → works_oob | 324 | 2.2% | |
| works_oob → borked | 254 | 1.7% | |

**78% ошибок — путаница tinkering↔works_oob (Stage 2)**

### Ключевые findings

1. **Stage 2 uncertainty zone**: 24.3% отчётов в зоне P(oob) ∈ [0.35, 0.65], accuracy там 56.3% (≈random)
2. **`variant` = shortcut**: 963K gain в Stage 2. Модель учит «GE = tinkering», что bias для GE-default пользователей
3. **Text embeddings** доминируют Stage 1 (text_emb_2: 2.2M gain), но почти бесполезны в Stage 2
4. **Temporal stability**: F1 ≈ 0.72-0.76 стабильно в течение года, drift отсутствует
5. **Calibration**: borked ECE=0.032 (хорошо), tinkering ECE=0.105 (overconfident)

---

## A. Неиспользованные данные из reports (zero-cost)

Поля из reports, которые **уже есть в БД**, но не используются как per-report фичи.

### A1. Structured fault booleans (87% покрытие)

8 булевых полей `audio_faults..significant_bugs` (yes/no) — **прямой сигнал** для Stage 1.

| Поле | yes count | Корреляция |
|------|-----------|------------|
| `stability_faults` | 29K | crash → borked |
| `performance_faults` | 41K | lag → borked/tinkering |
| `graphical_faults` | 23K | artifacts → tinkering |
| `input_faults` | 19K | controller issues → tinkering |
| `significant_bugs` | 20K | bugs → borked |
| `audio_faults` | 15.5K | audio → minor |
| `windowing_faults` | 22K | resolution issues → tinkering |
| `save_game_faults` | 4.3K | saves broken → tinkering |

**Сейчас** используется только `fault_notes_count` (кол-во заполненных notes_*_faults полей). Сами булевы — нет.

**Ожидание**: +0.002–0.008 F1. Прямые сигналы для Stage 1 (stability_faults + significant_bugs → borked).

### A2. Followup detail JSON (15-23K записей)

`followup_graphical_faults_json` → `{"heavyArtifacts": true, "missingTextures": true}`
`followup_anticheat_json` → `{"easyAntiCheat": true}`

Специфичные проблемы. Низкое покрытие (5-7%), но **high signal**: `heavyArtifacts` → borked, `easyAntiCheat` → borked.

### A3. Report-level behavioral fields

| Поле | Покрытие | Тип | Сигнал |
|------|----------|-----|--------|
| `tried_oob` | 101K (29%) | yes/no | **прямой**: tried=yes + verdict_oob=no → confirmed tinkering |
| `tinker_override` | 114K (33%) | yes/no | пользователь явно указал «пришлось тинкерить» |
| `duration` | 85K (24%) | categorical | moreThanTenHours → вероятно works; lessThanFifteenMinutes → вероятно borked |
| `installs` | ? | int | сколько раз ставил → больше = тинкеринг |
| `opens`/`starts_play` | ? | int | воронка: opens >> starts_play → проблема запуска |

**⚠️ `tried_oob` и `tinker_override` — потенциальный label leak!** Эти поля заполняются одновременно с verdict. Нужен эксперимент: если модель их использует и F1 взлетает — это leak. Если прирост скромный — полезный сигнал.

**`duration`**: безопасная фича. `lessThanFifteenMinutes` сильно коррелирует с borked (не смог запустить). `moreThanTenHours` → works.

### A4. Per-report flag/customization fields (1-8K, sparse)

`flag_use_wine_d3d11`, `flag_disable_esync`, `cust_protontricks` и т.д. — уже используются **агрегированно** per-game (`pct_needs_protontricks`, `pct_uses_wine_d3d11`), но **не per-report**.

Per-report binary flags → прямой сигнал для Stage 2: «этот конкретный пользователь использовал protontricks» = tinkering, не oob.

**Ожидание**: +0.005–0.015 F1 в Stage 2. Это **самый перспективный вектор** — напрямую адресует bottleneck.

### A5. `is_impacted_by_anticheat` (32K, 9.2%)

Поле уже есть. `yes` = 2447 отчётов. Сильный borked/tinkering сигнал. Не per-game (game_metadata.anticheat), а per-report — пользователь **сам** указал что античит мешает.

---

## B. Улучшение существующих фичей

### B1. Variant debiasing (Stage 2)

**Проблема**: `variant` = 963K gain в Stage 2. GE-Proton пользователи, ставящие его по-умолчанию (flatpak, distro default), получают bias к tinkering.

**Решение**: заменить raw `variant` на:
- `is_custom_proton`: 0/1 (GE или custom, а не «какой именно»)
- `variant_is_expected_oob`: 1 если variant = official/experimental/native (те, что идут с Steam)
- Убрать `variant` из Stage 2, оставить derived

### B2. Recalibration Stage 2

ECE tinkering = 0.105, overconfident. Варианты:
- Temperature scaling вместо isotonic (меньше overfitting на calibration set)
- Platt scaling (логистическая регрессия на logits)
- Focal loss в Stage 2 LightGBM (фокус на hard examples)

### B3. Threshold tuning per-variant

`official`: 90% accuracy, но F1=0.67 (bias к tinkering, 80.7% класс)
`ge`/`experimental`: 65% accuracy, но F1=0.68 (более сбалансированные)

Адаптивные пороги Stage 2: variant-specific `P(oob)` threshold.

---

## C. LLM-based text features (из PLAN_ML_5_LLM.md)

Самый высокий ожидаемый ROI — адресует именно Stage 2 bottleneck.

### C1. Structured text extraction

| Фича | Источник | Задача для LLM | Ожидаемый сигнал |
|------|----------|----------------|------------------|
| `effort_level` | notes + customizations | none/low/medium/high | **none → oob, high → tinkering** |
| `playability_score` | все notes | 0.0–1.0 | Stage 2 прямой сигнал |
| `issue_severity` | notes + faults | none/minor/major/critical | Stage 1 сигнал |
| `customization_complexity` | customizations | none/simple/complex | complex → tinkering |
| `text_sentiment` | concluding_notes | 0.0–1.0 | auxiliary signal |

**Ожидание**: +0.010–0.025 F1, **в основном в Stage 2**

**Cost**: ~106K записей с concluding_notes × ~$0.001/запись = ~$100 через batch API

### C2. Verdict relabeling

Error analysis показывает:
- **borked→tinkering ошибки**: отчёты с `mentions_fix=1`, `has_concluding_notes=1` — пользователь описал попытки починки, но игра НЕ работает
- **oob→tinkering ошибки**: GE пользователи помечающие oob, модель видит GE → tinkering

LLM может перечитать текст и подтвердить/опровергнуть label:
- «Установил GE-Proton, запустил — работает отлично» → works_oob (даже если GE)
- «Пробовал разные версии, нашёл workaround через env vars» → tinkering
- «Пробовал всё, ничего не работает» → borked (не tinkering!)

**Cost**: ~348K × $0.001 = ~$350, одноразово. Может дать +0.02–0.05 F1 за счёт чистки label noise.

---

## D. Новые внешние источники данных (детальный ресёрч)

### D1. Steam Reviews

**API**: `GET https://store.steampowered.com/appreviews/{appid}?json=1&num_per_page=0`

**Аутентификация**: не требуется
**Rate limit**: ~1 req/sec безопасно, ~200/5min строгий лимит

**Ответ** (num_per_page=0 → только summary, без тела отзывов):
```json
{
  "query_summary": {
    "num_reviews": 0,
    "review_score": 6,           // 1-9, Valve scoring bucket
    "review_score_desc": "Mostly Positive",
    "total_positive": 20371,
    "total_negative": 6467,
    "total_reviews": 26838
  }
}
```

**Параметры фильтрации**:
- `filter=recent|updated|all` — сортировка
- `language=english` — фильтр по языку
- `purchase_type=steam|non_steam_key|all`
- `review_type=positive|negative|all`
- ⚠️ **Нет фильтра `primarily_steam_deck`** в summary endpoint — только в full reviews
- ⚠️ **Нет фильтра Linux vs Windows** — платформа только в индивидуальных отзывах

**Фичи для ML**:
| Фича | Тип | Описание |
|------|-----|----------|
| `review_score` | int 1-9 | Valve review bucket (proxy качества игры) |
| `review_pct_positive` | float | total_positive / total_reviews |
| `review_count` | int | total_reviews (proxy популярности) |
| `review_count_log` | float | log(total_reviews+1) |

**Покрытие**: ~95%+ наших 31K игр (почти все Steam игры имеют хотя бы 1 отзыв)
**Сложность**: LOW — 31K запросов × 1/sec = ~9 часов, одноразово. Можно параллелить.
**Ожидание**: +0.002–0.005 F1. Слабый indirect signal — review_score коррелирует с качеством игры, но не с Proton compatibility. Game embeddings уже кодируют per-game reputation. Единственный уникальный signal — `review_count` как proxy популярности (популярные → Valve тестирует → лучше Proton support).

### D2. Steam Player Counts

**API**: `GET https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}`

**Аутентификация**: НЕ требуется (public endpoint)
**Rate limit**: ~10 req/sec (generous)

**Ответ**:
```json
{"response": {"player_count": 44899, "result": 1}}
```

**Ограничения**:
- ⚠️ Только **текущее** количество игроков (snapshot), без истории
- Нет peak/average за период — только «сейчас»
- Для исторических данных нужен SteamDB (закрытый) или SteamCharts (scraping)

**SteamSpy** (`steamspy.com/api.php`):
- ⚠️ **Блокирует автоматические запросы** (403 Forbidden, Cloudflare)
- Данные: `owners` (range, e.g. "20M..50M"), `ccu` (peak yesterday), `average_forever`/`median_forever` (playtime mins), `average_2weeks`, `positive`/`negative` reviews, `price`, `tags`, `genre`
- Rate limit: 1 req/sec для appdetails, **1 req/min** для bulk `?request=all`
- Неточные после 2018 (Valve закрыла public profile data, owners → ML-оценка)
- **Не рекомендуется** как источник — ненадёжен и медлен

**Фичи для ML**:
| Фича | Тип | Описание |
|------|-----|----------|
| `current_players` | int | Текущие игроки (snapshot) |
| `current_players_log` | float | log(current_players+1) |
| `is_active` | binary | current_players > 0 |

**Покрытие**: ~100% (endpoint работает для всех appid, возвращает 0 для неактивных)
**Сложность**: LOW — 31K запросов × 0.1sec = ~1 час
**Ожидание**: +0.001–0.003 F1. Очень слабый signal. Snapshot текущих игроков нестабилен (время суток, релизы, распродажи). Без исторических данных — шум. Game embeddings уже кодируют popularity через SVD.

**Вердикт**: ❌ Не стоит. Snapshot без истории бесполезен. SteamSpy заблокирован. Лучше использовать `review_count` из D1 как proxy популярности.

### D3. WineHQ AppDB

**URL**: `https://appdb.winehq.org/`

**API**: ❌ **REST API отсутствует**. Только HTML-страницы, нужен scraping.

**Структура данных** (HTML scraping):
- **Application page**: `objectManager.php?sClass=application&iId={wine_id}&sAction=view`
  - Название, категория (Games > RPG), разработчик
  - Версии (каждая — отдельная страница)
  - Rating: Platinum / Gold / Silver / Bronze / Garbage
- **Version page**: `objectManager.php?sClass=version&iId={version_id}&sAction=view`
  - Rating (Gold), Wine версия тестирования
  - Таблица тестов: OS, дата, Wine version, Installs?, Runs?, Workaround?, Rating, Submitter
  - Текстовые комментарии: что работает, что нет

**Масштаб**: **29,914 приложений**, из них **~8,845 игр** (23 подкатегории: FPS 585, Action 1208, RPG 564, Strategy 942, etc.)
**Поиск**: только по имени через HTML POST form, нет search API

**Проблемы**:
1. ⚠️ **Нет маппинга Steam app_id → Wine app_id**. Только fuzzy match по имени
2. ⚠️ **HTML scraping** — хрупкий, медленный
3. ⚠️ **Wine ≠ Proton** — рейтинги для vanilla Wine, Proton имеет дополнительные патчи (DXVK, vkd3d, etc.). Gold в Wine может быть Platinum в Proton и наоборот
4. ⚠️ **Устаревшие данные** — многие тесты 2015-2020, Wine 5.x-6.x. Proton ушёл далеко вперёд
5. ⚠️ **Качество маппинга** — «Half-Life 2» vs «Half-Life® 2» vs «Half-Life 2: Episode One» → множественные matches

**Фичи для ML**:
| Фича | Тип | Описание |
|------|-----|----------|
| `wine_rating` | categorical | Platinum/Gold/Silver/Bronze/Garbage |
| `wine_test_count` | int | Количество тестовых отчётов |
| `wine_latest_test_year` | int | Год последнего теста |

**Покрытие**: оценка ~20-30% наших 31K игр после fuzzy matching (многие инди-игры отсутствуют)
**Сложность**: HIGH — scraping + fuzzy matching + quality control = 15-20ч
**Ожидание**: +0.001–0.003 F1. Wine rating — indirect и устаревший signal. Proton настолько отличается от Wine, что Gold в Wine мало что говорит о Proton compatibility в 2024+.

**Вердикт**: ❌ Не стоит. Высокая сложность, низкий и ненадёжный signal. Wine ≠ Proton.

### D4. Lutris Database

**API**: REST API, без аутентификации

**Endpoints**:
```
GET https://lutris.net/api/games                    → список (paginated)
GET https://lutris.net/api/games?search={query}     → поиск по имени
GET https://lutris.net/api/games/{slug}             → детали игры
```

**Rate limit**: умеренный (~2 req/sec, 429 при превышении)

**Список игр** (paginated, page_size до 100):
```json
{
  "count": 333740,
  "results": [{
    "id": 129, "name": "Portal", "slug": "portal",
    "year": 2007, "platforms": [{"name": "Linux"}, ...],
    "provider_games": [{"slug": "400", "name": "steam"}, {"name": "igdb", ...}]
  }]
}
```

**Деталь игры** (`/api/games/portal`):
```json
{
  "id": 129, "name": "Portal", "slug": "portal",
  "steamid": 400,                              // ← маппинг на Steam!
  "year": 2007,
  "user_count": 26743,                         // пользователи Lutris
  "platforms": ["Android", "Linux", "Windows", ...],
  "genres": ["Puzzle"],
  "installers": [
    {"slug": "Portal-Steam", "runner": "steam", "version": "Steam"},
    {"slug": "Portal-AppleII", "runner": "mame", "version": "AppleII"}
  ]
}
```

**Масштаб**: **333,740 игр** в базе (включая non-Steam), **82,083** с installers, **15,437** install scripts
**Маппинг**: `steamid` поле в detail endpoint + **`POST /api/games/service/steam`** с `{"appids": [400, 620, 730]}` — bulk lookup по Steam app_id!

**Bulk data**: GitHub `lutris/website` → releases: PostgreSQL dumps (latest: 2024-06-29). Также `GET /api/installers` — все 15K скриптов.

**Фичи для ML**:
| Фича | Тип | Описание |
|------|-----|----------|
| `has_lutris_entry` | binary | Есть запись в Lutris |
| `lutris_user_count` | int | Популярность в Lutris community |
| `lutris_installer_count` | int | Кол-во install scripts |
| `lutris_has_wine_installer` | binary | Есть Wine/Proton installer (vs native) |
| `lutris_runner_type` | categorical | steam/wine/linux (основной runner) |

**Покрытие**: оценка ~60-70% наших 31K игр (Lutris покрывает популярные Steam игры через provider_games)
**Сложность**: MEDIUM — bulk download через GitHub releases SQL dump или 3340 API запросов (~30 мин). Маппинг через steamid.
**Ожидание**: +0.001–0.003 F1. `lutris_user_count` — ещё один proxy popularity (game embeddings уже покрывают). `has_wine_installer` — слабый signal. Уникального compatibility signal нет.

**Вердикт**: ⚠️ Низкий приоритет. Данные доступны, маппинг есть, но ML signal слабый. Возможно полезно для engine layer (installer instructions), но не для ML.

### D5. ProtonDB Summaries (upgrade текущего)

**API**: `GET https://www.protondb.com/api/v1/reports/summaries/{appid}.json`

**Аутентификация**: не требуется
**Rate limit**: неизвестен (вероятно ~1 req/sec)

**Ответ**:
```json
{
  "tier": "gold",
  "bestReportedTier": "platinum",
  "trendingTier": "gold",
  "confidence": "strong",         // "strong" | "good" | "moderate" | "low"
  "score": 0.67,                  // 0.0-1.0
  "total": 330                    // кол-во отчётов
}
```

**Bulk data** (GitHub: `bdefore/protondb-data`):
- 89 tar.gz файлов в `/reports/` — полные дампы всех отчётов
- Обновления ~каждые 1-4 месяца (последний: dec1_2025)
- Формат: JSON, те же поля что наши reports (variant, verdict, notes, etc.)
- ⚠️ **Это наш первичный источник данных** — bulk exports и есть наши 348K reports

**⚠️ Target leak analysis**:
- `tier` — **прямой target leak** (агрегат verdicts, включая текущий отчёт)
- `score` — **прямой target leak** (0-1 score на основе тех же verdicts)
- `confidence` — partial leak (зависит от total, но не от конкретного verdict)
- `trendingTier` — **потенциально безопасен** если рассчитан на недавних отчётах (не включая текущий). Но нет гарантии
- `bestReportedTier` — **target leak** (максимум из всех verdicts)
- `total` — безопасен (count, не связан с verdict)

**Безопасные фичи**:
| Фича | Тип | Описание |
|------|-----|----------|
| `protondb_total_reports` | int | Кол-во отчётов (proxy popularity + community attention) |
| `protondb_confidence` | categorical | strong/good/moderate/low |

**Покрытие**: ~85%+ наших 31K игр (ProtonDB — наш источник)
**Сложность**: LOW — 31K запросов, или парсинг bulk data
**Ожидание**: +0.001–0.002 F1 (только безопасные фичи). `total_reports` = ещё один popularity proxy. `confidence` мало информативен.

**Вердикт**: ⚠️ Почти бесполезен для ML из-за target leak. `total_reports` дублирует информацию из game embeddings. Полезен только для engine layer (показать tier пользователю).

### D6. PCGamingWiki расширение

**Текущее**: `graphics_apis` + `engine` + `drm` + `anticheat` (25% покрытие graphics_apis)

**Дополнительные данные** (те же Cargo queries):
- System requirements: min/rec GPU, CPU, RAM
- Native Linux quality (good/bad/wrapper/none)
- Controller support (full/partial/none)
- Engine version (Unity 2019.4, Unreal 5.1, etc.)

**Покрытие**: varies, 30-60% для system requirements
**Сложность**: MEDIUM — расширить существующий scraper
**Ожидание**: +0.002–0.005 F1. Engine version может помочь (Unity 2019 → больше проблем с Proton чем Unity 2022).

**Вердикт**: ✅ Средний приоритет. Engine version — уникальный signal, расширяет текущий `engine` categorical. Но покрытие неполное.

### D7. IGDB (Twitch API) — NEW

**Что**: game metadata с деталями, недоступными из Steam — engine, player perspectives, game modes.

**API**: `POST https://api.igdb.com/v4/games` (Apicalypse query language)
**Аутентификация**: Twitch OAuth2 (Client ID + Secret, бесплатный tier)
**Rate limit**: 4 req/s, 8 concurrent

**Ответ** (пример запроса):
```
fields name, game_engines.name, player_perspectives.name,
       game_modes.name, themes.name, external_games.*;
where external_games.category = 1 & external_games.uid = "400";
```

**Ключевые поля** (недоступные из Steam/PCGamingWiki):
| Поле | Тип | ML signal |
|------|-----|-----------|
| `game_engines.name` | string | **TOP**: Unity vs Unreal vs custom → разное поведение Proton |
| `player_perspectives` | categorical | first-person/third-person/isometric → rendering complexity |
| `game_modes` | multi-value | single/multi/co-op/MMO → античит/networking issues |
| `themes` | multi-value | open-world/horror/sci-fi → proxy для engine complexity |
| `similar_games` | list of ids | graph features, collaborative filtering |
| `keywords` | multi-value | VR/early-access/procedural → specific Proton issues |

**Маппинг**: `external_games` endpoint с `category=1` (Steam) → прямой маппинг app_id
**Покрытие**: 200K+ игр, отличное покрытие Steam через cross-references
**Сложность**: LOW — REST API, Python wrapper `igdb-api-python`
**Ожидание**: **+0.005–0.015 F1** — `game_engines` с деталями (Unity 2019 vs 2022, Unreal 4 vs 5) потенциально один из сильнейших одиночных предикторов. Текущий `engine` из PCGamingWiki покрывает 25%, IGDB может расширить до 60-80%.

**Вердикт**: ✅ **Высокий приоритет**. Уникальные фичи, простой API, высокое покрытие.

### D8. Proton/DXVK Release Notes — NEW

**Что**: парсинг release notes для извлечения per-game fix history.

**Источники** (все на GitHub):
- `ValveSoftware/Proton/releases` — official Proton
- `doitsujin/dxvk/releases` — DXVK (DX9/10/11 → Vulkan)
- `HansKristian-Work/vkd3d-proton/releases` — vkd3d-proton (DX12 → Vulkan)
- `GloriousEggroll/proton-ge-custom/releases` — GE-Proton

**Формат** (высоко-структурированный):
```
- Fixed [Game Name] crashing on startup (#1234)
- [Game Name] is now playable
- Improved [Game Name] performance
```

**Фичи для ML**:
| Фича | Тип | Описание |
|------|-----|----------|
| `proton_fix_count` | int | Сколько раз игра упоминалась в release notes |
| `proton_fix_components` | multi-hot | DXVK/vkd3d/Wine/Proton — какие компоненты чинились |
| `proton_last_fix_version` | float | Версия Proton, в которой последний раз чинили |
| `proton_versions_since_fix` | int | Сколько версий прошло с последнего фикса |
| `ge_fix_count` | int | Фиксы в GE-Proton (community effort) |
| `was_ever_broken` | binary | Упоминалась ли игра как сломанная |

**Покрытие**: 500+ уникальных игр across all releases
**Сложность**: LOW — `gh api` + regex extraction
**Ожидание**: **+0.003–0.008 F1** — версионно-специфичный signal (game × proton_version), который GitHub issues не покрывают. `fix_count` = proxy «проблемности» игры.

**Вердикт**: ✅ **Высокий приоритет**. Быстро парсится, уникальный interaction signal.

### D9. HowLongToBeat — NEW

**Что**: время прохождения игр — proxy для complexity.

**Доступ**: Kaggle dataset `zaireali/howlongtobeat-games-scraper-2162025` (22.5 MB, Feb 2025)
**Маппинг**: `game_profile_steam` (Steam URL → app_id parsing)

**Фичи**: `main_story_hours`, `completionist_hours`, `player_count_playing/completed/retired`
**Покрытие**: 30-50K игр, ~60% overlap с нашими 31K
**Сложность**: LOW — скачать Kaggle dump, распарсить
**Ожидание**: +0.001–0.003 F1 (слабый indirect signal)

**Вердикт**: ⚠️ Низкий приоритет. Completion time — очень indirect proxy.

### D10. Сводная таблица источников

| Источник | API | Auth | Маппинг | Покрытие | Сложность | ML signal | Приоритет |
|----------|-----|------|---------|----------|-----------|-----------|-----------|
| **IGDB** | REST JSON | Twitch OAuth2 | external_games | 80%+ | LOW | **+0.005–0.015** | ✅ **высокий** |
| **Proton release notes** | GitHub API | нет | regex game name | ~500 игр | LOW | **+0.003–0.008** | ✅ **высокий** |
| **Steam Reviews** | REST JSON | нет | app_id | 95%+ | LOW | +0.002–0.005 | ✅ средний |
| **PCGamingWiki ext** | Cargo query | нет | app_id | 30-60% | MEDIUM | +0.002–0.005 | ✅ средний |
| **HLTB** | Kaggle dump | нет | Steam URL | 60% | LOW | +0.001–0.003 | ⚠️ низкий |
| **Steam Players** | REST JSON | нет | app_id | 100% | LOW | +0.001–0.003 | ❌ snapshot бесполезен |
| **WineHQ** | HTML scraping | нет | fuzzy name | 20-30% | HIGH | +0.001–0.003 | ❌ Wine≠Proton |
| **Lutris** | REST JSON | нет | steamid | 60-70% | MEDIUM | +0.001–0.003 | ⚠️ низкий |
| **ProtonDB** | REST JSON | нет | app_id | 85%+ | LOW | +0.001–0.002 | ⚠️ target leak |
| vulkan.gpuinfo.org | REST | нет | GPU model | varies | MEDIUM | per-user match | ❌ позже |
| CrossOver | scraping | нет | fuzzy name | low | HIGH | indirect | ❌ skip |
| Reddit | API/PRAW | OAuth | fuzzy | sparse | HIGH | unstructured | ❌ skip |

---

## F. Собственное тестирование (self-testing)

### Анализ дыр в датасете

| Проблема | Масштаб | Как self-testing помогает |
|----------|---------|--------------------------|
| **63.6% игр с ≤3 отчётами** | 19,691 игр | Binary launch test для sparse games |
| **54.6% игр без отчётов с 2025** | 16,904 игр | Fresh data с актуальным Proton |
| **proton_version null в 97%** | 337K reports | Точная привязка к версии |
| **works_oob bias** в sparse games | 6.7% oob для 1-report vs 13.1% для 50+ | Контролируемое тестирование |

### Что можно измерить автоматически

| Метрика | Метод | Надёжность |
|---------|-------|------------|
| Запускается ли (launch test) | `timeout 60 steam -applaunch {appid}` + exit code | HIGH |
| Крашится ли в первые N сек | monitor process, exit code ≠ 0 | HIGH |
| FPS | MangoHud `MANGOHUD_OUTPUT` CSV log | HIGH |
| Звук | PulseAudio sink activity | MEDIUM |
| Proton prefix size | `du -s compatdata/{appid}` | HIGH |
| VRAM usage | MangoHud log | HIGH |

### Что НЕЛЬЗЯ измерить автоматически

- Графические артефакты (нужен visual regression / ML)
- Input/controller проблемы
- Мультиплеер/античит
- Gameplay beyond main menu
- Разница tinkering vs works_oob

### Практический план

**Оборудование**: Steam Deck + десктоп с NVIDIA
**Scope**: 1000 игр × 2 версии Proton (official + GE) × 2 hardware = 4000 тестов
**Время**: ~30 сек/тест → ~33 часа (можно за 2-3 дня)

**Скрипт автоматизации:**
```bash
# Псевдокод
for appid in target_games:
    for proton in [official, ge]:
        set_proton_version(appid, proton)
        result = timeout 60 steam -applaunch $appid
        record(appid, proton, gpu, driver, exit_code, duration, fps_log)
```

**Выбор target_games** (по приоритету):
1. Игры с ≤3 отчётами (19K) — random sample 500
2. Игры без отчётов 2025+ (17K) — random sample 300
3. Игры с высокой works_oob uncertainty (из uncertainty zone анализа) — 200

**Формат выходных данных:**
```
app_id, proton_version, gpu, driver, kernel, timestamp,
launches(bool), time_to_crash_sec, exit_code,
fps_avg, fps_1pct_low, vram_mb, audio_active(bool)
```

### ML impact

**Как training data**: 4000 записей = 1.1% от 348K. Слишком мало для прямого обучения.

**Как validation/calibration**: ✅ Высокая ценность
- Controlled environment → no label noise
- Binary signal (launches/crashes) → ground truth для Stage 1
- Per-version testing → заполняет `proton_version` gap

**Как augmentation для sparse games**: ✅ Средняя ценность
- Для 500 sparse games добавляем 1-2 отчёта с controlled conditions
- Уменьшает bias: sparse games → не автоматически tinkering

**Рекомендация**: реализовать как **validation pipeline**, не как training data. Запускать при каждом новом Proton release для regression detection.

---

## E. Архитектурные изменения

### E1. Ordinal regression вместо classification

**Проблема**: 3 класса — ordinal шкала (borked < tinkering < works_oob). Текущий каскад не учитывает порядок в Stage 2.

**Решение**: LightGBM с custom ordinal loss или ordinal regression wrapper:
```
P(borked) = σ(f₁(x))
P(tinkering) = σ(f₂(x)) - σ(f₁(x))
P(works_oob) = 1 - σ(f₂(x))
```

**Ожидание**: может улучшить calibration и уменьшить «прыжки через класс» (borked→oob ошибки).

### E2. Multi-task learning: per-fault prediction

Вместо одной 3-class модели — обучить auxiliary heads:
- `has_audio_issues`: binary
- `has_performance_issues`: binary
- `is_playable`: binary
- Main task: 3-class verdict

Shared trunk + task-specific heads. Фичи из auxiliary tasks информируют main task.

### E3. Hierarchical features: game cluster → report

Game embeddings показывают кластеры (UMAP подтверждает). Идея:
1. Кластеризовать игры (k=50-100)
2. Per-cluster statistics: средний borked rate, средний fault rate
3. Как фичи в модель: «этот кластер игр обычно работает/не работает»

### E4. Cross-report features

**Текущее ограничение**: модель видит каждый отчёт независимо.

Но для той же игры другие отчёты несут сигнал:
- `same_game_borked_rate_recent`: % borked среди последних 10 отчётов
- `same_game_same_gpu_verdict`: что другие с таким же GPU получили
- `same_game_proton_version_borked`: borked rate именно на этой версии Proton

**⚠️ Target leak**: если включить все отчёты. Решение: leave-one-out или использовать только **старшие** отчёты (по timestamp < текущий).

---

## Результаты Phase 6a (эксперимент)

> Скрипт: `scripts/experiment_phase6a.py`

### Сводная таблица

| Эксперимент | F1 macro | Δ | Stage 1 LL | Stage 2 LL | Verdict |
|---|---|---|---|---|---|
| **COMBINED (A1+A4+A3+A5)** | **0.8858** | **+0.1544** | 0.050 | 0.197 | ⚠️ TARGET LEAK (tried_oob) |
| A3b: dur+tried+tinker | 0.8799 | +0.1484 | 0.061 | 0.198 | ⚠️ TARGET LEAK |
| A3: duration+tried_oob | 0.8774 | +0.1459 | 0.065 | 0.198 | ⚠️ TARGET LEAK |
| **A1: Fault booleans** | **0.7464** | **+0.0149** | 0.098 | 0.395 | ✅ **РЕАЛЬНЫЙ ПРИРОСТ** |
| A4: Cust+Flag | 0.7340 | +0.0025 | 0.124 | 0.393 | ✅ слабый |
| A5: anticheat | 0.7323 | +0.0008 | 0.124 | 0.393 | ❌ нет эффекта |
| Baseline | 0.7315 | — | 0.124 | 0.394 | — |
| B1: drop variant S2 | 0.7005 | −0.0310 | 0.124 | 0.432 | ❌ ухудшение |

### Per-class метрики (лучшие без leak)

| Модель | borked P/R/F1 | tinkering P/R/F1 | works_oob P/R/F1 |
|---|---|---|---|
| Baseline | 0.854/0.806/0.830 | 0.843/0.860/0.851 | 0.522/0.505/0.514 |
| **A1: Faults** | **0.892/0.842/0.866** | 0.850/0.862/0.856 | 0.520/0.515/0.517 |

### Target leak analysis

**`tried_oob`** = прямой target leak:
- Вопрос в анкете ProtonDB: "Did you try the game out of the box?" → `tried_oob = yes/no`
- Если `tried_oob=yes` И игра работает → `verdict_oob=yes` (т.е. works_oob)
- `tried_oob_bin` gain в Stage 2: **2.5M** (топ-1, больше чем все остальные фичи вместе)
- Это часть того же вопросника, что и target. **Нельзя использовать.**

**`duration`** = soft leak:
- Пользователь играл "moreThanTenHours" → скорее всего works_oob (не будет играть 10ч в сломанную игру)
- gain в Stage 1: 252K, Stage 2: 31K
- Менее прямой leak чем tried_oob, но всё равно: duration известен только post-factum
- **Нельзя использовать** при предсказании для нового пользователя (duration ещё не известен)

**`tinker_override`** = leak:
- "Overrode the default tinkering label" — мета-информация о процессе лейблинга
- gain: 212K Stage 1, 54K Stage 2
- **Нельзя использовать**

### Реальные результаты (без leak)

**A1 (fault booleans)** — единственный значимый прирост: **+0.0149 F1**
- Stage 1 LogLoss: 0.124 → **0.098** (−21%!)
- `fault_count` = 3.8M gain в Stage 1 (топ-1 фича, больше чем text_emb)
- `stability_faults` = 874K (второй по значимости)
- 87.5% покрытие (новый формат анкеты с октября 2019)
- **Рекомендация: интегрировать**

**A4 (cust+flag)** — слабый +0.0025
- Покрытие: cust 11%, flag 3.3% — слишком мало для time-based split (новые фичи в test, старые без них в train)
- `cust_any` = 31K gain Stage 2, `cust_config_change` = 18K — для подмножества сильный signal
- **Рекомендация: интегрировать** (не навредит, LightGBM обрабатывает NaN)

**A5 (anticheat)** — нет эффекта (+0.0008)
- Покрытие 9.2%, game-level `anticheat`/`anticheat_status` уже покрывает этот сигнал
- **Рекомендация: интегрировать** (zero-cost)

**B1 (drop variant)** — ухудшение (−0.031)
- `variant` содержит реальный сигнал, не только bias
- GE-Proton действительно чаще требует tinkering (это факт, не bias)
- **Рекомендация: не трогать variant**

### Feature importance (COMBINED model, Stage 1 топ-5 новых)

| Фича | Gain Stage 1 | Gain Stage 2 |
|---|---|---|
| `fault_count` | 3,680K | 2,085 |
| `stability_faults` | 792K | 452 |
| `tried_oob_bin` ⚠️ leak | 754K | 2,539K |
| `fault_any` | 697K | 1,816 |
| `duration_ord` ⚠️ leak | 252K | 31K |

---

## Приоритизация (ROI) — обновлённая

| Приоритет | Действие | Факт/ожидание | Стоимость |
|-----------|----------|---------------|-----------|
| ~~1~~ | ~~A1: Fault booleans → интеграция~~ | ~~+0.0149 F1~~ | **DONE** |
| ~~2~~ | ~~A4: Cust+Flag → интеграция~~ | ~~+0.0025 F1~~ | **DONE** |
| ~~A5~~ | ~~anticheat per-report~~ | ~~+0.0008~~ | **DONE** (zero-cost) |
| **3** | **D7: IGDB (game_engines, modes)** | +0.005–0.015 | 4ч кода |
| **4** | C1: LLM text extraction | +0.010–0.025 | ~$100 + 4ч |
| **5** | D8: Proton release notes parsing | +0.003–0.008 | 3ч кода |
| **6** | C2: LLM verdict relabeling | +0.020–0.050 | ~$350 + 8ч |
| **7** | E4: Cross-report features | +0.005–0.015 | 4ч кода |
| **8** | F: Self-testing pipeline | validation, not F1 | 8ч кода + hardware |
| **9** | D1: Steam Reviews | +0.002–0.005 | 3ч кода |
| **10** | E1: Ordinal regression | architecture change | 6ч кода |
| ⚠️ | A3: duration+tried_oob | +0.146 (TARGET LEAK) | не использовать |
| ❌ | B1: drop variant S2 | −0.031 | не трогать |
| ❌ | D2-D5, D9: слабые источники | ~0 | не стоит |

---

## Порядок реализации (обновлённый)

### Phase 6a: zero-cost report features — **DONE**

1. ~~Эксперимент A1-A5, B1~~ **DONE** (scripts/experiment_phase6a.py)
2. ~~Интеграция A1+A4+A5~~ **DONE** (protondb_settings/ml/train.py, 183 features)
3. ~~Ретрейн: F1=0.7494 (+0.0179)~~ **DONE**

### Phase 6b: новые источники (приоритеты 3, 5)

4. **IGDB enrichment**: fetch game_engines, player_perspectives, game_modes via Twitch API
5. **Proton release notes**: parse ValveSoftware/Proton + DXVK + vkd3d + GE releases
6. Эксперимент с IGDB + release notes features

### Phase 6c: LLM features (приоритеты 4, 6)

7. Batch LLM extraction (effort_level, playability_score)
8. Эксперимент с LLM features
9. (Опционально) LLM relabeling

### Phase 6d: advanced features (приоритеты 7+)

10. Cross-report features (leave-one-out)
11. Self-testing validation pipeline
12. Steam Reviews enrichment

---

## Метрики успеха (обновлённые)

- **Phase 6a target**: F1 macro ≥ 0.745 (+0.014) → **ДОСТИГНУТО: 0.7464 (A1 faults)**
- **Phase 6b target**: F1 macro ≥ 0.760 (+0.029)
- **Stage 1 LogLoss**: < 0.100 (с текущих 0.124) → **ДОСТИГНУТО: 0.098 (A1 faults)**
- **Stage 2 LogLoss**: < 0.370 (с текущих 0.394)
- **works_oob recall**: ≥ 0.55 (с текущих 0.505)
- **Confidence ≥ 0.8**: accuracy ≥ 88% при coverage ≥ 75%
