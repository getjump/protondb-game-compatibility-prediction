# Steam PICS (Product Information Cache System) — Исследование

> Дата: 2026-03-14

## 1. Что такое PICS

PICS (Product Information Cache System) — внутренняя система Steam для хранения и распространения метаданных об приложениях (apps) и пакетах (packages). Это та же самая система, которую использует SteamCMD (`app_info_print`), SteamDB.info и сам клиент Steam.

**Ключевые концепции:**
- **Change Number** — глобально инкрементируемый счётчик. Каждое изменение в любом app/package получает новый номер. Позволяет делать инкрементальные обновления через `PICSChangesSince`.
- **Access Token** — некоторые apps/packages требуют токен для получения полной информации. Токены доступны только при наличии лицензии (владении игрой). Без токена `_missing_token=True` в ответе.
- **VDF** — формат данных Valve (KeyValue text). App info приходит в текстовом VDF, package info — в бинарном VDF.

## 2. Структура данных PICS app_info

Данные возвращаются в формате VDF (Valve Data Format) с иерархической структурой. Основные секции:

### 2.1 `common` — общая информация

```
"common"
{
    "name"                    "Counter-Strike 2"
    "type"                    "Game"          // Game, DLC, Tool, Demo, Music, Config, etc.
    "oslist"                  "windows,linux" // ← КЛЮЧЕВОЕ: список поддерживаемых ОС
    "osarch"                  "64"            // 32, 64 или пусто
    "icon"                    "..."
    "logo"                    "..."
    "logo_small"              "..."
    "clienticon"              "..."
    "clienttga"               "..."
    "controller_support"      "full"          // full, partial, none
    "steam_deck_compatibility"
    {
        "category"            "2"             // 0=Unknown, 1=Unsupported, 2=Playable, 3=Verified
        "tested"              "1"
        "test_timestamp"      "1679616000"
        "tests"
        {
            ...
        }
    }
    "metacritic_score"        "83"
    "metacritic_fullurl"      "..."
    "review_score"            "7"
    "review_percentage"       "83"
    "store_asset_mtime"       "..."
    "store_tags"
    {
        "0"                   "1663"          // tag IDs
        "1"                   "1774"
    }
    "category"                                // Steam categories
    {
        "category_1"          "1"             // Multi-player
        "category_2"          "1"             // Single-player
        "category_18"         "1"             // Partial Controller Support
        "category_22"         "1"             // Steam Achievements
        "category_28"         "1"             // Full controller support
        "category_29"         "1"             // Steam Trading Cards
        "category_35"         "1"             // In-App Purchases
        "category_41"         "1"             // Remote Play on Phone
        "category_42"         "1"             // Remote Play on Tablet
        "category_43"         "1"             // Remote Play on TV
        "category_44"         "1"             // Remote Play Together
        "category_62"         "1"             // Family Sharing
    }
    "genres"
    {
        "0"                   "1"             // genre IDs
    }
    "associations"
    {
        "0"
        {
            "type"            "developer"
            "name"            "Valve"
        }
        "1"
        {
            "type"            "publisher"
            "name"            "Valve"
        }
    }
    "primary_genre"           "1"
    "has_adult_content"       "0"
    "has_adult_content_sex"   "0"
    "has_adult_content_violence" "0"
    "market_presence"         "1"
    "workshop_visible"        "1"
    "community_hub_visible"   "1"
    "exfgls"                  "9"
    "supported_languages"
    {
        "english"
        {
            "supported"       "true"
            "full_audio"      "true"
            "subtitles"       "true"
        }
        "russian" { ... }
    }
    "small_capsule"           { ... }
    "header_image"            { ... }
    "library_assets"          { ... }
    "library_assets_full"     { ... }
    "content_descriptors"
    {
        "ids"
        {
            "0"               "2"             // Violence
            "1"               "5"             // General Mature Content
        }
    }
    "releasestate"            "released"
    "steam_release_date"      "1376020800"    // Unix timestamp
    "original_release_date"   "..."
}
```

**Ключевые поля для Proton-совместимости:**
- `oslist` — `"windows"` (только Windows), `"windows,macos,linux"` (мультиплатформа), `"windows,linux"` (Windows+Linux)
- `osarch` — `"64"` или `""` (32-bit или обе)
- `type` — `"Game"`, `"DLC"`, `"Tool"`, `"Demo"`, `"Music"`, `"Config"`
- `steam_deck_compatibility.category` — `0`/`1`/`2`/`3`
- `controller_support` — `"full"`, `"partial"`, `"none"`
- `store_tags` — теги из магазина (включая "Linux" тег и др.)
- `category` — steam categories (VAC, мультиплеер, etc.)

### 2.2 `config` — конфигурация запуска

```
"config"
{
    "installdir"              "Counter-Strike Global Offensive"
    "contenttype"             "3"
    "launch"
    {
        "0"
        {
            "executable"      "game/csgo.exe"  // или .sh для Linux
            "type"            "default"         // default, none, option, server, editor
            "config"
            {
                "oslist"      "windows"         // ← ОС для этого launch entry
                "osarch"      "64"              // архитектура
            }
            "description"     ""
            "description_loc"
            {
                "english"     ""
            }
        }
        "1"
        {
            "executable"      "game/csgo.sh"
            "type"            "default"
            "config"
            {
                "oslist"      "linux"
                "osarch"      "64"
            }
        }
        "2"
        {
            "executable"      "game/csgo.exe"
            "type"            "server"
            "config"
            {
                "oslist"      "windows"
            }
            "description"     "Play Counter-Strike 2 Dedicated Server"
        }
    }
    "steamcontrollertemplateindex"  "2"
    "steamcontrollertouchconfigdetails"
    {
        ...
    }
}
```

**Ключевые поля:**
- `launch.N.executable` — путь к исполняемому файлу
- `launch.N.type` — тип запуска (`default`, `none`, `option`, `server`, `editor`)
- `launch.N.config.oslist` — для какой ОС этот launch entry
- `launch.N.config.osarch` — архитектура (`32`, `64`)
- `launch.N.config.betakey` — для какой ветки (beta) этот launch entry
- `launch.N.description` — описание (для type=option)

### 2.3 `depots` — хранилища файлов

```
"depots"
{
    "branches"
    {
        "public"
        {
            "buildid"         "16488375"
            "timeupdated"     "1710000000"
        }
        "dpr"
        {
            "buildid"         "16488200"
            "pwdrequired"     "1"
            "timeupdated"     "1710000000"
        }
    }
    "730"                                     // depot ID = app ID (базовый)
    {
        "config"
        {
            "oslist"          "windows"        // ← ОС для этого depot
        }
        "manifests"
        {
            "public"          "2345678901234567890"
        }
        "maxsize"             "34567890123"
        "depotfromapp"        "730"            // ← наследование из другого app
    }
    "731"
    {
        "config"
        {
            "oslist"          "linux"          // ← Linux depot!
        }
        "manifests"
        {
            "public"          "3456789012345678901"
        }
    }
    "732"
    {
        "config"
        {
            "oslist"          "windows"
            "osarch"          "64"
        }
        "manifests"
        {
            "public"          "4567890123456789012"
        }
    }
}
```

**Ключевые поля:**
- `depots.DEPOT_ID.config.oslist` — ОС для depot (`"windows"`, `"linux"`, `"macos"`)
- `depots.DEPOT_ID.depotfromapp` — depot наследуется из другого app (важно для DLC)
- `depots.branches.public.buildid` — текущий build ID
- `depots.branches.public.timeupdated` — время последнего обновления

### 2.4 `extended` — расширенная информация

```
"extended"
{
    "developer"               "Valve"
    "developer_url"           "https://www.valvesoftware.com"
    "homepage"                "https://..."
    "gamedir"                 "csgo"
    "serverbrowsername"       "Counter-Strike 2"
    "denuvo_context"          "..."           // ← DRM-related!
    "listofdlc"               "1234,5678"     // список app ID DLC
    "noservers"               "1"
    "isfreeapp"               "1"
    "anti_cheat_support_url"  "..."
    "requireskbmouse"         "0"             // требуется ли KB+мышь
    "languages"               "english,french,german,..."
}
```

### 2.5 `ufs` — User File System (облачные сохранения)

```
"ufs"
{
    "rootoverrides"
    {
        "0"
        {
            "os"              "linux"
            "oscompare"       "equals"
            "rootadditionoverride" "..."
        }
    }
    "savefiles"
    {
        "0"
        {
            "root"            "gameinstall"
            "path"            "game/csgo/cfg"
            "pattern"         "*.cfg"
        }
    }
}
```

## 3. Python-библиотеки для доступа к PICS

### 3.1 ValvePython/steam (РЕКОМЕНДУЕТСЯ для нашего проекта)

**Пакет:** `steam[client]` на PyPI
**GitHub:** https://github.com/ValvePython/steam
**Лицензия:** MIT
**Python:** 2.7+, 3.4+
**Модель:** синхронная (gevent)

**Установка:**
```bash
pip install -U "steam[client]"
```

Зависимости: `gevent`, `protobuf`, `vdf`, `cachetools`, `six`

**Ключевые возможности:**
- `SteamClient` — полный клиент Steam CM протокола
- `anonymous_login()` — анонимный вход без учётных данных
- `get_product_info()` — получение PICS данных
- `get_changes_since()` — инкрементальные обновления
- `get_access_tokens()` — получение токенов доступа
- VDF парсинг встроен (зависимость `vdf`)

**Полный пример — получение app_info:**

```python
from steam.client import SteamClient
from steam.enums import EResult

client = SteamClient()

# Анонимный логин — не требует учётных данных
result = client.anonymous_login()
if result != EResult.OK:
    raise Exception(f"Login failed: {result}")

# Получение info для одного приложения
data = client.get_product_info(apps=[730])  # Counter-Strike 2
app_info = data['apps'][730]

# Структура app_info:
# {
#   'common': {'name': 'Counter-Strike 2', 'type': 'Game', 'oslist': 'windows,linux', ...},
#   'config': {'launch': {'0': {'executable': '...', 'config': {'oslist': '...'}}, ...}},
#   'depots': {'730': {'config': {'oslist': 'windows'}, ...}, ...},
#   'extended': {'developer': '...', ...},
#   '_missing_token': False,
#   '_change_number': 12345678,
#   '_sha': 'abcdef...',
#   '_size': 12345
# }

# Извлечение ключевых полей
common = app_info.get('common', {})
print(f"Name: {common.get('name')}")
print(f"Type: {common.get('type')}")
print(f"OS List: {common.get('oslist')}")
print(f"OS Arch: {common.get('osarch')}")

# Deck compatibility
deck = common.get('steam_deck_compatibility', {})
print(f"Deck Category: {deck.get('category')}")  # 0/1/2/3

# Launch configurations
config = app_info.get('config', {})
for launch_id, launch in config.get('launch', {}).items():
    lconfig = launch.get('config', {})
    print(f"Launch {launch_id}: {launch.get('executable')} "
          f"os={lconfig.get('oslist')} arch={lconfig.get('osarch')} "
          f"type={launch.get('type')}")

# Depots — check for Linux depot
depots = app_info.get('depots', {})
for depot_id, depot_data in depots.items():
    if isinstance(depot_data, dict) and 'config' in depot_data:
        os_list = depot_data['config'].get('oslist', '')
        if 'linux' in os_list:
            print(f"Linux depot found: {depot_id}")

client.logout()
```

**Пакетное получение данных:**

```python
from steam.client import SteamClient
from steam.enums import EResult

client = SteamClient()
client.anonymous_login()

# Можно запрашивать до ~сотен apps за раз
# Steam отвечает чанками (response_pending=True пока не все отправлены)
app_ids = [730, 570, 440, 292030, 1245620]  # CS2, Dota2, TF2, Witcher3, Elden Ring
data = client.get_product_info(apps=app_ids)

for app_id, info in data['apps'].items():
    common = info.get('common', {})
    print(f"{app_id}: {common.get('name')} | OS: {common.get('oslist')} | "
          f"Token missing: {info.get('_missing_token')}")

client.logout()
```

**Инкрементальные обновления:**

```python
from steam.client import SteamClient
from steam.enums import EResult

client = SteamClient()
client.anonymous_login()

# Получить текущий change number (начать с 0 для полного дампа — НЕ РЕКОМЕНДУЕТСЯ)
# Лучше сначала получить текущий номер
changes = client.get_changes_since(
    change_number=0,  # Начальная точка
    app_changes=True,
    package_changes=False
)

if changes:
    print(f"Current change number: {changes.current_change_number}")
    print(f"App changes: {len(changes.app_changes)}")

    # Каждое изменение содержит:
    for change in changes.app_changes[:5]:
        print(f"  App {change.appid}: change #{change.change_number}, "
              f"needs_token={change.needs_token}")

    # Далее можно запросить product_info для изменённых apps
    changed_app_ids = [c.appid for c in changes.app_changes]
    if changed_app_ids:
        # Запрашивать порциями по ~200
        chunk = changed_app_ids[:200]
        data = client.get_product_info(apps=chunk)

# Для последующих запросов использовать сохранённый change_number
# saved_change_number = changes.current_change_number

client.logout()
```

### 3.2 Gobot1234/steam.py — асинхронная альтернатива

**Пакет:** `steamio` на PyPI
**GitHub:** https://github.com/Gobot1234/steam.py
**Лицензия:** MIT
**Python:** 3.10+
**Модель:** asyncio (async/await)

```bash
pip install steamio
```

- Поддерживает PICS через `fetch_product_info(app_ids, package_ids)`
- Async-native, вдохновлён discord.py
- Использует `betterproto` вместо стандартного `protobuf`
- Более современный код, но менее зрелый чем ValvePython/steam
- Нет явного метода `anonymous_login` — ориентирован на ботов с аккаунтом

**Примечание:** Для нашего проекта (worker, batch processing) ValvePython/steam предпочтительнее — проще в использовании, не требует async runtime, доказан в production (steamctl построен на нём).

### 3.3 steamctl

**Пакет:** `steamctl` на PyPI
**GitHub:** https://github.com/ValvePython/steamctl
**Построен на:** ValvePython/steam

CLI-утилита с кэшированием product info. Полезна как reference implementation:
- `CachingSteamClient` — обёртка с файловым кэшем app_info в JSON
- Поддержка анонимного логина (`--anonymous`)
- Batch-запросы packages по 100 штук

### 3.4 SteamKit2 (C#) — справочно

**GitHub:** https://github.com/SteamRE/SteamKit
**Язык:** C#/.NET
**Используется:** SteamDB.info

Это самая полная реализация Steam протокола. SteamDB построен на SteamKit2. Для нашего Python-проекта не подходит, но полезен как reference для понимания протокола.

### 3.5 vdf — парсер VDF формата

**Пакет:** `vdf` на PyPI
**GitHub:** https://github.com/ValvePython/vdf

```python
import vdf

# Текстовый VDF (app info)
data = vdf.loads(vdf_text)

# Бинарный VDF (package info)
data = vdf.binary_loads(vdf_bytes)

# Запись
text = vdf.dumps(data, pretty=True)
```

Устанавливается автоматически с `steam[client]`.

## 4. Аутентификация и доступ

### 4.1 Анонимный логин

```python
client = SteamClient()
result = client.anonymous_login()
# result == EResult.OK при успехе
```

**Что доступно анонимно:**
- ✅ `get_product_info(apps=[...])` — базовая информация обо всех приложениях
- ✅ `get_changes_since(change_number)` — отслеживание изменений
- ✅ `get_player_count(app_id)` — количество игроков
- ⚠️ Некоторые apps возвращают `_missing_token=True` — полные данные доступны только с токеном
- ❌ `get_access_tokens()` — токены выдаются только для игр в библиотеке аккаунта
- ❌ Package info обычно требует токены из лицензий

**Практический вывод:** Анонимного логина достаточно для получения `common`, `config`, `depots`, `extended` секций для большинства игр (type=Game). Секция `common` (содержащая `oslist`, `type`, `controller_support`, `steam_deck_compatibility`) доступна практически всегда.

### 4.2 Что доступно по данным SteamDB

Из FAQ SteamDB:
> «This is publicly available information about every application and package on Steam which can be acquired by anyone with a regular Steam account. You can verify this by opening the Steam console and typing `app_info_print 440`.»

SteamDB собирает токены через SteamTokenDumper — утилиту, которую пользователи запускают добровольно для отправки своих PICS-токенов.

### 4.3 Rate Limits

Явных документированных rate limits нет. Практические наблюдения:
- SteamDB обновляет топ-800 игр каждые 5 минут, остальные — каждые 10 минут
- Steam кэширует player count ~5 минут
- `get_product_info` поддерживает батчи (десятки-сотни apps за запрос)
- `get_changes_since` возвращает все изменения с указанного номера
- Слишком агрессивные запросы могут привести к disconnection от CM сервера
- Рекомендуется: не более ~50-100 apps за запрос, пауза между батчами 1-2 секунды

## 5. Практический подход для нашего проекта

### 5.1 Стратегия сбора данных

```
1. Initial Load:
   - anonymous_login()
   - Получить список всех app IDs из ProtonDB reports (уже есть в БД)
   - Батчами по 50-100 запросить get_product_info()
   - Сохранить change_number каждого app в SQLite
   - Сохранить max change_number глобально

2. Incremental Updates (периодически):
   - anonymous_login()
   - get_changes_since(saved_change_number)
   - Запросить product_info только для изменённых apps
   - Обновить БД

3. Извлечение фичей для ML:
   - has_linux_native: 'linux' in common.oslist
   - has_linux_depot: any depot with config.oslist containing 'linux'
   - has_linux_launch: any launch with config.oslist == 'linux'
   - is_64bit: common.osarch == '64' or any launch with osarch == '64'
   - app_type: common.type
   - controller_support: common.controller_support
   - deck_category: common.steam_deck_compatibility.category
   - has_native_linux_exe: launch entry with oslist=linux and executable ending in .sh
   - category_flags: multi-player, VAC, etc.
   - store_tags: tag IDs
   - depot_count: total depots
   - linux_depot_count: depots with oslist containing 'linux'
   - has_drm_fields: presence of denuvo_context in extended, thirdpartycdkey, etc.
```

### 5.2 Schema для SQLite

```sql
CREATE TABLE steam_app_info (
    app_id INTEGER PRIMARY KEY,
    name TEXT,
    app_type TEXT,                    -- Game, DLC, Tool, etc.
    oslist TEXT,                      -- 'windows,linux,macos'
    osarch TEXT,                      -- '64', '32', ''
    controller_support TEXT,          -- full, partial, none
    deck_category INTEGER,           -- 0,1,2,3
    deck_tested INTEGER DEFAULT 0,

    -- Launch configs (JSON array)
    launch_configs TEXT,              -- JSON: [{exe, oslist, osarch, type}, ...]

    -- Depot summary
    has_linux_depot INTEGER DEFAULT 0,
    has_windows_depot INTEGER DEFAULT 0,
    has_macos_depot INTEGER DEFAULT 0,
    depot_count INTEGER DEFAULT 0,

    -- Extended
    has_drm_indication INTEGER DEFAULT 0,
    developer TEXT,
    publisher TEXT,

    -- Metadata
    store_tags TEXT,                  -- JSON array of tag IDs
    categories TEXT,                  -- JSON array of category IDs
    genres TEXT,                      -- JSON array of genre IDs

    -- PICS metadata
    change_number INTEGER,
    pics_token_missing INTEGER DEFAULT 0,

    -- Timestamps
    fetched_at INTEGER,
    updated_at INTEGER
);
```

### 5.3 Полный рабочий скрипт

```python
"""
Steam PICS data fetcher for ProtonDB settings project.
Fetches app metadata via Steam CM protocol using anonymous login.
"""
import time
import json
import sqlite3
import logging
from steam.client import SteamClient
from steam.enums import EResult

log = logging.getLogger(__name__)


class SteamPICSFetcher:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = SteamClient()
        self.client.set_credential_location('.')

    def connect(self) -> bool:
        """Connect to Steam via anonymous login."""
        result = self.client.anonymous_login()
        if result != EResult.OK:
            log.error(f"Anonymous login failed: {result}")
            return False
        log.info("Anonymous login successful")
        return True

    def disconnect(self):
        self.client.logout()

    def fetch_apps_batch(self, app_ids: list[int], batch_size: int = 50) -> dict:
        """Fetch product info for multiple apps in batches."""
        all_data = {}

        for i in range(0, len(app_ids), batch_size):
            batch = app_ids[i:i + batch_size]
            log.info(f"Fetching batch {i // batch_size + 1}: "
                     f"apps {i+1}-{min(i+batch_size, len(app_ids))} of {len(app_ids)}")

            try:
                data = self.client.get_product_info(apps=batch)
                if data:
                    all_data.update(data.get('apps', {}))
            except Exception as e:
                log.error(f"Error fetching batch: {e}")

            # Rate limiting — pause between batches
            if i + batch_size < len(app_ids):
                time.sleep(1.5)

        return all_data

    def extract_features(self, app_id: int, info: dict) -> dict:
        """Extract ML-relevant features from PICS app_info."""
        common = info.get('common', {})
        config = info.get('config', {})
        depots = info.get('depots', {})
        extended = info.get('extended', {})

        # OS detection
        oslist = common.get('oslist', '')
        has_linux = 'linux' in oslist

        # Launch configs
        launches = []
        for lid, launch in config.get('launch', {}).items():
            lconf = launch.get('config', {})
            launches.append({
                'id': lid,
                'executable': launch.get('executable', ''),
                'oslist': lconf.get('oslist', ''),
                'osarch': lconf.get('osarch', ''),
                'type': launch.get('type', ''),
            })

        has_linux_launch = any(
            'linux' in l.get('oslist', '') for l in launches
        )

        # Depot analysis
        linux_depot_count = 0
        windows_depot_count = 0
        macos_depot_count = 0
        depot_count = 0

        for depot_id, depot_data in depots.items():
            if not isinstance(depot_data, dict):
                continue
            if 'config' not in depot_data and 'manifests' not in depot_data:
                continue
            depot_count += 1
            depot_os = depot_data.get('config', {}).get('oslist', '')
            if 'linux' in depot_os:
                linux_depot_count += 1
            if 'windows' in depot_os:
                windows_depot_count += 1
            if 'macos' in depot_os:
                macos_depot_count += 1

        # Deck compatibility
        deck = common.get('steam_deck_compatibility', {})
        deck_category = int(deck.get('category', 0))

        # DRM indication
        has_drm = bool(
            extended.get('denuvo_context')
            or extended.get('thirdpartycdkey')
            or any('drm' in str(v).lower() for v in extended.values()
                   if isinstance(v, str))
        )

        # Store tags
        store_tags = list(common.get('store_tags', {}).values())

        # Categories
        categories = []
        for k, v in common.get('category', {}).items():
            if v == '1':
                # Extract category number from 'category_N'
                cat_num = k.replace('category_', '')
                categories.append(cat_num)

        return {
            'app_id': app_id,
            'name': common.get('name', ''),
            'app_type': common.get('type', ''),
            'oslist': oslist,
            'osarch': common.get('osarch', ''),
            'controller_support': common.get('controller_support', ''),
            'deck_category': deck_category,
            'deck_tested': int(bool(deck.get('tested'))),
            'launch_configs': json.dumps(launches),
            'has_linux_depot': int(linux_depot_count > 0),
            'has_windows_depot': int(windows_depot_count > 0),
            'has_macos_depot': int(macos_depot_count > 0),
            'depot_count': depot_count,
            'has_drm_indication': int(has_drm),
            'developer': (extended.get('developer')
                          or next((a['name'] for a in common.get('associations', {}).values()
                                   if isinstance(a, dict) and a.get('type') == 'developer'), '')),
            'publisher': (next((a['name'] for a in common.get('associations', {}).values()
                                if isinstance(a, dict) and a.get('type') == 'publisher'), '')),
            'store_tags': json.dumps(store_tags),
            'categories': json.dumps(categories),
            'genres': json.dumps(list(common.get('genres', {}).values())),
            'change_number': info.get('_change_number', 0),
            'pics_token_missing': int(info.get('_missing_token', False)),
            'fetched_at': int(time.time()),
        }

    def get_changes_since(self, change_number: int) -> tuple[list[int], int]:
        """Get list of changed app IDs since a change number.

        Returns (changed_app_ids, current_change_number).
        """
        resp = self.client.get_changes_since(
            change_number=change_number,
            app_changes=True,
            package_changes=False
        )

        if resp is None:
            return [], change_number

        changed_ids = [c.appid for c in resp.app_changes]
        return changed_ids, resp.current_change_number
```

## 6. Поля, полезные для предсказания Proton-совместимости

| Поле | Путь в PICS | Значение для ML |
|------|------------|-----------------|
| Native Linux | `common.oslist` contains "linux" | Сильный позитивный сигнал |
| Linux depot | `depots.N.config.oslist` = "linux" | Есть Linux-файлы |
| Linux launch | `config.launch.N.config.oslist` = "linux" | Нативный Linux-запуск |
| 64-bit | `common.osarch` = "64" | Архитектура |
| App type | `common.type` | Game vs DLC vs Tool |
| Controller | `common.controller_support` | Deck-readiness |
| Deck verified | `common.steam_deck_compatibility.category` | Прямой сигнал |
| Store tags | `common.store_tags` | Жанр, тип игры |
| Categories | `common.category` | VAC, мультиплеер |
| DRM hints | `extended.denuvo_context`, etc. | DRM = проблемы |
| Launch type | `config.launch.N.type` | default vs server |
| `.exe` vs `.sh` | `config.launch.N.executable` | Формат исполняемого файла |
| Depot from app | `depots.N.depotfromapp` | Наследование depot |
| Build freshness | `depots.branches.public.timeupdated` | Как давно обновлялось |

## 7. Сравнение с Steam Store API

| Критерий | PICS (CM Protocol) | Store API (`/api/appdetails`) |
|----------|-------------------|-------------------------------|
| Формат | VDF (parsed to dict) | JSON |
| Аутентификация | Анонимный логин через CM | Без аутентификации (HTTP) |
| Batch запросы | Да, десятки-сотни за раз | Только 1 app за запрос |
| Rate limit | Мягкий (disconnect при злоупотреблении) | ~200 запросов/5 мин |
| Данные: oslist | ✅ Полные | ✅ Только platforms bool |
| Данные: launch config | ✅ Полные (exe, os, arch, type) | ❌ Нет |
| Данные: depots | ✅ Полные (depot IDs, OS, manifests) | ❌ Нет |
| Данные: extended | ✅ Полные (DRM hints, etc.) | ❌ Ограничено |
| Данные: Deck compat | ✅ Полные | ❌ Нет |
| Данные: store tags | ✅ Tag IDs | ✅ Категории + жанры |
| Change tracking | ✅ change_number system | ❌ Нет |
| Протокол | TCP (Steam CM) | HTTPS |

**Вывод:** PICS значительно превосходит Store API по полноте данных и эффективности batch-запросов. Store API полезен как дополнение для данных, которых нет в PICS (описания, скриншоты, системные требования текстом).

## 8. Как это делает SteamDB

По данным из FAQ и анализа их open-source инструментов:

1. **SteamKit2 (C#)** — основная библиотека для подключения к Steam CM
2. **PICSChangesSince** — периодический polling изменений (каждые несколько секунд)
3. **PICSProductInfo** — запрос обновлённых apps/packages
4. **SteamTokenDumper** — open-source утилита, которую пользователи запускают добровольно для отправки PICS-токенов из своих Steam-клиентов. Это позволяет SteamDB получать access tokens для apps, которые недоступны анонимно.
5. **Store page parsing** — дополнительно парсят страницы магазина для данных, отсутствующих в PICS.
6. **Backend** — приватный репозиторий (SteamDatabaseBackend), не доступен публично.

## 9. Протокольные детали (Protobuf)

Основные сообщения (из `steammessages_clientserver_appinfo.proto`):

### PICSProductInfoRequest
```protobuf
message CMsgClientPICSProductInfoRequest {
    message AppInfo {
        optional uint32 appid = 1;
        optional uint64 access_token = 2;
    }
    message PackageInfo {
        optional uint32 packageid = 1;
        optional uint64 access_token = 2;
    }
    repeated PackageInfo packages = 1;
    repeated AppInfo apps = 2;
    optional bool meta_data_only = 3;
    optional uint32 num_prev_failed = 4;
    optional uint32 supports_package_tokens = 5;
}
```

### PICSProductInfoResponse
```protobuf
message CMsgClientPICSProductInfoResponse {
    message AppInfo {
        optional uint32 appid = 1;
        optional uint32 change_number = 2;
        optional bool missing_token = 3;
        optional bytes sha = 4;
        optional bytes buffer = 5;        // ← text VDF, декодируется vdf.loads()
        optional bool only_public = 6;
        optional uint32 size = 7;
    }
    message PackageInfo {
        optional uint32 packageid = 1;
        optional uint32 change_number = 2;
        optional bool missing_token = 3;
        optional bytes sha = 4;
        optional bytes buffer = 5;        // ← binary VDF, декодируется vdf.binary_loads()
        optional uint32 size = 6;
    }
    repeated AppInfo apps = 1;
    repeated uint32 unknown_appids = 2;
    repeated PackageInfo packages = 3;
    repeated uint32 unknown_packageids = 4;
    optional bool meta_data_only = 5;
    optional bool response_pending = 6;   // ← True = ещё будут чанки
}
```

### PICSChangesSinceRequest / Response
```protobuf
message CMsgClientPICSChangesSinceRequest {
    optional uint32 since_change_number = 1;
    optional bool send_app_info_changes = 2;
    optional bool send_package_info_changes = 3;
}

message CMsgClientPICSChangesSinceResponse {
    message AppChange {
        optional uint32 appid = 1;
        optional uint32 change_number = 2;
        optional bool needs_token = 3;
    }
    optional uint32 current_change_number = 1;
    optional uint32 since_change_number = 2;
    optional bool force_full_update = 3;
    repeated AppChange app_changes = 5;
}
```

### PICSAccessTokenRequest / Response
```protobuf
message CMsgClientPICSAccessTokenRequest {
    repeated uint32 packageids = 1;
    repeated uint32 appids = 2;
}

message CMsgClientPICSAccessTokenResponse {
    message AppToken {
        optional uint32 appid = 1;
        optional uint64 access_token = 2;
    }
    repeated AppToken app_access_tokens = 3;
    repeated uint32 app_denied_tokens = 4;
}
```

## 10. Рекомендации для проекта

1. **Библиотека:** `steam[client]` (ValvePython/steam) — зрелая, синхронная, с anonymous login и батчевыми запросами.

2. **Аутентификация:** Анонимный логин достаточен для наших целей (получение oslist, launch configs, depots, deck compat).

3. **Интеграция:** Добавить в worker как новый enrichment source, аналогично Steam Store API и PCGamingWiki.

4. **Частота обновления:** При первоначальной загрузке — батчами по 50 apps. Далее — инкрементально через `get_changes_since`.

5. **Кэширование:** Сохранять `change_number` для каждого app. Использовать глобальный `change_number` для инкрементальных обновлений.

6. **Обработка ошибок:**
   - Reconnect при disconnect
   - Retry при timeout
   - Skip apps с `_missing_token=True` (или пометить для повторной попытки с токеном)

7. **Приоритет фич для ML:**
   - `oslist` (native Linux) — высочайший
   - `deck_category` — высокий
   - `launch_configs` с oslist — высокий
   - `depot` OS distribution — средний
   - `store_tags` — средний
   - DRM indicators — средний
