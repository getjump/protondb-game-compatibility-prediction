# Исследование: Детекция технологий из Steam данных

Дата: 2026-03-14

Цель: Понять, какие технологии (движок, графический API, античит, DRM, middleware) можно определить из Steam данных для предсказания совместимости с Proton/Wine.

---

## 1. Как SteamDB определяет движок игры

### Основной метод: FileDetectionRuleSets

SteamDB использует **открытый проект** [FileDetectionRuleSets](https://github.com/SteamDatabase/FileDetectionRuleSets) -- набор regex-правил, которые запускаются по **именам файлов** в депо Steam (НЕ по содержимому файлов).

**Ключевые факты:**
- Сканируются только имена файлов (filenames), не содержимое -- 250M+ файлов
- Регулярные выражения case-insensitive
- Путь файла полный, например `game/bin/win64/dota2.exe`
- Правила определены в `rules.ini` в INI-формате

### Двухпроходная система

**Первый проход (slam dunk):** Ищет очевидные сигнатуры движков.
Один файл = достаточно для идентификации. Примеры:

| Движок | Паттерн файла |
|--------|---------------|
| Unity | `UnityEngine.dll`, `UnityPlayer.dll`, `globalgamemanagers.assets` |
| Unreal | `Engine/Shaders/Binaries/`, `Engine/Binaries/ThirdParty/`, `.uasset`, `.upk` |
| Godot | `project.godot`, `libgodotsteam`, `GodotSharp.dll` |
| Source | `vphysics.dll`, `bsppack.dll` |
| Source 2 | `gameinfo.gi` |
| CryEngine | `cry3dengine.dll`, `CryRenderD3D11.dll`, `CryRenderVulkan.dll` |
| Frostbite | `Engine.BuildInfo*.dll`, `Runtime_Win64_retail.BuildSettings` |
| REDengine | `.redscripts`, `.w2scripts` |
| RE Engine | `re_chunk_000.pak` |
| RPG Maker | `rgss*.dll`, `rpg_rt.exe`, `rpg_core.js`, `.rgssad` |
| Ren'Py | `renpy/`, `.rpyb` |
| GameMaker | `game.unx`, `Steamworks.gmk.dll`, `audiogroup1.dat` |
| GoldSource | `vgui.dll` |
| idTech | `.pk4`, `.streamed`, `.mega2` |
| Love2D | `liblove.dll` |
| FNA | `fna.dll` |
| XNA | `XNA`, `.xnb` |
| Defold | `game.dmanifest` |

**Второй проход (Evidence/TryDeduceEngine):** Для неопределённых после первого прохода.
Комбинация нескольких "косвенных" признаков:

```
GameMaker: options.ini + data.win + snd_*.ogg (2 из 3 = GameMaker)
Godot: .pck файлы + exe с тем же именем, или data.pck
BioWare Aurora: .bif + .tlk + (.rim | .tga)
BioWare Infinity: .bif + .tlk (без .rim/.tga)
RAGE: .rpf + metadata.dat
idTech0: DOSBOX + VSWAP
Build (Duke3D): DOSBOX + BUILD.EXE/COMMIT.DAT/GAME.CON
```

### Категории правил

| Секция | Назначение | Пример |
|--------|-----------|--------|
| **Engine** | Движки игр | Unity, Unreal, Source |
| **Evidence** | Косвенные признаки (2-й проход) | .arc, .bif, .pck |
| **Container** | Обёртки (Electron) | LICENSE.electron.txt |
| **Emulator** | Эмуляторы | dosbox.exe, scummvm.exe |
| **AntiCheat** | Античит | BattlEye, EAC, PunkBuster |
| **SDK** | Библиотеки/middleware | FMOD, Wwise, PhysX, DLSS |
| **Launcher** | Лаунчеры издателей | Ubisoft, EA, Rockstar |

### Ограничения

- Бот SteamDB должен иметь доступ к спискам файлов (через Token Dumper)
- HTML5-движки практически не детектируются
- Некоторые движки не оставляют уникальных следов
- Ложные срабатывания возможны (одна игра может матчить несколько движков -- напр. лаунчер на Unity + игра на другом)

---

## 2. Как определяется графический API

### Из FileDetectionRuleSets (файлы в депо)

Прямой детекции "Graphics API" категории в rules.ini **НЕТ**. Однако можно извлечь косвенные сигналы:

**Из движка CryEngine:**
```
CryRenderD3D11.dll -> DirectX 11
CryRenderD3D12.dll -> DirectX 12 (если бы был)
CryRenderVulkan.dll -> Vulkan
```

**Из SDK-правил:**
```ini
AMD_FidelityFX = amd_fidelityfx_dx12.dll | amd_fidelityfx_vk.dll  -> DX12/Vulkan
Intel_XeSS = igxess.dll | libxess.dll -> DX12 (XeSS требует DX12)
NVIDIA_DLSS = nvngx_dlss.dll -> DX11/DX12
DirectStorage = dstorage.dll -> DX12/NVMe
```

### Из pc_requirements (Steam Store API)

Поле `pc_requirements` в ответе `appdetails` содержит HTML с минимальными/рекомендуемыми требованиями.
DirectX версия часто упоминается, например:
- "DirectX: Version 9.0c"
- "DirectX: Version 11"
- "DirectX: Version 12"

**Формат:** HTML-строка, требует парсинга. Ненадёжно -- не все разработчики заполняют корректно.

### Из launch config (PICS appinfo)

Аргументы запуска могут содержать подсказки:
- `-dx11`, `-dx12`, `-vulkan`, `-opengl`, `-force-d3d11`
- Но это пользовательские launch options, не из PICS

### Из файлов в депо (потенциально, за пределами rules.ini)

Можно было бы детектировать по файлам:
- `d3d9.dll`, `d3d10.dll`, `d3d11.dll`, `d3d12.dll` -> DirectX 9/10/11/12
- `vulkan-1.dll` -> Vulkan
- `opengl32.dll` -> OpenGL
- `dxvk_d3d11.dll` -> DXVK (уже через Proton)

Но SteamDB **не делает** такой детекции -- эти DLL часто стандартные и не включены в депо.

### Рекомендация для проекта

Лучший источник -- **парсинг pc_requirements HTML** из Steam Store API + косвенные признаки из движка (Unity обычно DX11, Unreal может DX11/12/Vulkan).

---

## 3. Как определяется античит

### Из FileDetectionRuleSets (имена файлов)

```ini
[AntiCheat]
BattlEye = BEService_x64?.exe
EasyAntiCheat = EasyAntiCheat_(EOS_)?Setup.exe | EasyAntiCheat(_x64)?.dll | eac_server64.dll
EA_AntiCheat = EAAntiCheat.Installer.exe
EQU8 = equ8_conf.json
PunkBuster = PnkBstrA.exe | pbsvc.exe | pbsv.dll | Punkbuster/
Ricochet = Randgrid.sys
nProtect_GameGuard = gameguard.des
XIGNCODE3 = .xem
AntiCheatExpert = AntiCheatExpert/ | AceAntibotClient/
AnyBrain = anybrainSDK.dll | Cerebro.dll
BlackCipher = BlackCall64?.aes | BlackCat64.sys
HackShield = HShield/HSInst.dll
TenProtect = TP3Helper.exe
FredaikisAntiCheat = FredaikisAntiCheat/
NetEase = NeacSafe64(_ex)?.sys | NEP2.dll
```

### Из PICS данных

В PICS common секции может быть поле `anti_cheat_support` (недокументировано, но упоминается в контексте SteamDB).

### Из Steam Deck Compatibility

В результатах Deck verification проверяется наличие античита и его совместимость с Proton. Поле `steam_deck_compatibility` в appinfo.vdf содержит результаты тестирования, включая `SteamDeckVerified_TestResult_*` строки.

---

## 4. Как определяется DRM

### Из Steam Store API (appdetails)

```json
{
  "drm_notice": "Denuvo Anti-Tamper"
}
```

Поле `drm_notice` **присутствует** в JSON ответе `store.steampowered.com/api/appdetails?appids=XXXX` когда разработчик указал DRM. Пример: DOOM: The Dark Ages (appid 3017860) -> `"Denuvo Anti-Tamper"`.

Также есть поле `ext_user_account_notice` для сторонних аккаунтов:
```json
{
  "ext_user_account_notice": "Rockstar Games (Supports Linking to Steam Account)"
}
```

### Из PICS appinfo

В PICS common секции:
- `eulas` -- массив EULA (может содержать DRM-related EULA)
- Информация о 3rd-party DRM отражается в истории изменений SteamDB

### Из FileDetectionRuleSets

Правил для DRM **НЕТ** в текущем rules.ini. DRM определяется другими способами:
- `drm_notice` через Store API
- PCGamingWiki (основной источник для нашего проекта -- уже используется в PLAN_ENRICHMENT)
- AWACY (Are We Anti-Cheat Yet)

### Из файлов в депо (потенциально)

Denuvo можно было бы детектировать по:
- Размеру исполняемого файла (Denuvo увеличивает размер)
- Но для этого нужен доступ к размерам файлов через манифест

Steam DRM (встроенный):
- Проверяется наличием Steam API DLL + проверки владения
- Не видно напрямую из файловых листингов

---

## 5. Манифесты депо: доступ к спискам файлов

### Что содержит манифест

Каждый манифест содержит полный список файлов депо:

| Поле | Описание |
|------|----------|
| `filename` | Полный путь файла |
| `size` | Размер в байтах |
| `flags` | Флаги (директория/файл/исполняемый/symlink) |
| `sha_content` | SHA1 хэш содержимого |
| `sha_filename` | SHA1 хэш имени |
| `chunks` | Чанки с offset, crc, sha, размеры |
| `is_executable` | Является ли файл исполняемым |

Формат: Protobuf (сжатый).

### Доступ программно: ValvePython/steam

```python
from steam.client import SteamClient
from steam.client.cdn import CDNClient

client = SteamClient()
client.login(username, password)

cdn = CDNClient(client)
cdn.load_licenses()

# Получить манифесты для app
manifests = cdn.get_manifests(app_id=570)  # Dota 2

# Итерировать файлы (без скачивания содержимого!)
for manifest in manifests:
    for file in manifest.iter_files():
        print(file.filename, file.size, file.is_executable)

# Или фильтровать
for file in cdn.iter_files(app_id=570, filename_filter="*.dll"):
    print(file.filename)
```

### Требования к доступу

- **Аутентификация:** Нужен Steam аккаунт (логин+пароль)
- **Владение игрой:** Для получения manifest request code нужно владеть приложением
- **Manifest request codes:** Валидны только 15 минут
- **Token Dumper:** SteamDB использует краудсорсинг через [SteamTokenDumper](https://github.com/SteamDatabase/SteamTokenDumper) -- пользователи отдают access tokens для приложений которыми владеют

### Альтернативы

1. **SteamCMD** -- можно получить manifest через `download_depot`, но требует владение
2. **DepotDownloader** (C#, SteamKit2) -- аналогично
3. **Без аутентификации** -- НЕВОЗМОЖНО получить список файлов

### Для нашего проекта

**Вывод:** Мы НЕ можем использовать depot file listings напрямую -- нужно владеть каждой игрой. Вместо этого можно:
1. Использовать SteamDB Technologies API/данные (если доступен экспорт)
2. Парсить страницу steamdb.info/tech/ (но rate limited)
3. Использовать rules.ini для детекции на локально установленных играх

---

## 6. Другие технологические сигналы из Steam данных

### Middleware (из FileDetectionRuleSets SDK секции)

| Middleware | Файловый паттерн |
|-----------|-----------------|
| **Bink Video** | `bink2w64.dll`, `binkw32.dll` |
| **FMOD** | `fmod.dll`, `fmodstudio.dll`, `libfmod.so` |
| **Wwise** | `SoundbanksInfo.xml`, `.bnk`, `.wem`, `AkSoundEngine.dll` |
| **PhysX** | `PhysX_64.dll`, `PhysX3_x64.dll`, `PhysXCore.dll` |
| **NVIDIA DLSS** | `nvngx_dlss.dll`, `sl.dlss.dll` |
| **NVIDIA DLSS FG** | `nvngx_dlssg.dll`, `sl.dlss_g.dll` |
| **NVIDIA Reflex** | `nvlowlatencyvk.dll`, `sl.reflex.dll` |
| **Intel XeSS** | `igxess.dll`, `libxess.dll` |
| **AMD FSR** | `amd_fidelityfx_dx12.dll`, `amd_fidelityfx_vk.dll` |
| **DirectStorage** | `dstorage.dll` |
| **CEF** | `libcef.dll` (Chromium Embedded) |
| **Electron** | `LICENSE.electron.txt` |
| **SDL** | `sdl2.dll`, `sdl3.dll` |
| **OpenAL** | `OpenAL32.dll` |
| **Vivox** | `vivox`, `ortp.dll` |
| **CRIWARE** | `.cpk`, `.sfd`, `.usm`, `.adx`, `.acb` |
| **Havok** | (не в rules.ini -- не детектируется) |
| **OpenVR** | `openvr_api.dll` |
| **OpenXR** | `openxr_loader.dll` |
| **Vulkan SDK** | (нет прямого правила) |

### Архитектура (32/64-bit)

Из PICS appinfo:
```json
{
  "config": {
    "launch": {
      "0": {
        "executable": "game.exe",
        "config": {
          "oslist": "windows",
          "osarch": "64"
        }
      }
    }
  }
}
```

Поле `osarch` в launch config: `"32"` или `"64"`.
Также `osarch` на уровне `common` секции.

### Redistributables (.NET, VC++)

Из файлов в депо (потенциально):
- `vcredist_x64.exe` -> Visual C++ Runtime
- `dotnetfx35.exe` -> .NET Framework
- `dxsetup.exe` -> DirectX Runtime

### Steam Input API

Из `common.controller_support`:
- `"full"` -- полная поддержка
- `"partial"` -- частичная

Также из categories в Store API (category 28 = Full controller support, 18 = Partial).

### Shader Pre-Caching

Из PICS common секции: наличие поддержки shader pre-caching определяется полями типа `gpu_vulkan_*`.

### Steam Cloud

Из Store API categories: category 23 = Steam Cloud.
Из PICS: `ufs` секция содержит конфигурацию Cloud Save (quota, maxnumfiles, rootoverrides).

### Steam Deck Compatibility

Из PICS common секции:
- `steam_deck_compatibility` содержит результаты Deck verification
- Значения: Verified, Playable, Unsupported, Unknown
- Включает отдельные тест-кейсы: `SteamDeckVerified_TestResult_*`

### Поддерживаемые платформы

Из PICS common:
- `oslist`: `"windows"`, `"macos"`, `"linux"` (через запятую)

Наличие Linux-порта -- сильный сигнал для совместимости с Proton.

---

## 7. Open-source инструменты SteamDB

### GitHub: [github.com/SteamDatabase](https://github.com/SteamDatabase)

| Репозиторий | Язык | Назначение |
|------------|------|-----------|
| **[FileDetectionRuleSets](https://github.com/SteamDatabase/FileDetectionRuleSets)** | PHP/INI | Правила детекции технологий по именам файлов |
| **[SteamTokenDumper](https://github.com/SteamDatabase/SteamTokenDumper)** | C# | Сбор access tokens для PICS доступа |
| **[SteamAppInfo](https://github.com/SteamDatabase/SteamAppInfo)** | C# | Парсер appinfo.vdf и packageinfo.vdf |
| **[ValveResourceFormat](https://github.com/SteamDatabase/ValveResourceFormat)** | C# | Парсер форматов Source 2 |
| **[ValvePak](https://github.com/SteamDatabase/ValvePak)** | C# | Работа с .vpk архивами |
| **[ValveKeyValue](https://github.com/SteamDatabase/ValveKeyValue)** | C# | Парсер VDF (KeyValue) формата |
| **[BrowserExtension](https://github.com/SteamDatabase/BrowserExtension)** | JS | Расширение для Steam в браузере |
| **SteamDatabaseBackend** | C# | Бэкенд SteamDB (мониторинг изменений) |

### Другие полезные проекты

| Проект | Назначение |
|--------|-----------|
| [ValvePython/steam](https://github.com/ValvePython/steam) | Python: PICS, CDN, манифесты, Steam Client |
| [SteamRE/SteamKit](https://github.com/SteamRE/SteamKit) | C#: Библиотека для работы со Steam |
| [SteamRE/DepotDownloader](https://github.com/SteamRE/DepotDownloader) | C#: Загрузка депо Steam |
| [DoctorMcKay/steam-pics-api](https://github.com/DoctorMcKay/steam-pics-api) | Node.js: HTTP API для PICS |

---

## 8. Практическая реализация для нашего проекта

### Что можно получить БЕЗ владения играми

| Источник | Данные | Метод |
|----------|--------|-------|
| Steam Store API | `drm_notice`, `pc_requirements`, `categories`, `ext_user_account_notice`, `controller_support`, `platforms` | HTTP GET `store.steampowered.com/api/appdetails` |
| Steam Web API | Базовая информация, достижения | API ключ |
| PICS (без владения) | `common` секция: `oslist`, `osarch`, `type`, `controller_support`, `steam_deck_compatibility` | ValvePython/steam + логин |
| PCGamingWiki | DRM, античит, движок (частично) | API/скрейпинг (уже в проекте) |
| AWACY | Античит + Proton совместимость | API (уже в проекте) |
| ProtonDB | Пользовательские отчёты | API (основной источник) |

### Что НЕЛЬЗЯ получить без владения

| Данные | Почему |
|--------|--------|
| Списки файлов депо | Нужен access token + владение |
| FileDetectionRuleSets результаты | Нужны файловые листинги |
| Содержимое файлов | Нужно скачивание |

### Рекомендуемая стратегия детекции

**Уровень 1: Без дополнительных затрат (Steam Store API + PICS common)**

```
game_engine       <- pc_requirements парсинг + PCGamingWiki
graphics_api      <- pc_requirements HTML парсинг (DirectX version)
has_anticheat     <- AWACY + PCGamingWiki
has_drm           <- drm_notice из appdetails + PCGamingWiki
architecture      <- osarch из PICS launch config
controller_support <- common.controller_support
steam_deck_status <- steam_deck_compatibility из PICS
```

**Уровень 2: С использованием FileDetectionRuleSets (для локальных игр)**

Применить rules.ini к файлам локально установленных игр:
```python
import re
from configparser import ConfigParser

# Парсинг rules.ini
config = ConfigParser()
config.read('rules.ini')

# Для каждого файла в steamapps/common/<game>/
for filepath in game_files:
    for section in config.sections():
        for pattern_name, regex in config.items(section):
            if re.search(regex, filepath, re.IGNORECASE):
                print(f"{section}.{pattern_name}: {filepath}")
```

**Уровень 3: Crowdsourced данные**

Использовать данные SteamDB Technologies (steamdb.info/tech/) через кэширование/скрейпинг.

### Приоритет фичей для ML модели

По влиянию на Proton-совместимость:

1. **has_anticheat** (EAC/BattlEye) -- критично, часто = несовместимо
2. **has_drm** (Denuvo) -- проблемы с производительностью + offline
3. **game_engine** (Unity/Unreal) -- определяет базовый уровень совместимости
4. **graphics_api** (DX9/11/12/Vulkan) -- DX12 менее стабилен через DXVK/VKD3D
5. **architecture** (32/64-bit) -- 32-bit может иметь проблемы
6. **has_native_linux** -- сильнейший предиктор
7. **controller_support** -- косвенный сигнал качества портирования
8. **middleware** (FMOD/Wwise/PhysX) -- минимальное влияние

---

## Источники

- [SteamDatabase/FileDetectionRuleSets](https://github.com/SteamDatabase/FileDetectionRuleSets) -- основной инструмент детекции
- [SteamDB Technologies](https://steamdb.info/tech/) -- результаты детекции
- [SteamDB FAQ](https://steamdb.info/faq/) -- PICS документация
- [ValvePython/steam](https://github.com/ValvePython/steam) -- Python библиотека для Steam
- [steam CDN docs](https://steam.readthedocs.io/en/stable/api/steam.client.cdn.html) -- CDNClient API
- [steam manifest docs](https://steam.readthedocs.io/en/stable/api/steam.core.manifest.html) -- Manifest API
- [SteamRE/SteamKit](https://github.com/SteamRE/SteamKit) -- SteamKit2 (.NET)
- [SteamRE/DepotDownloader](https://github.com/SteamRE/DepotDownloader) -- DepotDownloader
- [SteamDatabase/SteamTokenDumper](https://github.com/SteamDatabase/SteamTokenDumper) -- Token Dumper
- [SteamDatabase/SteamAppInfo](https://github.com/SteamDatabase/SteamAppInfo) -- appinfo.vdf парсер
- [Steam DRM docs](https://partner.steamgames.com/doc/features/drm) -- Steamworks DRM
- [Steam Deck Compat](https://partner.steamgames.com/doc/steamdeck/compat) -- Deck verification
- [PCGamingWiki DRM list](https://www.pcgamingwiki.com/wiki/The_big_list_of_third-party_DRM_on_Steam)
- [PCGamingWiki Anti-cheat](https://www.pcgamingwiki.com/wiki/Anti-cheat_middleware)
- [StorefrontAPI docs](https://wiki.teamfortress.com/wiki/User:RJackson/StorefrontAPI)
- [WindowsGSM/SteamAppInfo](https://github.com/WindowsGSM/SteamAppInfo) -- примеры PICS JSON
- [Nik Davis: Steam Data Collection](https://nik-davis.github.io/posts/2019/steam-data-collection/)
