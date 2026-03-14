# Анализ качества данных ProtonDB dump

Файл: `reports_piiremoved.json`, 348,683 отчётов, 30,968 уникальных app_ids.

---

## 1. systemInfo

### systemInfo.gpu

- **Всего**: 348,679 непустых, **35,114 уникальных**
- **Валидных** (содержат nvidia|amd|radeon|geforce|intel|rx|gtx|rtx): 345,573 (99.1%)
- **Не прошли фильтр**: 3,106

| Категория | Примеры | Реально мусор? |
|---|---|---|
| nouveau (open-source NVIDIA) | `nouveau NVC8`, `nouveau NV134` | **Нет** — реальные NVIDIA GPU |
| Виртуальные GPU | `VMware, Inc. llvmpipe`, `Red Hat virgl` | **Нет** — валидные данные |
| OS в неправильном поле | `"Arch Linux" (64 bit)`, `Ubuntu 19.04` | **Да** — мусор |

**Решение**: LLM по уникальным строкам → `gpu_normalization`. Regex пропустит nouveau/llvmpipe/virgl.

### systemInfo.cpu

- **Всего**: 348,679 непустых, **Валидных**: 348,133 (99.8%), **Мусор**: 546
- Примеры мусора: `%s1`, `0x0`, `aaa...`, `Unterstützt`, `Spicy Silicon`, `VirtualApple @ 2.50GHz`

**Решение**: LLM по уникальным строкам → `cpu_normalization`.

### systemInfo.ram

- **Всего**: 348,679, **Валидных**: 348,448 (99.93%), **Мусор**: 231
- Мусор: `Intel Haswell HDMI`, `English`, `USB Mixer`, `Speakers (Realtek Audio)`, `enough`
- Топ: `16 GB` (103K), `32 GB` (101K), `15 GB` (55K), `64 GB` (20K), `8 GB` (18K)

**Решение**: regex `(\d+)` — формат стабилен.

### systemInfo.kernel

- **Всего**: 348,679, **Валидных**: 347,968 (99.8%), **Мусор**: 711
- Мусор: `Поддерживается`, `Unterstützt`, `支援`, `BlazinBowlsKernelBruh`, `0x6`

**Решение**: regex `(\d+\.\d+[\.\d]*)` — формат ядра Linux стабилен.

### systemInfo.xWindowManager

- **Непустых**: 150,941 (43.3%), **Уникальных**: 174
- Топ: KWin (55.8%), GNOME Shell (19.4%), Mutter (10.2%)

**Решение**: импортируем (ранее игнорировали). Полезен для определения Wayland/X11 и DE.

### systemInfo.steamRuntimeVersion

- **Непустых**: 150,873 (43.3%), **Уникальных**: 163
- Часто содержит lspci output — ненадёжен.

**Решение**: игнорируем.

---

## 2. responses: основные категориальные поля

| Поле | Покрытие | Значения |
|---|---|---|
| **verdict** | 100% | `yes` 80.3%, `no` 19.7% |
| **type** | 29.6% | `steamPlay` 23.0%, `tinker` 6.6% |
| **variant** | 70.4% | `official` 36.8%, `ge` 12.0%, `experimental` 11.3%, `native` 3.9%, `notListed` 3.5%, `older` 2.9% |
| **rating** | **0%** | **Не существует в дампе!** |
| **triedOob** | 29.1% | `yes` 19.5%, `no` 9.7% |
| **verdictOob** | 19.6% | `yes` 11.1%, `no` 8.5% |
| **tinkerOverride** | 32.6% | `no` 31.0%, `yes` 1.6% |

**Важно**: `rating` (Platinum/Gold/Silver/Bronze/Borked) — нет в данных. Удалён из схемы.

### responses.protonVersion

- **Непустых**: 113,168 (32%), **Уникальных**: 1,657
- **"Default"**: 87,852 (77.6% непустых) — бесполезно
- **Реальные версии**: ~25K записей (`6.3-8`, `7.0-6`, `Proton-6.21-GE-2`, `Experimental`)

**Решение**: trim + `"Default"|""` → NULL. Regex `(\d[\d.\-]+\d)` для версий.

### responses.customProtonVersion

- **Непустых**: 41,786 (12%), **Уникальных**: 1,087
- Топ: `GE-Proton9-27` (1,394), `GE-Proton8-25` (1,157), `GE-Proton9-11` (1,007)

**Важно**: это **основной источник конкретных версий GE-Proton**! `protonVersion` в 78% "Default".

---

## 3. Воронка запуска (91-100% покрытие)

| Поле | Покрытие | Yes | No |
|---|---|---|---|
| **installs** | 100% | 99.7% | 0.3% |
| **opens** | 99.7% | 91.1% | 8.7% |
| **startsPlay** | 91.1% | 96.0% | 4.0% |

**duration** (24.5%): `severalHours` 28.1%, `moreThanTenHours` 23.4%, `lessThanAnHour` 17.8%, `aboutAnHour` 15.6%, `lessThanFifteenMinutes` 15.1%

**Значение для scoring**: отчёт с `moreThanTenHours` весомее чем `lessThanFifteenMinutes`.

---

## 4. Fault fields (87.5% покрытие)

| Поле | % Yes | % No |
|---|---|---|
| **performanceFaults** | 11.8% | 75.7% |
| **stabilityFaults** | 8.4% | 79.1% |
| **graphicalFaults** | 6.6% | 80.9% |
| **windowingFaults** | 6.3% | 81.2% |
| **significantBugs** | 5.8% | 81.7% |
| **inputFaults** | 5.5% | 81.9% |
| **audioFaults** | 4.5% | 83.0% |
| **saveGameFaults** | 1.2% | 86.2% |

---

## 5. followUp (32.1% отчётов)

Детализация fault-полей. Два типа хранения:

**Checkbox-формат** (dict с True/False):
| Группа | Sub-keys |
|---|---|
| **audioFaults** | borked, crackling, lowQuality, missing, outOfSync, other |
| **graphicalFaults** | heavyArtifacts, minorArtifacts, missingTextures, other |
| **inputFaults** | bounding, controllerMapping, controllerNotDetected, controllerNotResponsive, drifting, inaccuracy, lag, other |
| **windowingFaults** | activatingFullscreen, fullNotFull, switching, other |
| **saveGameFaults** | errorLoading, errorSaving, other |
| **isImpactedByAntiCheat** | battleEye, easyAntiCheat, other |
| **controlLayoutCustomization** | enableGripButtons, gyro, rightTrackpad, other |

**Enum-формат** (одно значение):
| Группа | Значения | Кол-во |
|---|---|---|
| **performanceFaults** | `slightSlowdown` (26,833), `significantSlowdown` (14,213) | |
| **stabilityFaults** | `occasionally` (14,284), `frequentCrashes` (8,314), `notListed` (6,600) | |

---

## 6. customizationsUsed (11.0% отчётов)

| Sub-key | Кол-во True |
|---|---|
| **configChange** | 13,195 |
| **protontricks** | 10,251 |
| **customProton** | 6,308 |
| **notListed** | 6,175 |
| **winetricks** | 4,926 |
| **protonfixes** | 2,847 |
| **lutris** | 2,485 |
| **mediaFoundation** | 2,133 |
| **customPrefix** | 757 |
| **native2Proton** | 245 |

---

## 7. launchFlagsUsed (3.4% отчётов)

| Flag | Кол-во True |
|---|---|
| **useWineD3d11** | 4,384 |
| **disableEsync** | 3,305 |
| **enableNvapi** | 2,803 |
| **disableFsync** | 1,383 |
| **useWineD9vk** | 820 |
| **largeAddressAware** | 669 |
| **disableD3d11** | 522 |
| **hideNvidia** | 269 |
| **gameDrive** | 258 |

**Важно**: `disableEsync`, `enableNvapi`, `disableFsync`, `disableD3d11`, `gameDrive` — не были в первоначальной схеме.

---

## 8. notes (88.3% отчётов имеют хотя бы одну заметку)

24 различных sub-keys. Основные:

| Sub-key | Покрытие | Ср. длина |
|---|---|---|
| **notes.verdict** | 74.3% | 54 chars |
| **notes.extra** | 10.0% | 196 chars |
| **notes.performanceFaults** | 10.0% | 106 chars |
| **notes.customizationsUsed** | 7.1% | 149 chars |
| **notes.stabilityFaults** | 6.8% | 109 chars |
| **notes.significantBugs** | 5.7% | 131 chars |
| **notes.graphicalFaults** | 5.5% | 100 chars |
| **notes.windowingFaults** | 5.3% | 95 chars |
| **notes.inputFaults** | 4.6% | 103 chars |
| **notes.audioFaults** | 3.7% | 86 chars |
| **notes.launchFlagsUsed** | 2.7% | 106 chars |

### responses.concludingNotes

- **Непустых**: 112,108 (32.2%), **Ср. длина**: 174.6 chars, **Макс**: 2,500 chars

**Вывод для LLM**: основные текстовые источники для экстракции — `concludingNotes` (32%), `notes.extra` (10%), `notes.customizationsUsed` (7%), `notes.verdict` (74%, но короткие — 54 chars).

---

## 9. responses.launchOptions

- **Непустых**: 48,282 (13.8%), **Уникальных**: 16,822
- Топ: `gamemoderun %command%` (6,442), `PROTON_USE_WINED3D=1 %command%` (1,599), `gamemoderun mangohud %command%` (1,141)
- Содержит: env vars, аргументы, обёртки (gamescope, mangohud, prime-run), `SteamDeck=1`

**Решение**: LLM по 16K уникальным строкам → `launch_options_parsed`.

---

## 10. Steam Deck (13% отчётов — ~45K)

| Поле | Покрытие | Значения |
|---|---|---|
| **batteryPerformance** | 13.0% | `no` 83.9%, `yes` 16.1% |
| **readability** | 13.0% | `no` 87.2%, `yes` 12.8% |
| **didChangeControlLayout** | 13.0% | `no` 81.1%, `yes` 18.9% |
| **controlLayout** | 2.4% | community 30%, official 22.6%, keyboardAndMouse 18.5%, gamepadWithMouseTrackpad 10%, + ещё 5 |
| **controlLayoutCustomization** | 2.4% | `yes` 55.4%, `no` 44.6% |
| **frameRate** | 0.1% (376) | `gt60` 54.5%, `30to60` 39.6%, `20to30` 5.1% |

---

## 11. Launcher (76.3% покрытие)

- **launcher**: `steam` 97.5%, `notListed`, `lutris`, `bottles`, `gamehub`
- **secondaryLauncher** (13%): `no` 95.7%, `yes` 4.3%

---

## 12. Multiplayer

| Поле | Покрытие | Значения |
|---|---|---|
| **isMultiplayerImportant** | 9.2% | `yes` 59%, `no` 41% |
| **localMultiplayerAttempted** | 27.6% | `no` 91.8%, `yes` 8.2% |
| **localMultiplayerPlayed** | 2.3% | `yes` 95.5%, `no` 4.5% |
| **localMultiplayerAppraisal** | 2.2% | excellent 86%, good 7.9%, awful, acceptable, weak |
| **onlineMultiplayerAttempted** | 27.6% | `no` 50.2%, `yes` 49.8% |
| **onlineMultiplayerPlayed** | 13.8% | `yes` 94.3%, `no` 5.7% |
| **onlineMultiplayerAppraisal** | 13.4% | excellent 76.8%, good 13.8%, awful, acceptable, weak |

---

## 13. Anti-cheat

- **isImpactedByAntiCheat** (9.2%): `no` 92.4%, `yes` 7.6%
- **followUp.isImpactedByAntiCheat**: содержит конкретный тип — `battleEye`, `easyAntiCheat`, `other`

---

## 14. Прочие поля

| Поле | Покрытие | Значения |
|---|---|---|
| **extra** | 24.4% | `no` 71.3%, `yes` 28.7% |
| **appSelectionMethod** | 1.8% | `libraryLookup` 63.4%, `manual` 36.6% |
| **answerToWhatGame** | 100% | = app.steam.appId (дублирование) |

**timestamp**: от 2019-10-29 до 2026-03-02.

---

## Сводка решений по нормализации

| Поле | Метод | Обоснование |
|---|---|---|
| gpu (35K уник.) | LLM по уникальным | Покроет nouveau, APU, будущие GPU |
| cpu | LLM по уникальным | Покроет будущие CPU |
| ram | regex `(\d+)` | Стабильный формат |
| kernel | regex `(\d+\.\d+[\.\d]*)` | Стабильный формат |
| protonVersion | regex + trim | Стабильный формат версий |
| launchOptions (16K уник.) | LLM по уникальным | Сложная структура, нужен семантический парсинг |
| xWindowManager | импортируем as-is | Чистые данные, полезно для Wayland/X11 |
| steamRuntimeVersion | игнорируем | Мусор |

---

## 15. Внешние API (разведка)

### Steam Store API
- Endpoint: `GET https://store.steampowered.com/api/appdetails?appids={id}`
- Всегда HTTP 200, даже для несуществующих/удалённых игр
- Non-existent/delisted: `{"success": false}` — нельзя отличить причину
- Ответ 15-34 KB, 34-39 ключей в `data`
- `release_date.date` — человекочитаемый формат ("9 Dec, 2020"), НЕ ISO-8601 — нужен парсинг
- `platforms.linux` — bool
- Нет явных rate-limit headers, но рекомендуется 1 req/sec
- `Cache-Control: public, max-age=3600`

### PCGamingWiki Cargo API
- Batch по OR в WHERE работает — 5 игр за 1 запрос (~616ms)
- **Steam AppID может быть comma-separated**: "1091500,1495710,2060310" — нужен split
- **Duplicate rows от JOIN**: CS2 вернул 2 строки (Source 2 + Source) — нужна дедупликация
- **Engines с префиксом**: "Engine:REDengine" — нужен strip "Engine:"
- **Vulkan**: возвращает "true" (строка), не версию
- Non-existent: HTTP 200, `{"cargoquery": []}` — пустой результат
- Нет rate-limit headers

### AreWeAntiCheatYet
- 1166 записей, 681 с Steam ID, 485 без
- `storeIds` — dict: `{"steam": "730"}` или `{"epic": {...}}`
- Status distribution: Broken 643, Running 275, Supported 194, Denied 52, Planned 2
- Файл ~450KB, CDN GitHub — можно кешировать локально
- Anticheats: "VAC", "Easy Anti-Cheat", "BattlEye", etc.
