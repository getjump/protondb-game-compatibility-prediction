# AWACY: неиспользуемые данные и потенциал

## Что есть в games.json (полный формат)

```json
{
  "url": "https://www.fortnite.com/",
  "name": "Fortnite",
  "native": false,
  "status": "Denied",
  "anticheats": ["Easy Anti-Cheat"],
  "notes": [
    ["Works on Xbox-Cloud", "https://..."]
  ],
  "updates": [
    {
      "name": "Not confident in anti-cheat",
      "date": "Feb 07, 2022, 6:59 AM GMT+2",
      "reference": "https://twitter.com/..."
    },
    {
      "name": "Removed BattlEye",
      "date": "Jun 11, 2024, 5:34 PM GMT+1",
      "reference": "https://x.com/..."
    }
  ],
  "storeIds": {"steam": "730", "epic": {...}},
  "slug": "fortnite",
  "dateChanged": "2024-10-04T17:23:46.000Z"
}
```

## Что мы сейчас парсим

Из всего JSON берём только 2 поля:
```python
AWACYData(
    anticheats=["Easy Anti-Cheat"],  # list[str]
    status="Denied",                  # str
)
```

## Что теряем

### 1. `native` (boolean)
**Значение:** Есть ли нативная Linux-версия.
**Почему полезно:** Дублирует `has_linux_native` из Steam, но AWACY более аккуратно отслеживает для anticheat-игр. Может быть полезно для cross-validation.
**ML ценность:** Низкая — уже есть из Steam.

### 2. `updates` (list of events)
**Значение:** Хронология изменений anticheat-статуса с датами и ссылками.
**Что можно извлечь:**
- `last_update_date` — когда последний раз менялся статус
- `updates_count` — сколько раз менялся статус (нестабильность)
- `has_recent_update` — менялось ли за последние 6 месяцев
- `status_trajectory` — улучшение (Broken→Running→Supported) или деградация

**ML ценность:** Medium. Игра где статус недавно изменился может иметь нестабильную совместимость. Но покрытие ~681 игр из 31K (2.2%).

### 3. `notes` (list of [text, url])
**Значение:** Важные замечания (workarounds, cloud-gaming, specific versions).
**Что можно извлечь:**
- `has_workaround_note` — есть ли workaround → tinkering signal
- NLP на текст notes для извлечения типа заметки

**ML ценность:** Low. Слишком мало данных, notes в основном ссылки.

### 4. `dateChanged` (ISO timestamp)
**Значение:** Когда последний раз обновлялась запись.
**Что можно извлечь:**
- `awacy_freshness_days` — актуальность данных
- Cross с `updates`: если dateChanged свежий но status старый → стабильный статус

**ML ценность:** Low.

## Почему anticheat данные не помогли раньше

1. **Покрытие 2.2%** — 681 игр в AWACY из 31K в нашей базе
2. **Anticheat ≈ borked** — игры с EAC/BattlEye в основном просто не работают. Stage 1 уже ловит это через текст ("EAC", "BattlEye"). Для Stage 2 (tinkering vs oob) — нерелевантно
3. **Categorical noise** — `anticheat` как comma-separated string = высокая кардинальность
4. **Phase 7 ablation**: ΔF1 < 0.002 для `anticheat`, `anticheat_status`

## Что стоит попробовать

### A. Обогатить AWACYData (расширить парсинг)

Расширить `AWACYData` и `load_awacy`:
```python
class AWACYData(BaseModel):
    anticheats: list[str] = Field(default_factory=list)
    status: str | None = None
    native: bool = False
    updates_count: int = 0
    last_update_date: str | None = None
    has_notes: bool = False
    date_changed: str | None = None
```

**Стоимость:** ~15 строк в `anticheat.py`. Не требует перезагрузки — тот же JSON.

### B. Feature engineering для ML

Из расширенных данных → фичи для `game_metadata`:
- `anticheat_updates_count` (INT) — кол-во изменений статуса
- `anticheat_last_update_days` (INT) — дней с последнего обновления
- `anticheat_native` (BOOL) — нативная Linux версия по AWACY
- `anticheat_status_ordinal` — ordinal encoding: Denied=0, Broken=1, Planned=2, Running=3, Supported=4

Но честно: при покрытии 2.2% эффект будет **< 0.002 F1**. Не стоит пока.

### C. Когда станет полезно

Anticheat данные станут ценными если:
1. AWACY покрытие вырастет (сейчас 681 игр, в игровом мире их тысячи с античитами)
2. Мы сфокусируемся на **per-game prediction** вместо per-report (агрегация → предсказание для новой игры)
3. Anticheat status начнёт коррелировать с tinkering (если EAC/BattlEye начнут поддерживать Proton частично → "Supported" игры = works_oob, "Running" = tinkering)

## Решение

**Сейчас:** не тратить время. При 2.2% покрытии доп. поля из AWACY не дадут measurable improvement.

**Можно сделать дёшево:** расширить парсинг `load_awacy` на `updates`, `native`, `dateChanged` и сохранять в `enrichment_cache` — на будущее, если покрытие вырастет. ~15 строк, 0 impact на ML pipeline.
