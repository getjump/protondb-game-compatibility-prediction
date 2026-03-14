# Phase 7: Feature Ablation & Cleanup

## Контекст

После Phase 6 (A1/A4/A5 target leak removal + zero-importance cleanup) модель имела 116 фич, F1=0.7332.
Проведён полный ablation всех 26 групп фич каскадной модели.

## Результаты полного ablation (116 фич baseline, F1=0.7337)

| Эксперимент | F1 macro | ΔF1 | Категория |
|---|---|---|---|
| **BASELINE** | **0.7337** | — | 116 фич |
| drop_engine(1) | 0.7379 | **+0.0042** | Вредит |
| drop_cpu_vendor(1) | 0.7336 | −0.0000 | Безразлична |
| drop_os_family(1) | 0.7334 | −0.0003 | Безразлична |
| drop_cpu_gen(1) | 0.7333 | −0.0004 | Безразлична |
| drop_report_age(1) | 0.7327 | −0.0010 | Безразлична |
| drop_ram_gb(1) | 0.7327 | −0.0010 | Безразлична |
| drop_anticheat(1) | 0.7326 | −0.0011 | Безразлична |
| drop_gpu_vendor(1) | 0.7322 | −0.0015 | Пограничная |
| drop_gpu_tier(1) | 0.7322 | −0.0015 | Пограничная |
| drop_kernel_major(1) | 0.7321 | −0.0016 | Пограничная |
| drop_genre(1) | 0.7321 | −0.0016 | Пограничная |
| drop_agg_misc(4) | 0.7321 | −0.0015 | Пограничная |
| drop_agg_deck(2) | 0.7318 | −0.0019 | Слабая польза |
| drop_game_meta(4) | 0.7319 | −0.0018 | Слабая польза |
| drop_drm(2) | 0.7317 | −0.0020 | Слабая польза |
| drop_proton_all(3) | 0.7316 | −0.0021 | Слабая польза |
| drop_gfx_api(5) | 0.7315 | −0.0022 | Слабая польза |
| drop_gpu_bools(4) | 0.7312 | −0.0024 | Полезны |
| drop_gpu_family(1) | 0.7313 | −0.0024 | Полезна |
| drop_keywords(8) | 0.7307 | −0.0030 | Полезны |
| drop_gpu_emb(16) | 0.7306 | −0.0031 | Полезны |
| drop_drivers(2) | 0.7305 | −0.0032 | Полезны |
| drop_text_meta(5) | 0.7159 | −0.0178 | Очень полезны |
| drop_variant(1) | 0.7020 | −0.0317 | Критична |
| drop_text_emb(32) | 0.6973 | −0.0364 | Критичны |
| drop_game_emb(16) | 0.6733 | −0.0604 | Самая важная |

## Решение: дропнуть "слабая польза" и ниже (32 фичи)

### Удаляемые фичи (ΔF1 ≥ −0.0022 при одиночном удалении):

**Вредит модели (1):**
- `engine` — высокая кардинальность, добавляет шум (+0.004 при удалении)

**Безразличные (7):**
- `cpu_vendor` — информация дублируется в cpu_generation
- `os_family` — почти все отчёты Linux
- `cpu_generation` — cpu_emb уже удалены, мало сигнала
- `report_age_days` — нет temporal drift
- `ram_gb` — слабый сигнал для совместимости
- `anticheat` — мало данных (177 EAC + 50 VAC из 31K игр)

**Пограничные (8):**
- `gpu_vendor` — дублируется gpu_family/gpu_emb
- `gpu_tier` — дублируется gpu_family/gpu_emb
- `kernel_major` — Proton абстрагирует ядро
- `genre` — слабый сигнал
- `total_reports`, `avg_online_mp_score`, `pct_steam_launcher`, `pct_lutris_launcher` — per-game агрегаты со слабым сигналом

**Слабая польза (16):**
- `pct_deck_battery_ok`, `pct_deck_readable` — мало данных (13% заполнение)
- `release_year`, `has_linux_native`, `is_multiplayer`, `has_mp_reports` — game metadata
- `has_denuvo`, `drm_count` — DRM features
- `proton_major`, `is_ge_proton`, `has_proton_version` — Proton version features
- 5x `graphics_api_*` — дублируется engine/game_emb

### Остаются (84 фичи):
- **GPU embeddings** (16) — важнейший HW-сигнал
- **Game embeddings** (16) — самая важная группа (ΔF1=−0.060)
- **Text embeddings** (32) — критичны для borked detection
- **variant** (1) — вторая по важности
- **gpu_family** (1), **gpu_bools** (4: is_apu, is_igpu, is_mobile, is_steam_deck)
- **drivers** (2: nvidia_driver_version, mesa_driver_version)
- **keywords** (8: mentions_crash, mentions_fix, mentions_perfect, etc.)
- **text_meta** (5: fault_notes_count, has_concluding_notes, concluding_notes_length, etc.)

## Ранее удалённые фичи (Phase 6)

### Target leaks (34 фичи):
- **A1 per-report fault booleans** (10): audio_faults...significant_bugs + fault_any + fault_count
- **A4 per-report cust/flag** (23): cust_protontricks...flag_game_drive + derived
- **A5 per-report anticheat** (1): is_impacted_by_anticheat_report

### Per-game aggregated leaks (10 фичей):
- pct_needs_winetricks, pct_needs_protontricks, pct_needs_custom_proton
- pct_uses_wine_d3d11, pct_uses_d9vk
- pct_audio_faults...pct_stability_faults

### Zero-importance / шум (23 фичи):
- github_* (5): пустые данные
- anticheat_status (1): 100% null
- driver_major (1): дублируется nvidia/mesa versions
- cpu_emb_* (16): шумные, +0.001 F1 при удалении

## Итого: 183 → 116 → 84 фичи
