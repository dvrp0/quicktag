# Revamp visual verification

Canonical fixture: BR33 Volley Rifle **Vibrant Sport** model `80A9FF17`, owner `80A7AA89`, Deluxe (blue) **Cold Vigilance Scope** `80A60FED`, and Deluxe **Impulse Brake** `80A60608`. Fixed camera uses yaw −24° and pitch +14°.

Capture current renderer before a major revamp step:

```powershell
.\scripts\revamp_visual_verify.ps1 -Mode Capture
```

Verify after implementation:

```powershell
.\scripts\revamp_visual_verify.ps1 -Mode Verify
```

Harness renders final, base-color, diffuse, AO, specular, pre-tone HDR, and normal outputs at 1024×640 using fixed camera/environment values. Verify mode writes per-pass metrics to `target/quicktag-model-probe/*.metrics.json` and fails when default tolerance is exceeded:

- RGB mean absolute error: `1.5`
- p99 RGB channel delta: `8`
- pixels with any RGB delta greater than `8`: `2%`

Override thresholds only for diagnosed cross-GPU variance:

```powershell
$env:QUICKTAG_PROBE_MAX_MAE = "2.0"
$env:QUICKTAG_PROBE_MAX_P99 = "10"
$env:QUICKTAG_PROBE_MAX_CHANGED_FRACTION = "0.03"
.\scripts\revamp_visual_verify.ps1 -Mode Verify
```

Script disables default features and enables only `wordlist`; `xg.dll` is not loaded.
