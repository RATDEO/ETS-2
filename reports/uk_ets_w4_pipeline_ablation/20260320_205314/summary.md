# W4 Pipeline Ablation

Generated: 2026-03-20T20:53:25.508674

## Goal
- Isolate why the old `W4` number (`~58.98`) is so much lower than the corrected UK-specific `W4` numbers (`~101-103`).
- Hold the split fixed at `2021-10-16 / 2022-10-16 / 2023-06-20`.
- Toggle weather, UK auction proxy pack, and training regime separately on the corrected UK data root.

## Results

| candidate                            | source   | objective                                                                                            | run_dir                                                         |   path_mse |       h1 |      h5 |      h20 |     h30 |
|:-------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|-----------:|---------:|--------:|---------:|--------:|
| old_reference_generic_data           | imported | Original repeated-holdout W4 reference from the old generic Data tree.                               | /Users/davidwilkinson/Desktop/ETS 2/runs/20260316_225915_798bce |     58.979 |  4.71426 | 24.539  |  75.9278 | 101.185 |
| new_reference_current16              | imported | Corrected UK-data-root W4 with new training and current16 feature slate.                             | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200110_362947 |    102.833 | 11.3908  | 26.9551 | 127.085  | 218.574 |
| new_reference_energy16               | imported | Corrected UK-data-root W4 with new training and engineered energy16 feature slate.                   | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200137_e26524 |    101.735 | 11.4088  | 27.1403 | 124.654  | 216.043 |
| old_train_fullpacks_current16        | rerun    | Recreate the old training regime on the corrected UK data root with weather and proxy packs enabled. | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205314_9477be |    105.966 | 10.1602  | 23.6144 | 135.654  | 242.566 |
| old_train_no_weather_current13       | rerun    | Old training regime on corrected UK data, but with weather disabled.                                 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205317_a5e4de |    121.163 | 10.247   | 24.1333 | 154.31   | 290.355 |
| old_train_no_proxy_current14         | rerun    | Old training regime on corrected UK data, but with the UK auction proxy pack disabled.               | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205318_62d01e |    106.018 | 10.1492  | 23.5954 | 135.739  | 242.856 |
| old_train_no_weather_proxy_current11 | rerun    | Old training regime on corrected UK data, with both weather and proxy pack disabled.                 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205320_6dc5d8 |    121.219 | 10.2352  | 24.1075 | 154.388  | 290.716 |
| new_train_no_weather_current13       | rerun    | New regularized Huber training on corrected UK data, with weather disabled.                          | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205321_f5d355 |    102.373 | 11.4121  | 27.0988 | 126.184  | 217.459 |
| new_train_no_proxy_current14         | rerun    | New regularized Huber training on corrected UK data, with proxy pack disabled.                       | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205322_d7d80a |    102.879 | 11.3797  | 26.9841 | 127.154  | 218.668 |
| new_train_no_weather_proxy_current11 | rerun    | New regularized Huber training on corrected UK data, with both weather and proxy pack disabled.      | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_205324_ecfc47 |    102.42  | 11.4012  | 27.1288 | 126.255  | 217.556 |
