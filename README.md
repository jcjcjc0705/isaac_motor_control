# Isaac Motor Control - 馬達動態系統辨識

用神經狀態空間模型（NSSM）學習「馬達 + 連桿」系統的動態行為。給定一段力矩序列，
模型能以開迴路方式預測馬達與連桿的位置和速度軌跡，可作為後續控制器設計或 MPC 的
可微分 plant model。

模擬器只負責無阻尼的剛體積分；致動器的摩擦與激勵訊號都由這個 repo 提供，合成後
以一個命令送出。模型學到的因此是「無阻尼機構」的動態，之後把 `actuator.py` 換成
真實馬達的學習模型時，其餘部分不需改動。

## 流程總覽

```
Isaac Sim (isaac_motor_usd/*.usd)   無阻尼剛體，物理步 30 Hz、graph tick 60 Hz
        |  ROS 2: /joint_command 送兩個關節的 effort，/joint_states 回傳狀態
        v
speed_control/actuator.py       每個關節的摩擦
speed_control/data_collection.py  激勵訊號、安全保護、錄製
        |
        v  data/train.csv, data/test.csv
plot_data.py        檢查資料品質與是否撞到行程極限
        |
        v
train.py            訓練串聯式 NSSM
        |
        v  models/*.pth, scalers/*.json, runs/ (TensorBoard)
visualize_model.py  開迴路預測並與真值比對
        |
        v  plots/pred_*.png
```

## 環境需求

- ROS 2
- Isaac Sim（需啟用 ROS 2 Bridge）

### 兩個直譯器

`rclpy` 是編譯過的 C 擴充，檔名帶有 Python ABI 標籤，只能被 ROS 2 建置時所用的
那個 Python 版本載入。用其他版本 import 會得到：

```
ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
```

因此收資料與訓練分開跑在兩個環境。專案的模組已依此切分，訓練端完全不 import ROS：

| 用途 | 腳本 | 直譯器 | 依賴來源 |
|---|---|---|---|
| 收資料、增益校正 | `ros2 run speed_control ...` | ROS 2 所用的 Python | numpy / scipy / pandas，由系統套件管理員安裝 |
| 訓練、可視化、資料檢視 | `train.py`, `visualize_model.py`, `plot_data.py` | 任意，例如 conda | `requirements.txt` 或 `environment.yml` |

查看 ROS 2 對應哪個 Python：

```bash
ls -d /opt/ros/$ROS_DISTRO/lib/python3*/
```

ROS 端不需要任何 pip 套件。訓練端環境：

```bash
conda env create -f environment.yml
conda activate motor-control
# 或者
pip install -r requirements.txt
```

建議用兩個獨立的終端機：一個 source ROS 但不啟用 conda，另一個啟用 conda 但不
source ROS。`source setup.bash` 會把 ROS 的 site-packages 加進 `PYTHONPATH`，
而 conda 的 Python 會照單全收，混在一起容易出現難查的衝突。

兩邊透過 `data/` 目錄的 CSV 交換資料。**所有指令都從 repo 根目錄執行**，因為
config 中的路徑是相對路徑。各腳本啟動時都會印出實際使用的絕對路徑。

TensorBoard 為選用；未安裝時訓練仍可進行，只是不會記錄 scalar。

## Isaac 場景設定

開啟 `isaac_motor_usd/` 下的場景，Action Graph 需要這些節點：

| 節點 | 設定 |
|---|---|
| `OnPlaybackTick` | 驅動下面兩個分支 |
| `ROS2SubscribeJointState` | **`queueSize` 設為 1** |
| `IsaacArticulationController` | `targetPrim` 指向帶 `ArticulationRootAPI` 的 prim |
| `ROS2PublishJointState` | 時戳接 `IsaacReadSimulationTime` |

場景另外需要：

- 每個關節的 `drive:angular:physics:damping` 與 `physxJoint:jointFriction` 都設為 **0**
- **物理步 30 Hz**（Window → Physics Stage Settings 的 Time Steps Per Second）
- **`timeCodesPerSecond` 60**（stage 的 layer metadata）

### 頻率的關係

```
物理步 30 Hz ─┬─ graph tick 60 Hz ─→ 訊息 60 Hz（一半是重複的狀態，收集器會過濾）
              │                        ↓
              │                     有效狀態 30 Hz ─→ 控制週期 30 Hz，dt = 1/30
              └─ 死時間 = 1 個物理步 = 33.3 ms = 1 個控制週期
```

**graph tick 必須是物理步的兩倍。** 一比一時死時間會變成兩個物理步；
`record_decimation` 與 `dt` 也是依此設定的，改動任一項時三者要一起調整。

手動建立 PhysicsScene 時**務必明確設定重力**（`gravityMagnitude` 9.81、
`gravityDirection` (0,0,-1)）。stage 沒有寫 `metersPerUnit`，預設會把重力換算成
981，機構會直接失控。

## 所有參數都在 config.py

本專案的所有腳本都**不接受命令列參數**。要調整任何設定，請編輯：

```
src/speed_control/speed_control/config.py
```

該檔案是唯一的設定來源，資料收集端（ROS）與訓練端（PyTorch）共用同一份。
`ExperimentConfig` 是 frozen dataclass，需要臨時變體請用 `dataclasses.replace`。

| 分組 | 代表參數 | 說明 |
|---|---|---|
| 路徑 | `data_file`, `test_file`, `model_dir` | 模型與 scaler 檔名由 `data_file` 自動推導 |
| 回合結構 | `episode_len`, `reset_len`, `total_episodes`, `dt`, `record_decimation` | 一列 = 一個物理步 = 一個控制週期 |
| CSV 欄位 | `input_cols`, `target_cols` | 模型輸入是兩個關節實際施加的 effort |
| 模型 | `state_dim`, `history_window` | 狀態階數與堆疊的歷史命令步數 |
| 訓練 | `batch_size`, `learning_rate`, `epochs`, `val_ratio`, `seed`, `deterministic` | |
| 激勵訊號 | `signal_mix`, `signal_ranges`, `fade_in_steps` | 振幅不是設定值，由規劃器依預測擺幅推算 |
| 安全極限 | `hard_limit`, `abort_limit`, `planner_safe_limit`, `planner_lookahead_limit` | 見「安全機制」 |
| 機構增益 | `effort_to_pos_gain`, `plant_time_constant` | 決定規劃器的預測 |
| **致動器模型** | `b_viscous_*`, `b_coulomb_*`, `inertia_*`, `max_damping_torque` | 見「致動器模型」 |
| 歸零控制 | `reset_kp`, `reset_kd`, `max_effort` | |
| 增益校正 | `calib_*` | 模式 3 的施力範圍與中止角度 |
| 可視化 | `viz_train_episodes`, `viz_test_start`, `viz_channels` | |
| 資料檢視 | `inspect_episode_id` | plot_data.py 每份檔案要畫哪個回合 |

## 操作步驟

步驟 1 至 4 在 ROS 終端機執行，步驟 5 至 7 在 conda 終端機執行。

### 1. 建置 ROS package

```bash
colcon build --packages-select speed_control
source install/setup.bash
```

### 2. 啟動 Isaac Sim

依「Isaac 場景設定」確認組態後按 Play，然後確認 topic 已連通：

```bash
ros2 topic echo /joint_states --once
```

### 3. 校正 effort_to_pos_gain（換機構或改致動器係數時必做）

```bash
ros2 run speed_control data_collector     # 選 3
# 或直接執行
ros2 run speed_control gain_calibration
```

持續施加一個固定力矩，等關節停下後讀取重力與力矩平衡的角度，再把力矩加一階，
直到某個關節到達 `calib_limit`。輸出兩個值：

- **equilibrium gain** —— 平衡角度的斜率
- **peak gain** —— 含途中過衝的斜率，這是 `effort_to_pos_gain` 要用的值

### 4. 蒐集資料

```bash
ros2 run speed_control data_collector     # 選 1 收訓練資料，選 2 收測試資料
```

- 選 1：依 `signal_mix` 產生 `total_episodes` 個回合，寫入 `data_file`
- 選 2：以 `test_signal_type` 產生 `test_episodes` 個長回合，寫入 `test_file`

兩種模式都會先問 `effort_to_pos_gain`，填入步驟 3 量到的 peak gain，
直接按 Enter 則採用 config 的預設值。接著會印出該增益對應的行程：

```
  effort_to_pos_gain   : 1.013 rad per unit effort
  plant_time_constant  : 0.30 s
  travel aimed at      : 1.150 rad (65.9 deg) at full travel share, against a 1.570 rad limit
  held effort for that : 1.135 effort units; faster waveforms are given more, up to 2.50
```

收集開始前 node 會先等機構完全靜止。兩種模式都會直接覆寫目標檔案。

存檔時印出的這一行應該是 **1.00 步**：

```
  Command delay            : 1.00 steps (33 ms) between input_u and effort_motor
```

數值更大代表命令在某處排隊，檢查 graph 裡 `ROS2SubscribeJointState` 的
`queueSize` 是不是 1。

### 5. 檢查資料品質

```bash
python3 plot_data.py
```

把 `data/` 內每一份 CSV 都讀過一次，各自輸出超限統計與 `data/<檔名>_analysis.png`。
讀不動、沒有資料列、或缺少必要欄位的檔案會印出原因並跳過。

### 6. 訓練

```bash
python3 train.py
tensorboard --logdir runs
```

輸出最佳模型到 `models/`、scaler 到 `scalers/`、TensorBoard log 到 `runs/`。

### 7. 檢視預測結果

```bash
python3 visualize_model.py
```

對 `viz_train_episodes` 指定的訓練回合與 `test_file` 的指定切片做整段開迴路預測
（初始狀態為零，只餵命令序列），輸出到 `plots/pred_*.png`。

## 目錄結構

```
isaac_motor_control/
├── README.md
├── requirements.txt               訓練端依賴（pip）
├── environment.yml                訓練端依賴（conda）
├── train.py                       訓練驅動程式
├── visualize_model.py             開迴路預測與繪圖
├── plot_data.py                   資料品質檢查
├── isaac_motor_usd/               Isaac Sim 場景
├── isaac_scripts/
│   └── actuator_model.py          致動器模型的模擬器內版本（替代方案）
└── src/speed_control/speed_control/
    ├── config.py                  唯一設定來源
    ├── actuator.py                致動器模型：各關節的摩擦
    ├── signals.py                 激勵訊號產生與回合排程（不依賴 ROS）
    ├── joint_state.py             /joint_states 解析與 /joint_command 組裝
    ├── node_runner.py             node 啟動與關閉
    ├── data_collection.py         資料收集 node
    ├── limit_test.py              增益校正 node
    ├── features.py                特徵建構與正規化，訓練與推論共用
    ├── model.py                   NSSM 與 CascadedSystem
    ├── dataset.py                 回合資料集與分層切分
    └── metrics.py                 R2 與 MSE
```

## 致動器模型

`actuator.py` 為每個關節提供摩擦：

```
τ = -b_viscous · ω / (1 + b_viscous · dt / inertia) - b_coulomb · sign(ω)
```

分母的修正等同於隱式步的結果，讓係數可以取得比顯式形式更大。力矩會與激勵合成，
兩個關節各送一個值到 `/joint_command`；連桿的鉸鏈沒有激勵，但摩擦一樣走這條路。

**係數受物理步長限制。** 太大時機構會以物理步率的一半振盪且永不停止，而且上限是
**兩個關節合起來**的性質 —— 把兩者同時加倍會振盪，單獨加倍其中一個則不會。

改動係數、慣量或物理步長之後，一定要做這個檢查：**推動機構後放手，確認運動會衰減**。

`isaac_scripts/actuator_model.py` 是同一個模型的模擬器內版本，把程式碼烘進 USD 的
Script Node，以 body torque 施力。兩者**只能擇一啟用**，同時啟用會讓摩擦變成兩倍。

## 資料格式

| 欄位 | 說明 |
|---|---|
| `time_actual` | 模擬器時鐘，從收集開始起算的秒數 |
| `time_ideal` | `global_step * dt`，理想時間軸 |
| `episode_id` | 回合編號，從 0 起 |
| `input_u` | 送出的激勵力矩，已乘上淡入窗，不含摩擦；歸零控制器接管期間為 0 |
| `effort_motor` | 模擬器回報實際施加在馬達上的力矩，模型輸入 |
| `effort_joint1` | 同上，連桿鉸鏈，模型輸入 |
| `signal_type` | 該回合的激勵訊號種類，訓練時用於分層切分 |
| `pos_motor`, `vel_motor` | 馬達位置與速度 |
| `pos_joint1`, `vel_joint1` | 第一連桿位置與速度 |
| `pos_joint2`, `vel_joint2` | 第二連桿，僅在場景提供時才有 |

一個物理步記錄一列。`effort_*` 與同列的 `pos`/`vel` 出自同一筆訊息，描述同一個
瞬間；`input_u` 是該瞬間送出的激勵，它在**下一列**才會作用到關節上。

位置在解析時就折疊到 ±π。

## 模型架構

`CascadedSystem` 由兩段 `NSSM` 串聯：

```
u_seq --> [stage_motor] --> pred_motor (pos_motor, vel_motor)
   |                             | detach()
   +---------- concat -----------+
                 |
                 v
          [stage_joint] --> pred_joint (pos_joint1, vel_joint1)
```

連桿段會看到馬達段的預測，但該張量經過 `detach()`，兩段各自獨立學習。

單段 `NSSM` 為離散時間非線性狀態空間模型：

```
y_t     = g(x_t) + D u_t
x_{t+1} = (1 - a) x_t + a f(x_t, u_t)
```

`a = sigmoid(alpha_raw)` 是可學習的積分洩漏率，初始值約 0.12。這個殘差形式
（等同前向歐拉積分）讓狀態預設變化緩慢，是能對數百步做 BPTT 而不發散的關鍵。

輸入特徵為 `[u, u 的一階差分]` 再堆疊過去 `history_window` 步，因此每個時間點的
輸入維度為 `input_dim * 2 * history_window`。

## 安全機制

行程保護分四層，由寬到緊：

1. **`planner_safe_limit`** —— 規劃器的位置上限。每個回合的波形都會被縮放到
   「預測擺幅 = `travel_fraction` × 安全帶」，排程把 `travel_fraction` 掃過
   0.4 / 0.6 / 0.8 / 1.0。
2. **`planner_lookahead_limit`** —— 規劃器的速度上限。另外預測一次「位置 +
   `lookahead` 秒的速度」，超過就再縮。只有快速的波形會被它壓低。
3. **`abort_limit`** —— 執行期即時保護。每步做前瞻預測，一旦當前或預測位置越界，
   立刻放棄該回合剩餘的激勵，把馬達交給歸零控制器。
4. **`hard_limit`** —— 機構行程極限，用於事後統計。存檔時會報告有多少列、
   哪些回合曾經超過，以及是否出現非有限值。

### 激勵的淡入

波形不會從靜止直接跳到它的振幅。送出的命令會乘上一個長度為 `fade_in_steps`
的升餘弦窗，權重與斜率都由 0 起算：

```
w(i) = 0.5 × (1 − cos(π i / fade_in_steps))     i < fade_in_steps
w(i) = 1                                        i >= fade_in_steps
```

窗在兩個時機重新開始：每個回合的激勵段開頭，以及歸零控制器把致動器交還的那一步。
第二個時機同樣必要 —— 中止後波形會從它當下的相位接回，該處的振幅通常不是零。

`fade_in_steps` 要取得比物理步能承載的最高頻率的週期長數倍。太短則一步之內的
力矩變化足以把關節推上前瞻門檻，該回合會立刻被中止層接管；設為 0 則關閉淡入。

淡入只作用在激勵起步與接回處，不影響回合中段，方波類訊號的銳利反轉會原樣送出。

### 擺幅怎麼預測

把命令先用 `plant_time_constant` 做一階低通再取峰值：

```
預測擺幅 = effort_to_pos_gain × max| lowpass(u, plant_time_constant) |
預測前瞻 = effort_to_pos_gain × max| lowpass(u) + lookahead × d/dt lowpass(u) |
```

低通對持續的準位沒有衰減，對快速交替的訊號則按它停留的時間比例衰減。所以
**激勵振幅不是設定值**：波形以單位振幅產生，再縮放到上面兩個預測都合格為止，
最後截在 `max_effort`。

`plant_time_constant` 的校法是拿一份收好的資料，比較預測擺幅與實際擺幅，
取比值中位數為 1 的那個 tau。

預測有散度，安全帶與中止門檻之間的差距是留給它的。

中止層用的是線性外推（`位置 + lookahead × 速度`），對會提早反轉的訊號偏保守。
PRBS 的方波反轉最容易踩到它：關節的瞬時速度足以讓外推越界，但阻尼與下一次反轉
會在 `lookahead` 秒之前就把它停住，實際位置離 `abort_limit` 還很遠。代價是該
回合剩餘的激勵被歸零控制器接管。要收回這部分，把中止層的前瞻改成與規劃器相同的
低通預測；要更保守，則調低 `planner_safe_limit`。

## 激勵的頻率覆蓋

激勵要涵蓋機構的模態，否則模型在沒被激發過的頻段上沒有約束。檢查方式是對每個回合的
`input_u` 取 FFT，看功率落在哪些頻帶。MULTISINE 不看 `signal_ranges`，它從
`signals.py` 的 `MULTISINE_BANDS` 各抽一個音。

## 驗證集切分

`stratified_split` 依 CSV 的 `signal_type` 欄位分層，每種訊號各自以固定間隔抽出
驗證回合，因此無論 `val_ratio` 設成多少，各種訊號在驗證集中都會等比例出現。

若讀到的 CSV 沒有 `signal_type` 欄位，程式會印出警告並依 `signal_mix` 的交錯順序
回推種類。該回推只對交錯排程產生的資料有效。

## 注意事項

- **ROS package 名稱為 `speed_control`，控制方式為力矩。** `colcon build` 的產物
  名稱與 `ros2 run` 的呼叫方式都以這個名稱為準。
- **資料收集可重現。** 訊號產生器會在啟動時以 `config.seed` 設定隨機種子。
- **訓練預設不保證位元級可重現。** 需要嚴格重現時把 `deterministic` 設為 `True`。
- **模擬器若發散（`/joint_states` 出現 NaN），收集會立刻停止並捨棄該輪資料。**
  存檔時也會單獨報告非有限值的列數。
- **控制迴路一律掛在訊息流上，不要掛在計時器上。** 即時率不為 1 時，以牆上時鐘
  計時的迴路會與模擬時間脫節。
- **`data/`、`models/`、`runs/`、`plots/`、`scalers/` 都在 .gitignore 內**，
  跨機器搬移時需另外複製，且 `scalers/*.json` 與 `models/*.pth` 必須成對使用。
- **`scipy.interpolate.interp1d` 在新版 SciPy 中標記為 legacy**，未來若被移除，
  三次樣條可換成 `scipy.interpolate.CubicSpline`。使用處為 `signals.py` 的
  SMOOTH_NOISE 產生器。
