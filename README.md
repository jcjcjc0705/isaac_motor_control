# Isaac Motor Control - 馬達動態系統辨識

用神經狀態空間模型（NSSM）學習「馬達 + 連桿」系統的動態行為。給定一段力矩命令序列，
模型能以開迴路方式預測馬達與連桿的位置和速度軌跡，可作為後續控制器設計或 MPC 的
可微分 plant model。

## 流程總覽

```
isaac_scripts/actuator_model.py  把致動器阻尼寫進 USD，由模擬器每個物理步施加
        |
        v
Isaac Sim (isaac_motor_usd/*.usd)
        |  ROS 2: 發布 effort 到 /joint_command，訂閱 /joint_states
        v
data_collection.py  以設計好的激勵訊號驅動系統並錄製回應
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

阻尼由模擬器內部施加，收資料的 node 只負責送出激勵命令。ROS 這條路徑有往返延遲，
把速度回饋放在外面會讓阻尼在連桿的自然頻率附近變成能量來源。

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

兩邊透過 `data/` 目錄的 CSV 交換資料，不需要共用直譯器。**所有指令都從 repo
根目錄執行**，因為 config 中的路徑是相對路徑。各腳本啟動時都會印出實際使用的
絕對路徑，可據此確認。

TensorBoard 為選用；未安裝時訓練仍可進行，只是不會記錄 scalar。

## 所有參數都在 config.py

本專案的所有腳本都**不接受命令列參數**。要調整任何設定，請編輯：

```
src/speed_control/speed_control/config.py
```

該檔案是唯一的設定來源，資料收集端（ROS）與訓練端（PyTorch）共用同一份，
因此兩邊對回合長度、特徵寬度、安全極限的認知不可能不一致。

`ExperimentConfig` 是 frozen dataclass。若需要臨時變體，請用
`dataclasses.replace(cfg, ...)` 產生新實例，不要直接修改共用實例。

### 主要參數分組

| 分組 | 代表參數 | 說明 |
|---|---|---|
| 路徑 | `data_file`, `test_file`, `model_dir` | 輸入輸出位置；模型與 scaler 檔名由 `data_file` 自動推導 |
| 回合結構 | `episode_len`, `reset_len`, `total_episodes`, `dt` | 每回合 = 激勵段 + 歸零段，`seq_len` 為兩者之和 |
| 取樣 | `record_decimation`, `progress_every` | 每幾筆 `/joint_states` 記錄一列、每幾列印一次進度 |
| CSV 欄位 | `input_cols`, `target_cols` | `output_dim` 由 `target_cols` 長度自動推導 |
| 模型 | `state_dim`, `history_window` | 狀態階數與堆疊的歷史命令步數 |
| 訓練 | `batch_size`, `learning_rate`, `epochs`, `val_ratio`, `seed`, `deterministic` | `deterministic=True` 會開啟 cuDNN 決定性，較慢但可重現 |
| 激勵訊號 | `signal_mix`, `signal_ranges` | 各訊號種類的配比與參數範圍；振幅不是設定值，由規劃器依預測擺幅推算 |
| 安全極限 | `hard_limit`, `abort_limit`, `planner_safe_limit`, `planner_lookahead_limit` | 見下方「安全機制」 |
| 機構增益 | `effort_to_pos_gain`, `plant_time_constant` | 由模式 3 與資料量得，決定規劃器的預測 |
| 歸零控制 | `reset_kp`, `reset_kd`, `max_effort` | 歸零段與中止後把馬達交給誰處理 |
| 增益校正 | `calib_start_effort`, `calib_effort_step`, `calib_effort`, `calib_limit` | 模式 3 的施力範圍與中止角度 |
| 可視化 | `viz_train_episodes`, `viz_test_start`, `viz_channels` | 要畫哪些回合、測試檔切片範圍、要畫哪些通道 |
| 資料檢視 | `inspect_file`, `inspect_episode_id` | plot_data.py 的分析對象 |

## 操作步驟

步驟 1 至 5 在 ROS 終端機執行，步驟 6 至 8 在 conda 終端機執行，
兩者都從 repo 根目錄啟動。

### 1. 建置 ROS package（ROS 終端機）

```bash
colcon build --packages-select speed_control
source install/setup.bash
```

### 2. 準備 Isaac Sim 場景

開啟 `isaac_motor_usd/` 下的場景（`motor_sim_oneJoint.usd` / `motor_sim_twoJoint.usd` /
`motor_sim_threeJoint.usd`），確認 Action Graph 具備：

- `ROS2PublishJointState`，發布 `/joint_states`
- `IsaacArticulationController`，訂閱 `/joint_command` 並以 effort 驅動
- 一個 Script Node，用來執行致動器模型（見步驟 3）

每個要模擬阻尼的關節，把 `drive:angular:physics:damping` 與
`physxJoint:jointFriction` 都設為 0，阻尼一律交給致動器模型提供。
場景中需有一個 prim 帶 `ArticulationRootAPI`。

### 3. 掛上致動器模型（換模型或改參數時才需要）

`isaac_scripts/actuator_model.py` 提供關節的黏滯阻尼、庫倫摩擦與力矩上限，
在模擬器內部每個物理步施加，回報的關節速度不受影響。

在 Action Graph 內自行拉一個 Script Node、把 tick node 接到它的 Exec In，
然後編輯檔案末端的區塊：

```python
apply_actuator(
    joint_path = "/World/Cylinder/motor",   # 右鍵 -> Copy Prim Path
    b_viscous  = 0.3,                       # 黏滯阻尼 (N*m*s/rad)
    b_coulomb  = 0.0,                       # 與速度無關的乾摩擦 (N*m)
    max_torque = 2.0,                       # 此關節的力矩上限 (N*m)
)

install(
    script_node_path = "/World/ActionGraph/script_node",
)
```

把整份檔案貼進 Window > Script Editor 執行一次，存檔（Ctrl+S），再按 Stop 與 Play。
程式碼會寫進 Script Node 的 `inputs:script`，也就是存在 USD 裡，因此執行過後這個
檔案可以搬走或刪除，場景仍能運作；要改係數或換關節時再執行一次即可。

多關節就多寫幾個 `apply_actuator(...)` 區塊，不需要的係數整行刪掉即可。
執行後每個關節會印出 `[OK]` 與解析到的 body、axis，可據此確認路徑正確。

### 4. 校正 effort_to_pos_gain（換機構時必做）

```bash
ros2 run speed_control data_collector     # 選 3
# 或直接執行
ros2 run speed_control gain_calibration
```

持續施加一個固定力矩，等關節停下後讀取重力與力矩平衡的角度，再把力矩加一階，
直到某個關節到達 `calib_limit`。角度對力矩的斜率就是增益。輸出兩個值：

- **equilibrium gain** —— 平衡角度的斜率，機構本身的增益
- **peak gain** —— 含途中過衝的斜率，這是 `effort_to_pos_gain` 要用的值

訊號產生器問的是「一段波形會把關節甩到多遠」，所以要用含過衝的 peak gain。

這個增益描述的是機構本身（連桿、質量、摩擦），改動任何一項就得重測。
激勵振幅由它推算而來，所以重測這一個數字就足以讓整輪收集留在行程內。
`plant_time_constant` 則是換過機構後用第一份收好的資料回頭校正（見「擺幅怎麼預測」）。

### 5. 蒐集資料（ROS 終端機）

```bash
ros2 run speed_control data_collector     # 選 1 收訓練資料，選 2 收測試資料
```

- 選 1：依 `signal_mix` 產生 `total_episodes` 個回合，寫入 `data_file`
- 選 2：以 `test_signal_type` 產生 `test_episodes` 個長回合，寫入 `test_file`

兩種模式都會先問 `effort_to_pos_gain`，填入步驟 4 量到的 peak gain，
直接按 Enter 則採用 config 的預設值：

```
effort_to_pos_gain [1.15]:
```

接著會印出這個增益對應的行程，可在送出第一個命令前確認合理：

```
  effort_to_pos_gain   : 1.150 rad per unit effort
  plant_time_constant  : 0.30 s
  travel aimed at      : 1.150 rad (65.9 deg) at full travel share, against a 1.570 rad limit
  held effort for that : 1.000 effort units; faster waveforms are given more, up to 2.50
```

這個值在排程建立時就要定案，所以只能在這裡問，不能等 node 起來之後再改。
收集開始前 node 會先等機構完全靜止，因為訓練是以零初始狀態展開每個回合的。
兩種模式都會直接覆寫目標檔案，不會產生時間戳檔名。

### 6. 檢查資料品質（conda 終端機）

```bash
python3 plot_data.py
```

輸出超限統計與 `data/<name>_analysis.png`（單回合速度、單回合位置、全場總覽）。

### 7. 訓練（conda 終端機）

```bash
python3 train.py
tensorboard --logdir runs
```

輸出最佳模型到 `models/`、scaler 到 `scalers/`、TensorBoard log 到 `runs/`。
把 config 的 `launch_tensorboard` 設為 `True` 可在訓練開始時自動啟動 TensorBoard。

### 8. 檢視預測結果（conda 終端機）

```bash
python3 visualize_model.py
```

對 `viz_train_episodes` 指定的訓練回合，以及 `test_file` 的指定切片，
做整段開迴路預測（初始狀態為零，只餵命令序列），輸出到 `plots/pred_*.png`。

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
│   └── actuator_model.py          致動器模型，寫入 USD 的 Script Node
└── src/speed_control/speed_control/
    ├── config.py                  唯一設定來源
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

`model.py`、`features.py`、`metrics.py` 由訓練與推論兩邊共用，兩條路徑因此建出
相同的架構、相同的特徵、相同的分數。`signals.py` 不 import ROS，可以不開模擬器
就產生並檢視波形。

## 資料格式

| 欄位 | 說明 |
|---|---|
| `time_actual` | 模擬器時鐘，從收集開始起算的秒數 |
| `time_ideal` | `global_step * dt`，理想時間軸 |
| `episode_id` | 回合編號，從 0 起 |
| `input_u` | 送出的力矩命令 |
| `signal_type` | 該回合的激勵訊號種類，訓練時用於分層切分 |
| `pos_motor`, `vel_motor` | 馬達位置與速度 |
| `pos_joint1`, `vel_joint1` | 第一連桿位置與速度 |
| `pos_joint2`, `vel_joint2` | 第二連桿，僅在場景提供時才有 |

每一筆 `/joint_states` 就是一個控制步，每 `record_decimation` 筆記錄一列，
所以命令與狀態出自同一筆訊息，一列只需要一個時間戳，不需要內插對齊。
node 跟著模擬器的時鐘走，模擬器跑得慢只會花掉更多實際時間，不會少收資料。

位置在解析時就折疊到 ±π，因此控制器、統計與存檔看到的是同一個角度。

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

連桿段會看到馬達段的預測，但該張量經過 `detach()`，所以連桿的誤差不會回傳到
馬達模型，兩段各自獨立學習。

單段 `NSSM` 為離散時間非線性狀態空間模型：

```
y_t     = g(x_t) + D u_t
x_{t+1} = (1 - a) x_t + a f(x_t, u_t)
```

`a = sigmoid(alpha_raw)` 是可學習的積分洩漏率，初始值約 0.12。這個殘差形式
（等同前向歐拉積分）讓狀態預設變化緩慢，是能對數百步做 BPTT 而不發散的關鍵。

輸入特徵為 `[u, u 的一階差分]` 再堆疊過去 `history_window` 步，
因此每個時間點的輸入維度為 `input_dim * 2 * history_window`。

## 安全機制

行程保護分四層，由寬到緊：

1. **`planner_safe_limit`（1.2 rad）** —— 規劃器的位置上限。每個回合的波形都會被
   縮放到「預測擺幅 = `travel_fraction` × 安全帶」，排程把 `travel_fraction` 掃過
   0.4 / 0.6 / 0.8 / 1.0。
2. **`planner_lookahead_limit`（1.35 rad）** —— 規劃器的速度上限。位置合格的波形仍
   可能因為太快而踩到中止層，所以另外預測一次「位置 + `lookahead` 秒的速度」，
   超過就再縮。只有快速的波形會被它壓低。
3. **`abort_limit`（1.4 rad）** —— 執行期即時保護。每步用 `lookahead` 秒做前瞻預測，
   一旦當前或預測位置越界，立刻放棄該回合剩餘的激勵訊號，把馬達交給歸零控制器。
4. **`hard_limit`（1.57 rad，90 度）** —— 機構行程極限，用於事後統計。
   存檔時會報告有多少列、哪些回合曾經超過此值，以及是否出現非有限值。

### 擺幅怎麼預測

持續施力時關節會停在重力與之平衡的角度，快速反向則來不及走到那裡。把命令先用
`plant_time_constant` 做一階低通再取峰值，就同時描述這兩端 —— 低通對持續的準位沒有
衰減，對快速交替的訊號則按它停留的時間比例衰減：

```
預測擺幅 = effort_to_pos_gain × max| lowpass(u, plant_time_constant) |
預測前瞻 = effort_to_pos_gain × max| lowpass(u) + lookahead × d/dt lowpass(u) |
```

所以**激勵振幅不是設定值**：波形以單位振幅產生，再縮放到上面兩個預測都合格為止，
最後截在 `max_effort`。慢速訊號需要的力矩接近
`(planner_safe_limit - planner_margin) / effort_to_pos_gain`，快速訊號需要更多。
這讓五種訊號都用到同一份行程，而不是讓快速訊號只擺一半。

`plant_time_constant` 的校法是拿一份收好的資料，比較預測擺幅與實際擺幅：
比值中位數為 1 的那個 tau 就是它。預測對持續施力（DC）的行為與 `effort_to_pos_gain`
的定義一致，所以 PRBS 這類訊號的上界不受這個 tau 影響。

預測會有散度（實測範圍約 0.85–1.2 倍），安全帶與中止門檻之間的差距就是留給它的。
MULTISINE 是最容易低估的一種，因為低通模型沒有共振峰而它每回合都含共振附近的頻率；
實測會偶爾踩到中止層，截掉激勵段最後幾步。若要完全避免，把 `planner_safe_limit`
降到 1.15。

歸零控制器的 `reset_kp` 與 `reset_kd` 預設為 0，也就是不施力、讓關節靠自身摩擦
滑行到停止。若機構無法自行停下再調高它們，但那會把歸零段變成一個經由模擬器往返
的 PD 迴路，且每換一次機構就要重新調。

## 驗證集切分

`stratified_split` 依 CSV 的 `signal_type` 欄位分層，每種訊號各自以固定間隔抽出
驗證回合，因此無論 `val_ratio` 設成多少，各種訊號在驗證集中都會等比例出現。

若讀到的 CSV 沒有 `signal_type` 欄位，程式會印出警告並依 `signal_mix` 的交錯順序
回推種類。該回推只對交錯排程產生的資料有效。

## 注意事項

- **ROS package 名稱為 `speed_control`，控制方式為力矩。** `colcon build` 的產物名稱
  與 `ros2 run` 的呼叫方式都以這個名稱為準。
- **資料收集可重現。** 訊號產生器會在啟動時以 `config.seed` 設定 NumPy 與
  Python 的隨機種子。
- **訓練預設不保證位元級可重現。** GPU 上 cuDNN 的演算法選擇與浮點累加順序會有
  差異。需要嚴格重現時把 config 的 `deterministic` 設為 `True`。
- **模擬器若發散（`/joint_states` 出現 NaN），收集會立刻停止並捨棄該輪資料。**
  存檔時也會單獨報告非有限值的列數，因為與 NaN 比較永遠為假，超限統計看不出來。
- **`data/`、`models/`、`runs/`、`plots/`、`scalers/` 都在 .gitignore 內**，
  不會進版控。跨機器搬移時需另外複製，且 `scalers/*.json` 與 `models/*.pth`
  必須成對使用，混用會讓預測完全錯誤。
- **`scipy.interpolate.interp1d` 在新版 SciPy 中標記為 legacy**，目前仍可使用。
  未來若被移除，三次樣條可換成 `scipy.interpolate.CubicSpline`。使用處為
  `signals.py` 的 SMOOTH_NOISE 產生器。
