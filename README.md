# Isaac Motor Control - 馬達動態系統辨識

用神經狀態空間模型（NSSM）學習「馬達 + 連桿」系統的動態行為。給定一段力矩命令序列，
模型能以開迴路方式預測馬達與連桿的位置和速度軌跡，可作為後續控制器設計或 MPC 的
可微分 plant model。

## 流程總覽

```
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
| CSV 欄位 | `input_cols`, `target_cols` | `output_dim` 由 `target_cols` 長度自動推導 |
| 模型 | `state_dim`, `history_window` | 狀態階數與堆疊的歷史命令步數 |
| 訓練 | `batch_size`, `learning_rate`, `epochs`, `val_ratio`, `seed`, `deterministic` | `deterministic=True` 會開啟 cuDNN 決定性，較慢但可重現 |
| 激勵訊號 | `amplitude`, `signal_mix`, `signal_ranges` | 各訊號種類的配比與參數範圍 |
| 安全極限 | `hard_limit`, `abort_limit`, `planner_safe_limit`, `effort_to_pos_gain` | 見下方「安全機制」 |
| 歸零控制 | `reset_kp`, `reset_kd`, `max_effort` | 把馬達拉回 0 的 PD 控制器 |
| 可視化 | `viz_train_episodes`, `viz_test_start`, `viz_channels` | 要畫哪些回合、測試檔切片範圍、要畫哪些通道 |
| 資料檢視 | `inspect_file`, `inspect_episode_id` | plot_data.py 的分析對象 |

## 操作步驟

步驟 1 至 4 在 ROS 終端機執行，步驟 5 至 7 在 conda 終端機執行，
兩者都從 repo 根目錄啟動。

### 1. 建置 ROS package（ROS 終端機）

```bash
colcon build --packages-select speed_control
source install/setup.bash
```

### 2. 啟動 Isaac Sim（ROS 終端機）

開啟 `isaac_motor_usd/` 下的場景（`motor_sim_oneJoint.usd` / `motor_sim_twoJoint.usd` /
`motor_sim_threeJoint.usd`），啟用 ROS 2 Bridge 後按下 Play。

確認 topic 已連通：

```bash
ros2 topic echo /joint_states --once
```

### 3. 校正 effort_to_pos_gain（ROS 終端機，換場景時才需要）

```bash
ros2 run speed_control data_collector     # 選 3
# 或直接執行
ros2 run speed_control gain_calibration
```

以固定力矩脈衝反覆試探，脈衝時間逐輪加長，直到某個關節超過 `calib_limit`。
程式會用最後一次沒撞牆的脈衝反推增益，把印出的數值填回 config 的
`effort_to_pos_gain`。這個值決定訊號產生器預估漂移量的準確度。

### 4. 蒐集資料（ROS 終端機）

```bash
ros2 run speed_control data_collector     # 選 1 收訓練資料，選 2 收測試資料
```

- 選 1：依 `signal_mix` 產生 `total_episodes` 個回合，寫入 `data_file`
- 選 2：以 `test_signal_type` 產生 `test_episodes` 個長回合，寫入 `test_file`

選 1 或 2 之後會再問一次 `effort_to_pos_gain`，直接按 Enter 就用 config 的值：

```
effort_to_pos_gain [0.01]:
```

換場景試增益時不必反覆改 config 存檔。實際採用的值會印在啟動摘要裡，
在第一個命令送出之前。這個值在排程建立時就要定案，所以只能在這裡問，
不能等 node 起來之後再改。

這是唯一需要互動的步驟，因為「要收哪一份資料」與「用哪個增益」都是執行時的
選擇而非參數。兩種模式都會直接覆寫目標檔案，不會產生時間戳檔名。

### 5. 檢查資料品質（conda 終端機）

```bash
python3 plot_data.py
```

輸出超限統計與 `data/<name>_analysis.png`（單回合速度、單回合位置、全場總覽）。

### 6. 訓練（conda 終端機）

```bash
python3 train.py
```

輸出最佳模型到 `models/`、scaler 到 `scalers/`、TensorBoard log 到 `runs/`。
把 config 的 `launch_tensorboard` 設為 `True` 可在訓練開始時自動啟動 TensorBoard。

### 7. 檢視預測結果（conda 終端機）

```bash
python3 visualize_model.py
```

對 `viz_train_episodes` 指定的訓練回合，以及 `test_file` 的指定切片，
做整段開迴路預測（初始狀態為零，只餵命令序列），輸出到 `plots/pred_*.png`。

## 目錄結構

```
motor_control/
├── README.md
├── requirements.txt               訓練端依賴（pip）
├── environment.yml                訓練端依賴（conda）
├── train.py                       訓練驅動程式
├── visualize_model.py             開迴路預測與繪圖
├── plot_data.py                   資料品質檢查
├── isaac_motor_usd/               Isaac Sim 場景
└── src/speed_control/speed_control/
    ├── config.py                  唯一設定來源
    ├── signals.py                 激勵訊號產生與回合排程（不依賴 ROS）
    ├── joint_state.py             /joint_states 解析，兩個 node 共用
    ├── node_runner.py             node 啟動與關閉，兩個 node 共用
    ├── data_collection.py         資料收集 node
    ├── limit_test.py              增益校正 node
    ├── features.py                特徵建構與正規化，訓練與推論共用
    ├── model.py                   NSSM 與 CascadedSystem，唯一定義處
    ├── dataset.py                 回合資料集與分層切分
    └── metrics.py                 R2 與 MSE
```

### 為什麼要拆這麼多檔

- `model.py` 與 `features.py` 被訓練和推論兩邊共用。先前這兩段程式碼各有一份複製，
  改動其中一邊就會讓推論與訓練不一致 —— 特徵那份尤其危險，因為張量維度不變，
  不會拋出任何錯誤，只會讓 R2 莫名下降。
- `signals.py` 不 import ROS，所以可以不開模擬器就產生並檢視波形，方便調參。

## 資料格式

| 欄位 | 說明 |
|---|---|
| `time_actual` | 從節點啟動起算的實際秒數 |
| `time_ideal` | `global_step * dt`，理想時間軸 |
| `episode_id` | 回合編號，從 0 起 |
| `input_u` | 送出的力矩命令 |
| `signal_type` | 該回合的激勵訊號種類，訓練時用於分層切分 |
| `pos_motor`, `vel_motor` | 馬達位置與速度 |
| `pos_joint1`, `vel_joint1` | 第一連桿位置與速度 |
| `pos_joint2`, `vel_joint2` | 第二連桿，僅在場景提供時才有 |

`/joint_states` 走模擬器的時鐘，與命令 timer 並不同步，因此存檔前會把狀態
線性內插到命令的時間軸上，兩者才能共用同一列。

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
馬達模型，兩段實際上是各自獨立學習的。

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

行程保護分三層，由寬到緊：

1. **`planner_safe_limit`（1.2 rad）** —— 訊號產生器的虛擬牆。每段波形產生後會先用
   一個粗略的一階模型預估位置漂移，超過就整體等比例縮小。PRBS 則是直接限制每段
   的持續步數。
2. **`abort_limit`（1.4 rad）** —— 執行期即時保護。收集過程中每步用 `lookahead`
   秒做前瞻預測，一旦當前或預測位置越界，立刻放棄該回合剩餘的激勵訊號，
   切換成 PD 控制把馬達拉回零點。
3. **`hard_limit`（1.57 rad，90 度）** —— 真實機構極限，僅用於事後統計。
   存檔時會報告有多少列、哪些回合曾經超過此值。

## 驗證集切分

`stratified_split` 依 CSV 的 `signal_type` 欄位分層，每種訊號各自以固定間隔抽出
驗證回合，因此無論 `val_ratio` 設成多少，四種訊號在驗證集中都會等比例出現。

若讀到的 CSV 沒有 `signal_type` 欄位（重構前收集的舊資料），程式會印出警告並
依 `signal_mix` 的交錯順序回推種類。該回推只對交錯排程產生的資料有效，
建議重新收集資料以移除這層猜測。

## 注意事項

- **ROS package 名稱為 `speed_control`，但目前做的是力矩控制。** 這是早期速度控制
  版本留下的名稱。改名會影響 `colcon build` 產物名稱與 `ros2 run` 的呼叫方式，
  因此暫時保留。
- **資料收集已可重現。** 訊號產生器會在啟動時以 `config.seed` 設定 NumPy 與
  Python 的隨機種子。
- **訓練預設不保證位元級可重現。** GPU 上 cuDNN 的演算法選擇與浮點累加順序會有
  差異。需要嚴格重現時把 config 的 `deterministic` 設為 `True`。
- **`data/`、`models/`、`runs/`、`plots/`、`scalers/` 都在 .gitignore 內**，
  不會進版控。跨機器搬移時需另外複製，且 `scalers/*.json` 與 `models/*.pth`
  必須成對使用，混用會讓預測完全錯誤。
- **`scipy.interpolate.interp1d` 在新版 SciPy 中標記為 legacy**，目前仍可使用。
  未來若被移除，線性內插可換成 `numpy.interp`，三次樣條可換成
  `scipy.interpolate.CubicSpline`。使用處為 `data_collection.py` 的狀態對齊與
  `signals.py` 的 SMOOTH_NOISE 產生器。
