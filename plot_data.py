import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# ==========================================
# 🎯 繪圖分析設定區 (請在這裡修改參數)
# ==========================================
DATA_DIR = "data" 
HARD_LIMIT = 1.57

# 1. 指定要分析的檔案 (留空 "" 或檔案不存在時，將自動抓取最新的 .csv)
SPECIFIC_FILE = "train.csv"

# 2. 指定想要放大檢視的特定回合 ID (預設為 0)
PLOT_EPISODE_ID = 0  

# 雖然不分階段過濾，但畫圖時標示出 Reset 起點還是很有幫助的
RUN_STEPS = 200    
# ==========================================


def get_latest_csv(directory):
    list_of_files = glob.glob(os.path.join(directory, "*.csv"))
    if not list_of_files: return None
    return max(list_of_files, key=os.path.getctime)

def analyze_data(df):
    # 1. 找出所有超過極限的資料點
    exceed_mask = df['pos_joint'].abs() > HARD_LIMIT
    
    # 2. 抓出這些資料點對應的回合 ID
    failed_episodes = df[exceed_mask]['episode_id'].unique().astype(int)
    total_episodes = df['episode_id'].nunique()
    
    print(f"\n" + "="*45)
    print(f"📊 數據極限分析報告")
    print(f"---------------------------------------------")
    print(f"總回合數: {total_episodes}")
    print(f"❌ 發生超限的回合數: {len(failed_episodes)}")
    
    # 3. 印出超限的回合 ID
    if len(failed_episodes) > 0:
        print(f"⚠️ 超限的回合 ID: {failed_episodes.tolist()}")
    else:
        print(f"✨ 所有數據均符合安全規範，完全沒有超限！")
    print("="*45 + "\n")

    return failed_episodes

def get_target_csv(directory, target_filename):
    """根據參數決定要使用的 CSV 檔案路徑，具備自動防呆與 Fallback 機制"""
    if target_filename:
        # 防呆：如果忘記打副檔名，自動補上
        if not target_filename.endswith('.csv'):
            target_filename += '.csv'
            
        specific_path = os.path.join(directory, target_filename)
        if os.path.exists(specific_path):
            print(f"\n✅ 找到指定檔案: {specific_path}")
            return specific_path
        else:
            print(f"\n⚠️ 找不到指定檔案 '{specific_path}'，將自動切換為最新檔案...")
    
    # Fallback: 使用最新檔案 (當 SPECIFIC_FILE 為空，或找不到檔案時)
    latest_file = get_latest_csv(directory)
    if latest_file:
        print(f"\n✅ 使用最新檔案: {latest_file}")
    else:
        print(f"\n❌ 錯誤：在 '{directory}' 目錄下找不到任何 CSV 檔案！")
        
    return latest_file

def plot_motor_data():
    csv_path = get_target_csv(DATA_DIR, SPECIFIC_FILE)
    if not csv_path: 
        return
    
    df = pd.read_csv(csv_path)
    failed_eps = analyze_data(df)

    time_col = 'time_actual'
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
    
    # --- 子圖 1 & 2 (細節) ---
    # 檢查使用者指定的 ID 是否存在於資料中
    target_ep = PLOT_EPISODE_ID
    if target_ep not in df['episode_id'].values:
        print(f"⚠️ 找不到回合 ID {target_ep}，自動改為繪製第一回合 (ID: {int(df['episode_id'].iloc[0])})")
        target_ep = df['episode_id'].iloc[0]
        
    subset = df[df['episode_id'] == target_ep]
    
    ax1.plot(subset[time_col], subset['input_u'], 'b--', alpha=0.6, label='Target (u)')
    ax1.plot(subset[time_col], subset['vel_joint'], 'g-', label='Joint Vel')
    
    # 畫出 Reset 開始的輔助線 (防呆：確保資料夠長才畫)
    reset_idx = min(RUN_STEPS - 1, len(subset) - 1)
    if reset_idx > 0:
        ax1.axvline(x=subset[time_col].iloc[reset_idx], color='orange', linestyle=':', label='Reset Start')
        
    ax1.set_title(f"Velocity Detail (Episode {int(target_ep)})")
    ax1.legend()

    ax2.plot(subset[time_col], subset['pos_joint'], 'g-')
    ax2.axhline(y=HARD_LIMIT, color='r', linestyle='-.')
    ax2.axhline(y=-HARD_LIMIT, color='r', linestyle='-.')
    ax2.set_title(f"Position Detail (Episode {int(target_ep)})")

    # --- 子圖 3: 全局視圖與分類標註 ---
    ax3.plot(df[time_col], df['pos_joint'], color='green', alpha=0.5)
    ax3.axhline(y=HARD_LIMIT, color='red', linestyle='-.', alpha=0.8)
    ax3.axhline(y=-HARD_LIMIT, color='red', linestyle='-.', alpha=0.8)
    
    first_fail_labeled = False
    for ep_id, group in df.groupby('episode_id'):
        if ep_id in failed_eps:
            t_start, t_end = group[time_col].min(), group[time_col].max()
            # 只在 legend 顯示一次標籤，避免重複
            label = 'Exceed Limit' if not first_fail_labeled else ""
            ax3.axvspan(t_start, t_end, color='red', alpha=0.3, label=label)
            first_fail_labeled = True
            
    ax3.set_title("Full Overview (Red highlights indicate limit exceedance)")
    ax3.set_xlabel("Time [sec]")
    if first_fail_labeled:
        ax3.legend(loc='upper right')
    
    plt.tight_layout()
    output_png = os.path.join(DATA_DIR, os.path.basename(csv_path).replace('.csv', '_analysis.png'))
    plt.savefig(output_png)
    plt.close()
    
    print(f"✅ 分析圖表已儲存至: {output_png}")

if __name__ == "__main__":
    plot_motor_data()