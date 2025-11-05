import pandas as pd

data = pd.read_csv("detections.csv")

# Only keep people in exhibit zones
data = data[data["zone"] > 0]

# Group by zone
zone_stats = {}

for zone_id, zone_data in data.groupby("zone"):
    zone_data = zone_data.sort_values("frame")
    
    # Each entry of same 'person' would normally be tracked by ID,
    # but since we don't have tracking, we’ll measure total presence over time
    total_time = zone_data["time_s"].max() - zone_data["time_s"].min()
    zone_stats[zone_id] = total_time

print("\n📊 Zone durations (approximate total presence):")
for zone_id, duration in zone_stats.items():
    print(f"Zone {zone_id}: {duration:.2f} seconds of total human presence")

# Optional: visualize per-frame occupancy
import matplotlib.pyplot as plt

zone_counts = data.groupby(["time_s", "zone"]).size().unstack(fill_value=0)
zone_counts.plot()
plt.title("People detected per zone over time")
plt.xlabel("Time (s)")
plt.ylabel("Count")
plt.show()
