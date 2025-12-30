import os
import csv
import bisect
import shutil
import numpy as np
import pandas as pd


def generate_vehicle_rsu_data(num_vehicles, num_rsus, vehicle_speed, output_path,
                               area_size=1000, time_frames=800, time_slot=0.5, rsu_range=200):
    # 设置路径
    base_raw_path = os.path.join(output_path, 'raw_data')
    rsu_file = os.path.join(base_raw_path, 'RSU_coordinates.csv')
    vehicle_dir = os.path.join(base_raw_path, f"{num_rsus}r{num_vehicles}v")
    process_dir = os.path.join(output_path, 'process_data', f"{num_rsus}r{num_vehicles}v")

    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(process_dir, exist_ok=True)
    os.makedirs(os.path.dirname(rsu_file), exist_ok=True)

    # 生成 RSU 位置
    rsu_positions_array = np.random.uniform(0, area_size, (num_rsus, 2))
    rsu_data = pd.DataFrame({
        "rsu_id": np.arange(num_rsus),
        "x": rsu_positions_array[:, 0],
        "y": rsu_positions_array[:, 1]
    })
    rsu_data.to_csv(rsu_file, index=False, header=False)

    def in_rsu_range(x, y):
        distances = np.linalg.norm(rsu_positions_array - np.array([x, y]), axis=1)
        return np.any(distances <= rsu_range)

    for vehicle_id in range(num_vehicles):
        x, y = np.random.uniform(0, area_size, 2)
        while not in_rsu_range(x, y):
            x, y = np.random.uniform(0, area_size, 2)

        trajectory = []
        for t in range(time_frames):
            trajectory.append([t * time_slot, x, y])

            new_x = x + np.random.uniform(-vehicle_speed, vehicle_speed)
            new_y = y + np.random.uniform(-vehicle_speed, vehicle_speed)
            if in_rsu_range(new_x, new_y):
                x, y = new_x, new_y

        df = pd.DataFrame(trajectory)
        df.to_csv(f"{vehicle_dir}/{vehicle_id}.csv", index=False, header=False)

    def load_rsu_positions(file_path):
        rsu_pos = {}
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rsu_id, x, y = int(row[0]), float(row[1]), float(row[2])
                rsu_pos[rsu_id] = (x, y)
        return rsu_pos

    def calculate_distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def find_closest_time_frame_index(car_data, v, t):
        times = [frame[0] for frame in car_data[v]]
        idx = bisect.bisect_left(times, t)
        if idx == 0:
            return idx
        elif idx == len(times):
            return idx - 1
        else:
            prev_diff = abs(times[idx - 1] - t)
            curr_diff = abs(times[idx] - t)
            return idx - 1 if prev_diff < curr_diff else idx

    def process_data():
        rsu_positions = load_rsu_positions(rsu_file)
        car_files = [f for f in os.listdir(vehicle_dir) if f.endswith('.csv')]
        rsu_folder = os.path.join(process_dir, 'car_in_rsu')
        os.makedirs(rsu_folder, exist_ok=True)
        shutil.copy(rsu_file, rsu_folder)

        car_data = {}
        max_time = 0
        for car_file in car_files:
            car_id = os.path.splitext(car_file)[0]
            car_data[car_id] = []
            with open(os.path.join(vehicle_dir, car_file), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    t, x, y = float(row[0]), float(row[1]), float(row[2])
                    car_data[car_id].append((t, x, y))
                    if t > max_time:
                        max_time = t

        total_steps = int(max_time // time_slot) + 1
        rsu_writers = {}
        for rsu_id in rsu_positions:
            f = open(os.path.join(rsu_folder, f"{rsu_id}.csv"), 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(['时间帧', '范围内车辆数', '车辆ID按距离排序'])
            rsu_writers[rsu_id] = f

        for step in range(total_steps):
            t = step * time_slot
            vehicles_in_rsu = {r: [] for r in rsu_positions}
            for car_id, records in car_data.items():
                for record in records:
                    if record[0] == t:
                        car_x, car_y = record[1], record[2]
                        for rsu_id, (rsu_x, rsu_y) in rsu_positions.items():
                            if calculate_distance(car_x, car_y, rsu_x, rsu_y) <= rsu_range:
                                vehicles_in_rsu[rsu_id].append(car_id)
                        break

            for rsu_id, cars in vehicles_in_rsu.items():
                writer = csv.writer(rsu_writers[rsu_id])
                sorted_ids = sorted(cars, key=lambda v: calculate_distance(
                    car_data[v][find_closest_time_frame_index(car_data, v, t)][1],
                    car_data[v][find_closest_time_frame_index(car_data, v, t)][2],
                    rsu_positions[rsu_id][0],
                    rsu_positions[rsu_id][1],
                ))
                writer.writerow([t, len(sorted_ids)] + sorted_ids)

        for f in rsu_writers.values():
            f.close()
        return car_data, rsu_positions

    def write_car_files(car_data, rsu_positions):
        for car_id, records in car_data.items():
            car_file = os.path.join(process_dir, f"{car_id}.csv")
            with open(car_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['时间帧', '车辆x坐标', '车辆y坐标', '车辆能通信的RSU数量', '车辆在RSU范围内的RSU编号依次排序'])
                for record in records:
                    t, x, y = record
                    rsus = [(rid, calculate_distance(x, y, *pos)) for rid, pos in rsu_positions.items()
                            if calculate_distance(x, y, *pos) <= rsu_range]
                    sorted_ids = [rid for rid, _ in sorted(rsus, key=lambda x: x[1])]
                    writer.writerow([t, x, y, len(sorted_ids)] + sorted_ids)

    print(f"开始生成车辆与RSU数据：{num_rsus} RSUs，{num_vehicles} 车辆，输出目录：{output_path}")
    car_data, rsu_positions = process_data()
    write_car_files(car_data, rsu_positions)
