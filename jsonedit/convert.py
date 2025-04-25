import json

# Konversi dari meter per detik (m/s) ke kilometer per jam (km/h)
MPS_TO_KPH = 3.6

# Membaca dataset dari file teks
def read_speeds(file_path):
    """Membaca nilai kecepatan dari file teks dan mengembalikannya sebagai daftar float."""
    with open(file_path, "r") as file:
        speeds = [float(line.strip()) for line in file.readlines()]
    return speeds

# Mengonversi kecepatan dan memformat ke dalam struktur JSON
def convert_to_json(speeds, video_name):
    """Mengonversi nilai kecepatan ke format JSON dengan nama file dan nilai kecepatan bulat."""
    data = {}
    for i, speed in enumerate(speeds, start=1):
        frame_name = f"{video_name}_frame_{i:05d}.jpg"
        data[frame_name] = {"speed": round(speed * MPS_TO_KPH)}  # Membulatkan ke angka terdekat
    return data

# Menyimpan data JSON ke file
def save_json(data, output_path):
    """Menyimpan data JSON ke file yang ditentukan."""
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    input_txt = "train.txt"  # Path ke file teks input berisi nilai kecepatan
    output_json = "data_filtered.json"  # Path untuk menyimpan file JSON output
    video_name = "00_0000_comma"  # Prefiks untuk nama frame video

    speeds = read_speeds(input_txt)  # Membaca nilai kecepatan dari file teks
    json_data = convert_to_json(speeds, video_name)  # Mengonversi ke JSON
    save_json(json_data, output_json)  # Menyimpan data JSON ke file

    print(f"JSON data berhasil disimpan ke {output_json}")
