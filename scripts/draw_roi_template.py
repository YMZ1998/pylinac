import matplotlib.pyplot as plt
import numpy as np

roi_dist_mm = 58.7
roi_radius_mm = 5
rotation_offset = 0

roi_settings = {
    "Air": {"angle": 90, "hu": -1000.0},
    "PMP": {"angle": 60, "hu": -196.0},
    "LDPE": {"angle": 0, "hu": -104.0},
    "Poly": {"angle": -60, "hu": -47.0},
    "Acrylic": {"angle": -120, "hu": 115.0},
    "Delrin": {"angle": -180, "hu": 365.0},
    "Teflon": {"angle": 120, "hu": 1000.0},
    "Air2": {"angle": -90, "hu": -1000.0},
}


def draw_ctp404_template():
    phantom_radius = 100
    size = 256
    img = np.zeros((size, size), dtype=np.float32)

    cx, cy = size // 2, size // 2
    mm_per_pixel = phantom_radius / (size / 2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # =========================
    # 1. 生成 HU phantom
    # =========================
    yy, xx = np.mgrid[0:size, 0:size]
    xx = (xx - cx) * mm_per_pixel
    yy = (yy - cy) * mm_per_pixel

    # 外圆 phantom mask
    phantom_mask = xx ** 2 + yy ** 2 <= phantom_radius ** 2
    img[phantom_mask] = 0

    # ROI 填 HU
    for name, cfg in roi_settings.items():
        angle = np.deg2rad(-cfg["angle"] + rotation_offset)

        x0 = roi_dist_mm * np.cos(angle)
        y0 = roi_dist_mm * np.sin(angle)

        mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= roi_radius_mm ** 2
        img[mask] = cfg["hu"]

    ax[0].imshow(img, cmap='gray', origin='lower')
    ax[0].set_title("Simulated HU Phantom")

    # =========================
    # 2. 画模板（你原来的图）
    # =========================
    circle = plt.Circle((0, 0), phantom_radius, fill=False, linewidth=2)
    ax[1].add_patch(circle)

    for name, cfg in roi_settings.items():
        angle = np.deg2rad(-cfg["angle"] + rotation_offset)

        x = roi_dist_mm * np.cos(angle)
        y = roi_dist_mm * np.sin(angle)

        roi_circle = plt.Circle((x, y), roi_radius_mm, fill=False)
        ax[1].add_patch(roi_circle)

        ax[1].text(x, y, name, fontsize=9)

    ax[1].plot(0, 0, 'ro')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(-120, 120)
    ax[1].set_ylim(-120, 120)
    ax[1].set_title("ROI Template")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_ctp404_template()
