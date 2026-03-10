import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from pylinac import CatPhan600
from scripts.nii_dcm import get_image_basename, nii_to_dicom_series


def polyfit_through_first_point(x, y, degree):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = x[0]
    y0 = y[0]

    # 构造设计矩阵
    X = np.vstack([(x - x0) ** i for i in range(1, degree + 1)]).T

    # 右侧
    Y = y - y0

    # 最小二乘
    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]

    # 构造完整多项式
    poly = np.poly1d([0])
    for i, c in enumerate(coeffs, start=1):
        poly += np.poly1d([c]) * np.poly1d([1, -x0]) ** i

    poly += y0

    return poly


def fit_hu_curve(dicom_dir, degree=2):
    print("Analyzing CatPhan HU inserts...")

    cbct = CatPhan600(dicom_dir)
    cbct.analyze()

    results, rois_hu, slice_num = cbct.get_hu()
    print(results)
    # cbct.plot_analyzed_subimage("hu")

    hu_module = cbct.ctp404

    measured_hu = hu_module.roi_vals_as_dict
    # expected_hu = hu_module.get_expected_hu_values
    # expected_hu = {'Air': -1000.0, 'PMP': -200.0, 'LDPE': -100.0, 'Poly': -35.0, 'Acrylic': 120.0, 'Delrin': 340.0,
    #            'Teflon': 990.0}
    # expected_hu = {'Air': -1000.0, 'LDPE': -100.0, 'Delrin': 340.0, 'Teflon': 990.0}
    expected_hu = {'Air': -1000.0, 'LDPE': -104.0, 'Delrin': 365.0, 'Teflon': 990.0}
    print("Measured HU: ", measured_hu)
    print("Expected HU: ", expected_hu)

    measured = []
    expected = []

    print("\nHU measurement:")

    for name in expected_hu:
        m = measured_hu[name]
        e = expected_hu[name]

        measured.append(m)
        expected.append(e)

        print(f"{name:10s} measured={m:8.2f} expected={e:8.2f} error={m - e:8.2f}")

    measured = np.array(measured)
    expected = np.array(expected)

    # 多项式拟合
    # coeffs = np.polyfit(measured, expected, degree)
    # poly = np.poly1d(coeffs)

    poly = polyfit_through_first_point(measured, expected, degree)
    print("first point check:", poly(measured[0]), expected[0])

    print("\nHU calibration polynomial:")
    print(poly)

    # 绘制曲线
    x = np.linspace(min(measured) - 200, max(measured) + 200, 500)
    y = poly(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(measured, expected, label="CatPhan inserts")
    plt.plot(x, y, label=f"{degree} order fit")

    plt.xlabel("Measured HU")
    plt.ylabel("True HU")
    plt.title("HU Calibration Curve")
    plt.grid(True)
    plt.legend()

    # plt.show()

    return poly


def correct_cbct_volume(dicom_dir, output_dir, poly):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(dicom_dir))

    print("\nApplying HU correction...")

    for f in files:

        if not f.endswith(".dcm"):
            continue

        path = os.path.join(dicom_dir, f)

        ds = pydicom.dcmread(path)

        pixel = ds.pixel_array.astype(np.float32)

        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)

        # 转 HU
        hu = pixel * slope + intercept

        # 应用校正
        hu_corrected = poly(hu)
        hu_corrected = np.rint(hu_corrected)

        # 转回 pixel
        pixel_corrected = (hu_corrected - intercept) / slope

        pixel_corrected = np.clip(pixel_corrected, -32768, 32767)

        pixel_corrected = pixel_corrected.astype(ds.pixel_array.dtype)

        ds.PixelData = pixel_corrected.tobytes()

        out_path = os.path.join(output_dir, f)

        ds.save_as(out_path)

    print("\nCorrected DICOM saved to:")
    print(output_dir)


def correct_mhd_volume(mhd_path, output_path, poly):
    print("Reading MHD volume...")

    img = sitk.ReadImage(mhd_path)

    volume = sitk.GetArrayFromImage(img).astype(np.int16)

    hu = volume

    print("Applying HU correction...")

    hu_corrected = poly(hu)
    hu_corrected = np.rint(hu_corrected)
    print("hu_corrected: ", np.min(hu_corrected))

    corrected_img = sitk.GetImageFromArray(hu_corrected)
    corrected_img.CopyInformation(img)
    corrected_img = sitk.Cast(corrected_img, img.GetPixelIDValue())

    sitk.WriteImage(corrected_img, output_path)

    print("Corrected MHD saved to:")
    print(output_path)


def main():
    nii_path = r"E:\cbct\A_output.mhd"

    dicom_dir = os.path.join(os.path.dirname(nii_path), "temp", get_image_basename(nii_path))
    nii_to_dicom_series(nii_path, dicom_dir, use_random_id=True)

    output_dir = r"E:\cbct\A_output_HU_corrected"
    nii_output_path = r"E:\cbct\A_output_HU_corrected.mhd"

    # Step1 HU拟合
    poly = fit_hu_curve(dicom_dir, degree=2)

    # Step2 校正CBCT体数据
    correct_cbct_volume(dicom_dir, output_dir, poly)
    correct_mhd_volume(nii_path, nii_output_path, poly)

    poly2 = fit_hu_curve(output_dir, degree=2)


if __name__ == "__main__":
    main()
