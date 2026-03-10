import argparse
import json
import os

import SimpleITK as sitk
import numpy as np
import pydicom

from pylinac import CatPhan600
from scripts.nii_dcm import get_image_basename, nii_to_dicom_series


# ---------- 多项式拟合经过第一个点 ----------
def polyfit_through_first_point(x, y, degree):
    x, y = np.asarray(x), np.asarray(y)
    x0, y0 = x[0], y[0]
    X = np.vstack([(x - x0) ** i for i in range(1, degree + 1)]).T
    Y = y - y0
    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
    poly = np.poly1d([0])
    for i, c in enumerate(coeffs, start=1):
        poly += np.poly1d([c]) * np.poly1d([1, -x0]) ** i
    poly += y0
    return poly, coeffs, x0, y0


# ---------- HU 拟合 ----------
def fit_hu_curve(dicom_dir, degree=2, json_path=None):
    print("Analyzing CatPhan HU inserts...")
    cbct = CatPhan600(dicom_dir)
    cbct.analyze()
    results, rois_hu, slice_num = cbct.get_hu()
    print(results)
    print(rois_hu)
    print(slice_num)

    hu_module = cbct.ctp404

    measured_hu = hu_module.roi_vals_as_dict

    # 读取 JSON 的标准 HU 或保存默认
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            expected_hu = json.load(f)["expected_hu"]
        print(f"Loaded expected HU from {json_path}")
    else:
        expected_hu = {'Air': -1000.0, 'LDPE': -104.0, 'Delrin': 365.0, 'Teflon': 990.0}
        print("Using default expected HU")
    print("Expected HU: ", expected_hu)
    print("Measured HU: ", measured_hu)
    measured = [measured_hu[name] for name in expected_hu]
    expected = [expected_hu[name] for name in expected_hu]

    poly, coeffs, x0, y0 = polyfit_through_first_point(measured, expected, degree)
    print("First point check:", poly(measured[0]), expected[0])
    print("HU calibration polynomial: ", poly)

    c1 = coeffs[0]
    c2 = coeffs[1]

    a2 = c2
    a1 = c1 - 2 * c2 * x0
    a0 = y0 - c1 * x0 + c2 * x0 * x0

    print("a2 =", a2)
    print("a1 =", a1)
    print("a0 =", a0)

    # 保存多项式系数到 JSON
    if json_path:
        hu_info = {
            "expected_hu": expected_hu,
            "poly_degree": degree,
            "a2": float(a2),
            "a1": float(a1),
            "a0": float(a0)
        }
        with open(json_path, "w") as f:
            json.dump(hu_info, f, indent=2)
        print(f"Saved HU polynomial coefficients to {json_path}")

    return poly


# ---------- DICOM 校正 ----------
def correct_cbct_volume(dicom_dir, poly):
    print("\nApplying HU correction to DICOMs...")
    files = sorted(os.listdir(dicom_dir))
    for f in files:
        if not f.endswith(".dcm"):
            continue
        path = os.path.join(dicom_dir, f)
        ds = pydicom.dcmread(path)
        pixel = ds.pixel_array.astype(np.float32)
        slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        hu = pixel * slope + intercept
        hu_corrected = np.rint(poly(hu))
        pixel_corrected = (hu_corrected - intercept) / slope
        pixel_corrected = np.clip(pixel_corrected, -32768, 32767)
        pixel_corrected = pixel_corrected.astype(ds.pixel_array.dtype)
        ds.PixelData = pixel_corrected.tobytes()
        ds.save_as(path)  # 直接覆盖


# ---------- MHD 校正 ----------
def correct_mhd_volume(mhd_path, poly):
    print(f"\nReading MHD volume: {mhd_path}")
    img = sitk.ReadImage(mhd_path)
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    hu_corrected = np.rint(poly(hu))
    corrected_img = sitk.GetImageFromArray(hu_corrected.astype(np.int16))
    corrected_img.CopyInformation(img)
    correct_mhd_path=mhd_path.replace(".mhd", "_HU_corrected.mhd")
    sitk.WriteImage(corrected_img, correct_mhd_path)
    print(f"Corrected MHD saved: {correct_mhd_path}")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="CBCT HU Calibration Tool (single MHD & JSON)")
    parser.add_argument("--mhd_path", type=str, default=r"E:\cbct\A_output.mhd", help="Input CBCT .mhd file")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for HU calibration")
    parser.add_argument("--json", type=str, default=r"E:\cbct\hu_calibration.json",
                        help="Path to JSON for expected HU & coefficients")
    args = parser.parse_args()

    dicom_dir = os.path.join(os.path.dirname(args.mhd_path), "temp", get_image_basename(args.mhd_path))
    nii_to_dicom_series(args.mhd_path, dicom_dir, use_random_id=True)

    poly = fit_hu_curve(dicom_dir, degree=args.degree, json_path=args.json)

    # correct_cbct_volume(dicom_dir, poly)
    # correct_mhd_volume(args.mhd_path, poly)


if __name__ == "__main__":
    main()
