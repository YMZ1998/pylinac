import argparse
import datetime
import json
import os
import sys
import time

import SimpleITK as sitk
import numpy as np
import pydicom

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pylinac import CatPhan600
from scripts.nii_dcm import get_image_basename, nii_to_dicom_series


DEFAULT_EXPECTED_HU = {
    "Air": -1000.0,
    "PMP": -196.0,
    "LDPE": -104.0,
    "Poly": -47.0,
    "Acrylic": 115.0,
    "Delrin": 365.0,
    "Teflon": 1000.0,
}


def image_input_mtime(image_path):
    mtimes = [os.path.getmtime(image_path)]
    if image_path.lower().endswith(".mhd"):
        with open(image_path, "r") as f:
            for line in f:
                if line.strip().startswith("ElementDataFile"):
                    data_file = line.split("=", 1)[1].strip()
                    if data_file and data_file.upper() not in {"LOCAL", "LIST"}:
                        data_path = os.path.join(os.path.dirname(image_path), data_file)
                        if os.path.exists(data_path):
                            mtimes.append(os.path.getmtime(data_path))
                    break
    return max(mtimes)


def dicom_series_is_current(image_path, dicom_dir):
    if not os.path.isdir(dicom_dir):
        return False
    dcm_files = [
        entry
        for entry in os.scandir(dicom_dir)
        if entry.is_file() and entry.name.lower().endswith(".dcm")
    ]
    if not dcm_files:
        return False
    return min(entry.stat().st_mtime for entry in dcm_files) >= image_input_mtime(
        image_path
    )


def prepare_dicom_series(image_path, dicom_dir, force=False):
    if not force and dicom_series_is_current(image_path, dicom_dir):
        print(f"Reusing existing DICOM series: {dicom_dir}")
        return
    nii_to_dicom_series(image_path, dicom_dir, use_random_id=True)


def apply_poly(poly, values):
    coeffs = np.asarray(poly.coeffs, dtype=np.float32)
    result = np.empty_like(values, dtype=np.float32)
    result.fill(coeffs[0])
    for coeff in coeffs[1:]:
        result *= values
        result += coeff
    return result


def apply_piecewise_linear(measured_points, expected_points, values):
    measured_points = np.asarray(measured_points, dtype=np.float32)
    expected_points = np.asarray(expected_points, dtype=np.float32)
    order = np.argsort(measured_points)
    measured_points = measured_points[order]
    expected_points = expected_points[order]

    result = np.interp(values, measured_points, expected_points).astype(np.float32)
    left = values < measured_points[0]
    right = values > measured_points[-1]

    if np.any(left):
        slope = (expected_points[1] - expected_points[0]) / (
            measured_points[1] - measured_points[0]
        )
        result[left] = expected_points[0] + (values[left] - measured_points[0]) * slope
    if np.any(right):
        slope = (expected_points[-1] - expected_points[-2]) / (
            measured_points[-1] - measured_points[-2]
        )
        result[right] = expected_points[-1] + (
            values[right] - measured_points[-1]
        ) * slope
    return result


def apply_calibration(calibration, values):
    if calibration["method"] == "piecewise":
        return apply_piecewise_linear(
            calibration["measured"], calibration["expected"], values
        )
    return apply_poly(calibration["poly"], values)


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
def build_calibration(measured, expected, method="piecewise", degree=2):
    measured = np.asarray(measured, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if len(measured) < 2:
        raise ValueError("At least 2 HU calibration points are required")

    if method == "piecewise":
        fitted = apply_piecewise_linear(measured, expected, measured)
        calibration = {
            "method": method,
            "measured": measured,
            "expected": expected,
            "poly": None,
            "coefficients": [],
        }
    elif method == "poly":
        fit_degree = min(degree, len(measured) - 1)
        poly = np.poly1d(np.polyfit(measured, expected, fit_degree))
        fitted = poly(measured)
        calibration = {
            "method": method,
            "measured": measured,
            "expected": expected,
            "poly": poly,
            "coefficients": [float(c) for c in poly.coeffs],
        }
    elif method == "poly_anchor":
        fit_degree = min(degree, len(measured) - 1)
        poly, _, _, _ = polyfit_through_first_point(measured, expected, fit_degree)
        fitted = poly(measured)
        calibration = {
            "method": method,
            "measured": measured,
            "expected": expected,
            "poly": poly,
            "coefficients": [float(c) for c in poly.coeffs],
        }
    else:
        raise ValueError(f"Unknown fit method: {method}")

    errors = fitted - expected
    calibration["errors"] = errors
    calibration["max_abs_error"] = float(np.max(np.abs(errors)))
    calibration["mean_abs_error"] = float(np.mean(np.abs(errors)))
    return calibration


def fit_hu_curve(
    dicom_dir,
    degree=2,
    json_path=None,
    angle_offset_deg=0,
    fit_method="piecewise",
):
    print("Analyzing CatPhan HU inserts...")
    cbct = CatPhan600(dicom_dir, angle_offset_deg=angle_offset_deg)
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
        expected_hu = DEFAULT_EXPECTED_HU
        print("Using default expected HU")
    print("Expected HU: ", expected_hu)
    print("Measured HU: ", measured_hu)
    roi_names = [name for name in expected_hu if name in measured_hu]
    measured = [measured_hu[name] for name in roi_names]
    expected = [expected_hu[name] for name in roi_names]

    calibration = build_calibration(
        measured, expected, method=fit_method, degree=degree
    )
    print(f"HU calibration method: {fit_method}")
    if calibration["poly"] is not None:
        print("HU calibration polynomial: ", calibration["poly"])
    print("Calibration residuals:")
    for name, measured_value, expected_value, error in zip(
        roi_names, measured, expected, calibration["errors"]
    ):
        print(
            f"{name:10s} measured={measured_value:8.2f} "
            f"expected={expected_value:8.2f} residual={error:8.2f}"
        )
    print("Max abs residual:", calibration["max_abs_error"])
    print("Mean abs residual:", calibration["mean_abs_error"])

    # 保存多项式系数到 JSON
    if json_path:
        hu_info = {
            "Catphan": 600,
            "expected_hu": expected_hu,
            "measured_hu": {name: float(measured_hu[name]) for name in roi_names},
            "fit_method": fit_method,
            "poly_degree": degree,
            "coefficients": calibration["coefficients"],
            "points": [
                {
                    "name": name,
                    "measured": float(measured_value),
                    "expected": float(expected_value),
                    "residual": float(error),
                }
                for name, measured_value, expected_value, error in zip(
                    roi_names, measured, expected, calibration["errors"]
                )
            ],
            "max_abs_error": calibration["max_abs_error"],
            "mean_abs_error": calibration["mean_abs_error"],
        }
        if len(calibration["coefficients"]) == 3:
            hu_info["a2"] = calibration["coefficients"][0]
            hu_info["a1"] = calibration["coefficients"][1]
            hu_info["a0"] = calibration["coefficients"][2]
        with open(json_path, "w") as f:
            json.dump(hu_info, f, indent=2)
        print(f"Saved HU polynomial coefficients to {json_path}")

    return calibration


# ---------- DICOM 校正 ----------
def correct_cbct_volume(dicom_dir, calibration):
    print("\nApplying HU correction to DICOMs...")
    files = sorted(f for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm"))
    for f in files:
        path = os.path.join(dicom_dir, f)
        ds = pydicom.dcmread(path)
        pixel = ds.pixel_array
        original_dtype = pixel.dtype
        hu = pixel.astype(np.float32, copy=False)
        slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        if slope != 1 or intercept != 0:
            hu = hu * slope + intercept
        pixel_corrected = apply_calibration(calibration, hu)
        np.rint(pixel_corrected, out=pixel_corrected)
        if slope != 1 or intercept != 0:
            pixel_corrected -= intercept
            pixel_corrected /= slope
        np.clip(pixel_corrected, -32768, 32767, out=pixel_corrected)
        pixel_corrected = pixel_corrected.astype(original_dtype)
        ds.PixelData = pixel_corrected.tobytes()
        ds.save_as(path)  # 直接覆盖


# ---------- MHD 校正 ----------
def correct_mhd_volume(mhd_path, calibration):
    print(f"\nReading MHD volume: {mhd_path}")
    img = sitk.ReadImage(mhd_path)
    hu = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
    hu_corrected = apply_calibration(calibration, hu)
    np.rint(hu_corrected, out=hu_corrected)
    corrected_img = sitk.GetImageFromArray(hu_corrected.astype(np.int16))
    corrected_img.CopyInformation(img)
    correct_mhd_path = mhd_path.replace(".mhd", "_HU_corrected.mhd")
    sitk.WriteImage(corrected_img, correct_mhd_path)
    print(f"Corrected MHD saved: {correct_mhd_path}")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="CBCT HU Calibration Tool (single MHD & JSON)")
    parser.add_argument("--mhd_path", type=str, default=r"E:\cbct\A_output.mhd", help="Input CBCT .mhd file")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for HU calibration")
    parser.add_argument("--json_path", type=str, default=r"E:\cbct\hu_calibration.json",
                        help="Path to JSON for expected HU & coefficients")
    parser.add_argument("--angle_offset_deg", type=float, default=0,
                        help="Angle offset in degrees for CatPhan600 CTP404 HU ROIs")
    parser.add_argument("--fit_method", choices=("piecewise", "poly", "poly_anchor"),
                        default="piecewise",
                        help="HU fit method. piecewise passes through all calibration points.")
    parser.add_argument("--force_dicom_conversion", action="store_true",
                        help="Always regenerate temporary DICOM files instead of reusing current files")
    args = parser.parse_args()

    dicom_dir = os.path.join(os.path.dirname(args.mhd_path), "temp", get_image_basename(args.mhd_path))
    prepare_dicom_series(
        args.mhd_path,
        dicom_dir,
        force=args.force_dicom_conversion,
    )

    calibration = fit_hu_curve(
        dicom_dir,
        degree=args.degree,
        json_path=args.json_path,
        angle_offset_deg=args.angle_offset_deg,
        fit_method=args.fit_method,
    )

    # correct_cbct_volume(dicom_dir, calibration)
    # correct_mhd_volume(args.mhd_path, calibration)


if __name__ == "__main__":
    start = time.time()
    main()
    print("\nTotal Time:", str(datetime.timedelta(seconds=int(time.time() - start))))
