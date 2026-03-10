import os

from pylinac import CatPhan600
from scripts.nii_dcm import nii_to_dicom_series, get_image_basename

if __name__ == "__main__":
    nii_path = r"D:\Data\cbct\A_output.mhd"
    out_dir = os.path.join(os.path.dirname(nii_path), "temp", get_image_basename(nii_path))
    nii_to_dicom_series(nii_path, out_dir, use_random_id=True)
    cbct = CatPhan600(out_dir)
    cbct.analyze()
    # print(cbct.results())
    results, measured_hu, slice_num = cbct.get_hu()
    print(results)
    cbct.plot_analyzed_subimage("hu")

    # expected_hu = {'Air': -1000.0, 'PMP': -200.0, 'LDPE': -100.0, 'Poly': -35.0, 'Acrylic': 120.0, 'Delrin': 340.0,
    #            'Teflon': 990.0}
    expected_hu = {'Air': -1000.0, 'LDPE': -104.0, 'Delrin': 365.0, 'Teflon': 990.0}

    print("\nHU measurement:")

    for name in expected_hu:
        m = measured_hu[name]
        e = expected_hu[name]

        print(f"{name:10s} measured={m:8.2f} expected={e:8.2f} error={m - e:8.2f}")
