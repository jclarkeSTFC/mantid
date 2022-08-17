# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from isis_powder.routines.param_map_entry import ParamMapEntry
from isis_powder.routines.common import PARAM_MAPPING


# Maps friendly user name (ext_name) -> script name (int_name)
attr_mapping = [
    ParamMapEntry(ext_name="attenuation_files", int_name="attenuation_files"),
    ParamMapEntry(ext_name="attenuation_file", int_name="attenuation_file"),
    ParamMapEntry(ext_name="config_file", int_name="config_file_name"),
    ParamMapEntry(ext_name="calibration_mapping_file", int_name="cal_mapping_path"),
    ParamMapEntry(ext_name="calibration_directory", int_name="calibration_dir"),
    ParamMapEntry(ext_name="create_cal_rebin_1_params", int_name="cal_rebin_1"),
    ParamMapEntry(ext_name="create_cal_rebin_2_params", int_name="cal_rebin_2"),
    ParamMapEntry(ext_name="cross_corr_reference_spectra", int_name="reference_spectra"),
    ParamMapEntry(ext_name="cross_corr_ws_index_max", int_name="cross_corr_ws_max"),
    ParamMapEntry(ext_name="cross_corr_ws_index_min", int_name="cross_corr_ws_min"),
    ParamMapEntry(ext_name="cross_corr_x_min", int_name="cross_corr_x_min"),
    ParamMapEntry(ext_name="cross_corr_x_max", int_name="cross_corr_x_max"),
    ParamMapEntry(ext_name="do_absorb_corrections", int_name="absorb_corrections"),
    ParamMapEntry(ext_name="file_ext", int_name="file_extension", optional=True),
    ParamMapEntry(ext_name="focused_bin_widths", int_name="focused_bin_widths"),
    ParamMapEntry(ext_name="custom_focused_bin_widths", int_name="custom_focused_bin_widths"),
    ParamMapEntry(ext_name="focused_cropping_values", int_name="tof_cropping_values"),
    ParamMapEntry(ext_name="custom_focused_cropping_values", int_name="custom_tof_cropping_values"),
    ParamMapEntry(ext_name="generate_absorb_corrections", int_name="gen_absorb"),
    ParamMapEntry(ext_name="get_det_offsets_d_ref", int_name="d_reference"),
    ParamMapEntry(ext_name="get_det_offsets_step", int_name="get_det_offsets_step"),
    ParamMapEntry(ext_name="get_det_offsets_x_min", int_name="get_det_offsets_x_min"),
    ParamMapEntry(ext_name="get_det_offsets_x_max", int_name="get_det_offsets_x_max"),
    ParamMapEntry(ext_name="merge_drange", int_name="merge_drange"),
    ParamMapEntry(ext_name="monitor_lambda_crop_range", int_name="monitor_lambda"),
    ParamMapEntry(ext_name="monitor_integration_range", int_name="monitor_integration_range"),
    ParamMapEntry(ext_name="monitor_mask_regions", int_name="monitor_mask_regions"),
    ParamMapEntry(ext_name="monitor_spectrum_number", int_name="monitor_spec_no"),
    ParamMapEntry(ext_name="monitor_spline_coefficient", int_name="monitor_spline"),
    ParamMapEntry(ext_name="output_directory", int_name="output_dir"),
    ParamMapEntry(ext_name="perform_attenuation", int_name="perform_atten"),
    ParamMapEntry(ext_name="raw_data_tof_cropping", int_name="raw_data_crop_vals"),
    ParamMapEntry(ext_name="first_cycle_run_no", int_name="run_in_range"),
    ParamMapEntry(ext_name="run_number", int_name="run_number"),
    ParamMapEntry(ext_name="sample_empty", int_name="sample_empty", optional=True),
    ParamMapEntry(ext_name="sample_empty_scale", int_name="sample_empty_scale"),
    ParamMapEntry(ext_name="spline_coefficient", int_name="spline_coefficient"),
    ParamMapEntry(ext_name="subtract_empty_can", int_name="subtract_empty_can"),
    ParamMapEntry(ext_name="suffix", int_name="suffix", optional=True),
    ParamMapEntry(ext_name="grouping_filename", int_name="grouping"),
    ParamMapEntry(ext_name="custom_grouping_filename", int_name="custom_grouping"),
    ParamMapEntry(ext_name="user_name", int_name="user_name"),
    ParamMapEntry(ext_name="vanadium_absorb_filename", int_name="van_absorb_file"),
    ParamMapEntry(ext_name="absorb_corrections_out_filename",
                  int_name="absorb_out_file",
                  optional=True),
    ParamMapEntry(ext_name="vanadium_tof_cropping", int_name="van_tof_cropping"),
    ParamMapEntry(ext_name="vanadium_normalisation", int_name="van_norm"),
    ParamMapEntry(ext_name="xye_filename", int_name="xye_filename")
]
attr_mapping.extend(PARAM_MAPPING)
